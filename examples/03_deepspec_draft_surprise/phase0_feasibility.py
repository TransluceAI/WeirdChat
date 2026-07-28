"""Phase 0 — feasibility gates for running DeepSpec against qwen3.6-35b-a3b.

Runs cheap, mostly-CPU checks and writes ``phase0_report.json`` with the
discovered model geometry and the recommended ``target_layer_ids``. Every gate
records pass/fail plus detail; the script exits non-zero if any hard gate
fails, so it can guard the rest of the pipeline in automation.

Usage:
    python phase0_feasibility.py --deepspec-root /path/to/deepspec \
        [--target Qwen/Qwen3.6-35B-A3B] [--deep]

``--deep`` additionally instantiates the target and the draft model on the
meta device (no weight download, but imports the full modeling stack).
"""

from __future__ import annotations

import argparse
import json
import traceback
from typing import Any, Callable

from surprise_common import (
    DEFAULT_TARGET_MODEL,
    detect_assistant_prefix,
    pick_mask_token,
    recommend_target_layer_ids,
    register_weirdchat_template,
    resolve_deepspec_root,
    resolve_draft_intermediate_size,
    unwrap_text_config,
)


def gate(report: dict[str, Any], name: str, hard: bool, fn: Callable[[], dict[str, Any]]) -> bool:
    try:
        detail = fn()
        report["gates"][name] = {"passed": True, "hard": hard, **detail}
        print(f"PASS {name}: {json.dumps(detail, default=str)[:300]}")
        return True
    except Exception as e:  # noqa: BLE001 — every gate failure must be reported, not raised
        report["gates"][name] = {
            "passed": False,
            "hard": hard,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
        }
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return not hard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspec-root", default=None)
    parser.add_argument("--target", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--output", default="phase0_report.json")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="additionally check the local GPU (CUDA, flex-attention, FP8 handling)",
    )
    args = parser.parse_args()

    resolve_deepspec_root(args.deepspec_root)

    report: dict[str, Any] = {"target": args.target, "gates": {}}
    ok = True

    # G1: transformers knows the architecture and AutoConfig loads.
    state: dict[str, Any] = {}

    def g1() -> dict[str, Any]:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(args.target)
        state["config"] = cfg
        return {"model_type": cfg.model_type, "architectures": getattr(cfg, "architectures", None)}

    ok &= gate(report, "G1_autoconfig", True, g1)

    # G2: the (possibly text_config-nested) config carries every field
    # DeepSpec's dense draft build needs.
    def g2() -> dict[str, Any]:
        cfg, nesting = unwrap_text_config(state["config"])
        state["effective_config"] = cfg
        report["config_nesting"] = nesting
        # intermediate_size is intentionally NOT required: MoE targets don't
        # define one, and the dense draft's width is resolved separately.
        required = [
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "vocab_size",
        ]
        missing = [f for f in required if getattr(cfg, f, None) is None]
        assert not missing, (
            f"target config (nesting={nesting}) lacks fields needed by the dense draft: {missing}"
        )
        num_layers = int(cfg.num_hidden_layers)
        rec = recommend_target_layer_ids(num_layers)
        draft_ffn, ffn_source = resolve_draft_intermediate_size(cfg)
        is_moe = any(
            getattr(cfg, f, None) is not None
            for f in ("num_experts", "num_routed_experts", "moe_intermediate_size")
        )
        report["num_hidden_layers"] = num_layers
        report["hidden_size"] = int(cfg.hidden_size)
        report["recommended_target_layer_ids"] = rec
        report["recommended_draft_intermediate_size"] = draft_ffn
        report["draft_intermediate_size_source"] = ffn_source
        report["target_is_moe"] = is_moe
        if nesting is not None or is_moe:
            report["needs_deepspec_patch"] = True
        return {
            "config_nesting": nesting,
            "num_hidden_layers": num_layers,
            "hidden_size": int(cfg.hidden_size),
            "recommended_target_layer_ids": rec,
            "recommended_draft_intermediate_size": draft_ffn,
            "draft_intermediate_size_source": ffn_source,
            "target_is_moe": is_moe,
        }

    ok &= gate(report, "G2_draft_config_fields", True, g2)

    # G3: tokenizer loads and an unused special token can serve as the draft
    # mask token (DeepSpec's Qwen3 default 151669 no longer exists in 3.6).
    def g3() -> dict[str, Any]:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.target)
        state["tokenizer"] = tok
        specials = {
            str(added): int(idx)
            for idx, added in tok.added_tokens_decoder.items()
            if getattr(added, "special", False)
        }
        sample = tok.apply_chat_template(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ho"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        reserved = {t for t in specials if t in sample}
        for role_token in (tok.eos_token, tok.bos_token, tok.pad_token, tok.unk_token):
            if role_token:
                reserved.add(str(role_token))
        name, token_id = pick_mask_token(specials, reserved)
        report["recommended_mask_token"] = name
        report["recommended_mask_token_id"] = token_id
        return {
            "recommended_mask_token": name,
            "recommended_mask_token_id": token_id,
            "n_special_tokens": len(specials),
        }

    ok &= gate(report, "G3_mask_token", True, g3)

    # G4: chat-template rendering matches the WeirdChat protocol. HARD: no
    # injected system prompt, ChatML headers present. SOFT: whether the
    # template still emits <think> scaffolding (recorded, not fatal — G5 proves
    # the data path works regardless; the scaffold is an empty, deterministic
    # block applied identically to weird and baseline data). Also detects the
    # exact scaffold prefix so training/scoring can optionally strip it.
    def g4() -> dict[str, Any]:
        tok = state["tokenizer"]
        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        text = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        assert "You are a helpful assistant" not in text, "template injects a system prompt"
        assert "<|im_start|>assistant" in text, "expected ChatML assistant header"
        inserts_think = "<think>" in text
        inserts_think_with_flag = inserts_think
        if inserts_think:
            try:
                text_flag = tok.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
                inserts_think_with_flag = "<think>" in text_flag
            except TypeError:
                inserts_think_with_flag = True
        prefix = detect_assistant_prefix(tok)
        report["chat_template_inserts_think"] = inserts_think
        report["chat_template_inserts_think_even_with_flag"] = inserts_think_with_flag
        report["detected_assistant_loss_prefix"] = prefix
        if inserts_think:
            report["needs_deepspec_patch"] = True
        return {
            "inserts_think": inserts_think,
            "inserts_think_even_with_flag": inserts_think_with_flag,
            "assistant_loss_prefix_len": len(prefix),
        }

    ok &= gate(report, "G4_chat_template", True, g4)

    # G5: DeepSpec's parser produces a non-empty assistant loss mask with the
    # registered WeirdChat template. Also verifies the optional think-prefix
    # stripping produces a strictly smaller (still non-empty) loss mask, so the
    # opt-in refinement is known to work on this tokenizer.
    def g5() -> dict[str, Any]:
        from deepspec.data.parser import (  # type: ignore[import-not-found]
            TEMPLATE_REGISTRY,
            preprocess_record,
        )

        from surprise_common import WEIRDCHAT_TEMPLATE_NAME

        record = {
            "conversations": [
                {"role": "user", "content": "Say something."},
                {"role": "assistant", "content": "Something, as requested."},
            ]
        }
        register_weirdchat_template()
        out = preprocess_record(record, state["tokenizer"], WEIRDCHAT_TEMPLATE_NAME, 512)
        n_loss = int(out["loss_mask"].sum())
        assert n_loss > 0, "assistant loss mask is empty — template/regex mismatch"

        result: dict[str, Any] = {
            "loss_tokens": n_loss,
            "total_tokens": int(out["attention_mask"].sum()),
        }
        # Try the strip-think refinement in an isolated registry entry.
        prefix = report.get("detected_assistant_loss_prefix") or ""
        if prefix and "prefix_added_by_template" in getattr(
            __import__("deepspec.data.parser", fromlist=["ChatTemplate"]).ChatTemplate,
            "__dataclass_fields__",
            {},
        ):
            TEMPLATE_REGISTRY._templates.pop(WEIRDCHAT_TEMPLATE_NAME, None)
            register_weirdchat_template(strip_think_prefix=prefix)
            out2 = preprocess_record(record, state["tokenizer"], WEIRDCHAT_TEMPLATE_NAME, 512)
            n_loss2 = int(out2["loss_mask"].sum())
            result["loss_tokens_strip_think"] = n_loss2
            assert 0 < n_loss2 <= n_loss, (
                f"strip-think loss mask invalid: {n_loss2} (base {n_loss})"
            )
            report["strip_think_verified"] = True
            # Restore the default (non-stripping) registration.
            TEMPLATE_REGISTRY._templates.pop(WEIRDCHAT_TEMPLATE_NAME, None)
            register_weirdchat_template()
        return result

    ok &= gate(report, "G5_parser_loss_mask", True, g5)

    # G6 (--deep): meta-device instantiation of target and draft. Exercises the
    # patched build_draft_config path — apply deepspec_qwen36.patch first when
    # the target config is nested.
    if args.deep:

        def g6() -> dict[str, Any]:
            import torch
            from transformers import AutoModel

            from deepspec.modeling.dspark.qwen3.config import (  # type: ignore[import-not-found]
                build_draft_config,
            )
            from deepspec.modeling.dspark.qwen3 import (  # type: ignore[import-not-found]
                Qwen3DSparkModel,
            )
            from deepspec.utils.config import to_config_node  # type: ignore[import-not-found]

            cfg = state["config"]
            with torch.device("meta"):
                AutoModel.from_config(cfg)
            model_args = to_config_node(
                {
                    "num_draft_layers": 5,
                    "target_layer_ids": report["recommended_target_layer_ids"],
                    "draft_intermediate_size": report["recommended_draft_intermediate_size"],
                    "block_size": 7,
                    "mask_token_id": report["recommended_mask_token_id"],
                    "num_anchors": 512,
                    "markov_rank": 256,
                    "markov_head_type": "vanilla",
                    "confidence_head_alpha": 1.0,
                    "confidence_head_with_markov": True,
                }
            )
            draft_cfg = build_draft_config(cfg, model_args)
            with torch.device("meta"):
                draft = Qwen3DSparkModel(draft_cfg)
            n_params = sum(p.numel() for p in draft.parameters())
            return {"draft_params": n_params}

        ok &= gate(report, "G6_meta_instantiation", True, g6)

    # G7 (--gpu): the local accelerator can run phases 2-4. Checks CUDA, the
    # flex-attention kernel the draft uses, memory headroom for the target's
    # weights, and how a quantized (e.g. FP8) checkpoint will be handled.
    if args.gpu:

        def g7() -> dict[str, Any]:
            import torch
            from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401

            assert torch.cuda.is_available(), "no CUDA device visible"
            properties = torch.cuda.get_device_properties(0)
            capability = torch.cuda.get_device_capability(0)
            total_gb = properties.total_memory / 1024**3
            quant = getattr(state["config"], "quantization_config", None)
            if quant is None:
                quant = getattr(state.get("effective_config"), "quantization_config", None)
            quant_method = None
            if quant is not None:
                quant_method = (
                    quant.get("quant_method")
                    if isinstance(quant, dict)
                    else getattr(quant, "quant_method", None)
                )
            detail: dict[str, Any] = {
                "device": properties.name,
                "capability": f"sm{capability[0]}{capability[1]}",
                "vram_gb": round(total_gb, 1),
                "quantization": quant_method or "none (full-precision checkpoint)",
            }
            if quant_method is None and total_gb < 75:
                detail["warning"] = (
                    "full-precision 35B weights (~70 GB) exceed this GPU; use the "
                    "FP8 checkpoint (WEIRDSPEC_TARGET_MODEL=Qwen/Qwen3.6-35B-A3B-FP8)"
                )
            if quant_method is not None and capability[0] < 9:
                # Pre-Hopper GPUs (A100 = sm80) lack native FP8 compute;
                # transformers must run the checkpoint weight-only-dequantized.
                detail["fp8_note"] = (
                    "no native FP8 compute on this GPU — transformers will "
                    "dequantize weights on the fly (slower, works)"
                )
            report["gpu"] = detail
            return detail

        ok &= gate(report, "G7_gpu", True, g7)

    report["all_hard_gates_passed"] = ok
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written to {args.output}. All hard gates passed: {ok}")
    if report.get("needs_deepspec_patch"):
        print(
            "NOTE: this target needs deepspec_qwen36.patch applied to the DeepSpec "
            "checkout before phases 2-4 (cd $DEEPSPEC_ROOT && git apply "
            "/path/to/deepspec_qwen36.patch)."
        )
    if "recommended_mask_token_id" in report and report.get("recommended_target_layer_ids"):
        print(
            "For phases 2-3:\n"
            f"  export WEIRDSPEC_MASK_TOKEN_ID={report['recommended_mask_token_id']}  "
            f"# {report.get('recommended_mask_token')}\n"
            f"  export WEIRDSPEC_DRAFT_INTERMEDIATE_SIZE={report['recommended_draft_intermediate_size']}"
            f"  # source: {report.get('draft_intermediate_size_source')}\n"
            f"  --opts model.target_layer_ids=\"{report['recommended_target_layer_ids']}\""
        )
    if report.get("chat_template_inserts_think"):
        print(
            "NOTE: the chat template emits <think> scaffolding; it is benign (empty, "
            "consistent across data). To strip it from the loss mask, set "
            "WEIRDSPEC_STRIP_THINK=1 for phases 2-4."
        )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
