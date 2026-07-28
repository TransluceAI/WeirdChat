"""Phase 4 — teacher-forced draft-surprise scoring of transcripts.

For every assistant token in every input conversation this records how well
the baseline-trained DSpark draft predicts it, given the target model's hidden
states — token-level "surprise" signals for interpretation:

- ``nll_draft``  — -log p_draft(actual token) at block offset 0 (next token).
- ``nll_target`` — -log p_target-proxy(actual token), from the draft head
  applied to the target's last hidden state (exactly the ``aligned_target_logits``
  the training loss distills against). High values mean the token was unlikely
  even for the target — the temperature-1 sampling tail.
- ``conf_next``  — sigmoid of the confidence head's offset-0 prediction: the
  draft's own calibrated estimate that it matches the target here.

``nll_draft - nll_target`` (the *excess*) isolates tokens that are atypical
relative to the model's normal behavior rather than merely high-entropy.

Anchors are enumerated exhaustively and deterministically: DeepSpec's anchor
sampler keeps every valid candidate whenever ``num_anchors`` >= the candidate
count, so scoring in loss-mask windows with overlap ``block_size`` covers each
anchor exactly once without touching DeepSpec internals. Scored positions are
all assistant tokens except the first token of each assistant turn (an anchor
must itself carry loss mask — the same constraint training has).

Run this twice: on the WeirdChat export and on the baseline held-out slice
(the null distribution). Requires a GPU (the DSpark forward uses
flex-attention) and the DeepSpec environment.

Usage:
    python phase4_score_surprise.py --deepspec-root $DEEPSPEC_ROOT \
        --draft ~/checkpoints/weirdspec/dspark_block7_qwen36_35b_a3b/step_latest \
        --data data/weird_transcripts.jsonl --output data/weird_scores.jsonl \
        [--target Qwen/Qwen3.6-35B-A3B] [--target-device-map auto] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from surprise_common import (
    DEFAULT_TARGET_MODEL,
    WEIRDCHAT_TEMPLATE_NAME,
    register_weirdchat_template,
    resolve_deepspec_root,
    summarize_token_scores,
)


def resolve_backbone(target_model):
    """Find the decoder stack, including nested wrappers (Qwen3.6 MoE, Gemma4)."""
    for holder in (target_model, getattr(target_model, "model", None)):
        language_model = getattr(holder, "language_model", None) if holder else None
        if language_model is not None and hasattr(language_model, "layers"):
            return language_model
    backbone = getattr(target_model, "model", target_model)
    assert hasattr(backbone, "layers"), (
        f"cannot locate decoder layers on {type(target_model).__name__}"
    )
    return backbone


def capture_target_states(target_model, input_ids, attention_mask, layer_ids, out_device):
    """Forward the target once, capturing hidden states at ``layer_ids``.

    Mirrors DeepSpec's ``prepare_target_cache.run_target_forward_with_hooks``
    but moves captures to ``out_device`` so it also works under device_map
    sharding of a large target, and resolves nested backbones. The final
    hidden state is taken from the backbone's closing norm hook when the
    wrapper's output lacks ``last_hidden_state``.
    """
    import torch

    backbone = resolve_backbone(target_model)
    captured: dict[int, Any] = {}
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer_id] = t.detach().to(out_device)

        return hook

    norm_key = 10**9  # sentinel key for the final-norm capture
    try:
        if -1 in layer_ids:
            handles.append(backbone.embed_tokens.register_forward_hook(make_hook(-1)))
        for layer_id in layer_ids:
            if layer_id >= 0:
                handles.append(backbone.layers[layer_id].register_forward_hook(make_hook(layer_id)))
        if hasattr(backbone, "norm"):
            handles.append(backbone.norm.register_forward_hook(make_hook(norm_key)))
        with torch.no_grad():
            out = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
            last_hidden = getattr(out, "last_hidden_state", None)
            if last_hidden is not None:
                last_hidden = last_hidden.detach().to(out_device)
            else:
                assert norm_key in captured, (
                    "target output has no last_hidden_state and the backbone has no "
                    "final norm to hook"
                )
                last_hidden = captured[norm_key]
            stacked = torch.cat([captured[i] for i in layer_ids], dim=-1)
    finally:
        for h in handles:
            h.remove()
    return stacked, last_hidden


def anchor_candidates(loss_mask: list[int], start: int, end: int) -> list[int]:
    """Valid anchor positions in [start, end]: loss_mask[a] and loss_mask[a+1]."""
    n = len(loss_mask)
    return [
        a
        for a in range(max(start, 0), min(end, n - 2) + 1)
        if loss_mask[a] and loss_mask[a + 1]
    ]


def score_sample(draft_model, input_ids, loss_mask, target_hidden, target_last_hidden, *,
                 block_size: int, anchor_chunk: int):
    """Teacher-forced scoring of one sample. Returns per-position records."""
    import torch

    device = input_ids.device
    seq_len = int(input_ids.shape[1])
    lm_list = [int(v) for v in loss_mask[0].tolist()]
    all_candidates = anchor_candidates(lm_list, 0, seq_len - 2)
    per_position: dict[int, dict[str, float]] = {}
    offset_nll_sums = [0.0] * block_size
    offset_nll_counts = [0] * block_size

    for chunk_start in range(0, len(all_candidates), anchor_chunk):
        chunk = all_candidates[chunk_start : chunk_start + anchor_chunk]
        lo, hi = chunk[0], chunk[-1]
        # Window the loss mask so DeepSpec's sampler sees exactly this chunk's
        # anchors, while labels up to block_size past the last anchor stay
        # loss-masked for eval_mask.
        windowed = torch.zeros_like(loss_mask)
        win_end = min(hi + block_size + 1, seq_len)
        windowed[0, lo:win_end] = loss_mask[0, lo:win_end]
        window_candidates = anchor_candidates(
            [int(v) for v in windowed[0].tolist()], 0, seq_len - 2
        )
        # Anchors beyond `hi` can become valid through the overlap; they are
        # rescored by the next chunk, so drop them here.
        keep_upto = sum(1 for a in window_candidates if a <= hi)
        assert window_candidates[:keep_upto] == chunk, "anchor enumeration diverged"

        draft_model.num_anchors = len(window_candidates)
        with torch.no_grad():
            out = draft_model(
                input_ids=input_ids,
                target_hidden_states=target_hidden,
                loss_mask=windowed,
                target_last_hidden_states=target_last_hidden,
            )
        draft_logp = torch.log_softmax(out.draft_logits.float(), dim=-1)
        target_logp = torch.log_softmax(out.aligned_target_logits.float(), dim=-1)
        token_lp_draft = torch.gather(draft_logp, -1, out.target_ids.unsqueeze(-1)).squeeze(-1)
        token_lp_target = torch.gather(target_logp, -1, out.target_ids.unsqueeze(-1)).squeeze(-1)
        conf = (
            torch.sigmoid(out.confidence_pred) if out.confidence_pred is not None else None
        )
        eval_mask = out.eval_mask[0]
        keep = out.block_keep_mask[0]

        for i in range(keep_upto):
            if not bool(keep[i]):
                continue
            a = window_candidates[i]
            for j in range(block_size):
                if not bool(eval_mask[i, j]):
                    break
                nll_d = float(-token_lp_draft[0, i, j])
                nll_t = float(-token_lp_target[0, i, j])
                offset_nll_sums[j] += nll_d
                offset_nll_counts[j] += 1
                if j == 0:
                    per_position[a + 1] = {
                        "nll_draft": nll_d,
                        "nll_target": nll_t,
                        "conf_next": float(conf[0, i, 0]) if conf is not None else float("nan"),
                    }
    del target_hidden, target_last_hidden
    return per_position, offset_nll_sums, offset_nll_counts, seq_len


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspec-root", default=None)
    parser.add_argument("--draft", required=True, help="DSpark draft checkpoint path")
    parser.add_argument("--target", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--data", required=True, help="JSONL with `conversations` rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--anchor-chunk", type=int, default=512)
    parser.add_argument("--device", default="cuda:0", help="device for the draft model")
    parser.add_argument(
        "--target-device-map",
        default=None,
        help='e.g. "auto" to shard a large target across GPUs; default: same device as draft',
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    resolve_deepspec_root(args.deepspec_root)

    import torch
    from transformers import AutoModel, AutoTokenizer

    from deepspec.data.jsonl_dataset import JsonLineDataset  # type: ignore[import-not-found]
    from deepspec.data.parser import preprocess_record  # type: ignore[import-not-found]
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # type: ignore[import-not-found]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    # Must match the rendering used to build the target cache the draft trained
    # against — honor WEIRDSPEC_STRIP_THINK the same way the training config does.
    strip_prefix = None
    if os.environ.get("WEIRDSPEC_STRIP_THINK") == "1":
        from surprise_common import detect_assistant_prefix

        strip_prefix = detect_assistant_prefix(tokenizer) or None
    register_weirdchat_template(strip_think_prefix=strip_prefix)

    draft_model = (
        Qwen3DSparkModel.from_pretrained(args.draft, dtype=torch.bfloat16).to(device).eval()
    )
    block_size = int(draft_model.block_size)
    layer_ids = [int(i) for i in draft_model.target_layer_ids]
    print(f"draft loaded: block_size={block_size} target_layer_ids={layer_ids}")

    from transformers import AutoConfig

    target_auto_config = AutoConfig.from_pretrained(args.target)
    target_kwargs: dict[str, Any] = {"attn_implementation": "sdpa"}
    # Quantized checkpoints (e.g. the FP8 repo the dataset was generated from)
    # carry their own dtype plan — forcing bf16 would conflict with it.
    if getattr(target_auto_config, "quantization_config", None) is not None:
        target_kwargs["dtype"] = "auto"
    else:
        target_kwargs["dtype"] = torch.bfloat16
    if args.target_device_map:
        target_kwargs["device_map"] = args.target_device_map
    try:
        target_model = AutoModel.from_pretrained(args.target, **target_kwargs)
    except ValueError:
        # Architectures outside the AutoModel mapping (e.g. the qwen3.6 MoE
        # *ForConditionalGeneration wrapper) load via their concrete class.
        import transformers as transformers_module

        architecture = str(target_auto_config.architectures[0])
        target_model = getattr(transformers_module, architecture).from_pretrained(
            args.target, **target_kwargs
        )
    if not args.target_device_map:
        target_model = target_model.to(device)
    target_model = target_model.eval()

    dataset = JsonLineDataset(data_paths=[args.data])
    n_total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n_written = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for index in range(n_total):
            record = dataset[index]
            parsed = preprocess_record(
                record, tokenizer, WEIRDCHAT_TEMPLATE_NAME, args.max_length
            )
            input_ids = parsed["input_ids"].unsqueeze(0).to(device)
            loss_mask = parsed["loss_mask"].unsqueeze(0).to(device)
            attention_mask = parsed["attention_mask"].unsqueeze(0).to(device)
            if int(loss_mask.sum()) < 2:
                print(f"[{index}] skipped: <2 loss tokens")
                continue

            target_hidden, target_last = capture_target_states(
                target_model, input_ids, attention_mask, layer_ids, device
            )
            per_position, off_sums, off_counts, seq_len = score_sample(
                draft_model,
                input_ids,
                loss_mask,
                target_hidden.to(torch.bfloat16),
                target_last.to(torch.bfloat16),
                block_size=block_size,
                anchor_chunk=args.anchor_chunk,
            )
            positions = sorted(per_position)
            ids_list = input_ids[0].tolist()
            row = {
                "index": index,
                "id": record.get("id"),
                "seq_len": seq_len,
                "scored_positions": positions,
                "token_ids": [ids_list[t] for t in positions],
                "token_strs": [tokenizer.decode([ids_list[t]]) for t in positions],
                "nll_draft": [round(per_position[t]["nll_draft"], 4) for t in positions],
                "nll_target": [round(per_position[t]["nll_target"], 4) for t in positions],
                "conf_next": [round(per_position[t]["conf_next"], 4) for t in positions],
                "offset_mean_nll_draft": [
                    round(s / c, 4) if c else None for s, c in zip(off_sums, off_counts)
                ],
                "aggregates": {
                    k: round(v, 4)
                    for k, v in summarize_token_scores(
                        [per_position[t]["nll_draft"] for t in positions],
                        [per_position[t]["nll_target"] for t in positions],
                    ).items()
                },
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            n_written += 1
            if n_written % 10 == 0 or index == n_total - 1:
                print(f"[{index + 1}/{n_total}] scored, {n_written} written")
    dataset.close()
    print(f"done: {n_written} samples -> {args.output}")


if __name__ == "__main__":
    main()
