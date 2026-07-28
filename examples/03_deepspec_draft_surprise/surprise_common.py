"""Shared helpers for the DeepSpec draft-surprise pipeline.

This module is imported by the phase scripts. It deliberately keeps the parts
that need heavy dependencies (torch, transformers, deepspec) behind function
calls so that the pure logic (layer-id recommendation, conversation conversion,
aggregation math) stays importable and testable in the plain `weirdchat`
environment.

DeepSpec is consumed as an external checkout: point ``DEEPSPEC_ROOT`` (env var
or ``--deepspec-root``) at a clone of the DeepSpec repository. Nothing in the
DeepSpec tree is modified — the extra chat template and the new target config
are injected from here, which DeepSpec's ``config_path`` mechanism supports.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# The WeirdChat subject model this pipeline targets, and the sampling protocol
# the dataset was generated with (see the WeirdChat blog post appendix):
# temperature 1.0, no system prompt, reasoning disabled, max 1024 new tokens.
SUBJECT_MODEL_SLUG = "qwen/qwen3.6-35b-a3b"
DEFAULT_TARGET_MODEL = os.environ.get("WEIRDSPEC_TARGET_MODEL", "Qwen/Qwen3.6-35B-A3B")
WEIRDCHAT_TEMPERATURE = 1.0
WEIRDCHAT_MAX_NEW_TOKENS = 1024

# Name of the chat template this pipeline registers in DeepSpec's registry.
# It matches DeepSpec's builtin "qwen" template except that no system prompt is
# injected, mirroring how WeirdChat sampled its subject models.
WEIRDCHAT_TEMPLATE_NAME = "qwen_weirdchat"


def resolve_deepspec_root(cli_value: str | None = None) -> str:
    """Locate the DeepSpec checkout and make it importable."""
    root = cli_value or os.environ.get("DEEPSPEC_ROOT")
    if not root:
        raise SystemExit(
            "DeepSpec checkout not found: pass --deepspec-root or set DEEPSPEC_ROOT."
        )
    root = os.path.abspath(root)
    marker = os.path.join(root, "deepspec", "data", "parser.py")
    if not os.path.isfile(marker):
        raise SystemExit(f"{root} does not look like a DeepSpec checkout ({marker} missing).")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def detect_assistant_prefix(tokenizer, header: str = "<|im_start|>assistant\n") -> str:
    """Return the scaffold the chat template emits before assistant content.

    Renders a sentinel assistant turn and extracts whatever the template puts
    between the assistant header and the actual content — e.g. an empty
    ``<think>\\n\\n</think>\\n\\n`` block for thinking-by-default Qwen models.
    Empty string if the template adds nothing. Uses only the tokenizer, so it
    runs without model weights (Colab-friendly).
    """
    sentinel = "WEIRDSPEC_CONTENT_SENTINEL"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=False,
    )
    start = text.rfind(header)
    if start < 0:
        return ""
    after = text[start + len(header) :]
    end = after.find(sentinel)
    if end < 0:
        return ""
    return after[:end]


def register_weirdchat_template(strip_think_prefix: str | None = None) -> None:
    """Idempotently register the system-prompt-free Qwen template in DeepSpec.

    With `deepspec_qwen36.patch` applied, the template also forces
    ``enable_thinking=False`` at render time. On an unpatched DeepSpec the
    ``enable_thinking``/``prefix_added_by_template`` fields don't exist and are
    skipped.

    ``strip_think_prefix`` (the scaffold from :func:`detect_assistant_prefix`)
    is optional: when given and supported, the registered template strips that
    prefix from the assistant loss mask so training/scoring focus on real
    content rather than the empty ``<think></think>`` scaffold. Off by default
    to keep the exact rendering that phase 0's parser gate validated.
    """
    from deepspec.data.parser import TEMPLATE_REGISTRY, ChatTemplate  # type: ignore[import-not-found]

    try:
        TEMPLATE_REGISTRY.get(WEIRDCHAT_TEMPLATE_NAME)
        return
    except KeyError:
        pass
    fields = getattr(ChatTemplate, "__dataclass_fields__", {})
    kwargs: dict[str, Any] = {}
    if "enable_thinking" in fields:
        kwargs["enable_thinking"] = False
    if strip_think_prefix and "prefix_added_by_template" in fields:
        kwargs["assistant_loss_prefix"] = strip_think_prefix
        kwargs["prefix_added_by_template"] = True
    TEMPLATE_REGISTRY.register(
        WEIRDCHAT_TEMPLATE_NAME,
        ChatTemplate(
            assistant_header="<|im_start|>assistant\n",
            user_header="<|im_start|>user\n",
            system_prompt=None,
            end_of_turn_token="<|im_end|>\n",
            **kwargs,
        ),
    )


def unwrap_text_config(config: Any) -> tuple[Any, str | None]:
    """Resolve nested wrapper configs (e.g. Qwen3.5/3.6 MoE, Gemma4).

    Returns ``(effective_config, nesting)`` where nesting is ``"text_config"``
    when the transformer geometry lives one level down, else ``None``.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(config, "num_hidden_layers", None) is None:
        return text_config, "text_config"
    return config, None


# Preference order for repurposing a special token as the draft mask token
# (DeepSpec's Qwen3 configs use `<|fim_pad|>` = 151669, which no longer exists
# in the qwen3.6 tokenizer).
_MASK_TOKEN_KEYWORDS = ("fim_pad", "mask", "pad", "unused", "reserved", "fim")


def resolve_draft_intermediate_size(config: Any) -> tuple[int, str]:
    """Pick a dense FFN width for the draft, returning (size, source).

    The DSpark draft is always dense, so it needs a scalar ``intermediate_size``.
    Dense targets carry one. For MoE targets the per-expert
    ``moe_intermediate_size`` alone can be tiny (fine-grained experts, e.g. 512
    on qwen3.6-35b-a3b) — a draft that thin would be crippled. Use the *active*
    capacity (per-expert width x experts per token) when it is at least the
    conventional dense 8/3·hidden width, else fall back to that dense width.
    """
    dense = getattr(config, "intermediate_size", None)
    if dense is not None:
        return int(dense), "intermediate_size"
    hidden = int(config.hidden_size)
    derived = round(hidden * 8 / 3 / 128) * 128
    moe = getattr(config, "moe_intermediate_size", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    if moe is not None and top_k:
        active = int(moe) * int(top_k)
        if active >= derived:
            return active, "moe_active_capacity"
    return derived, "derived_from_hidden_size"


def pick_mask_token(special_tokens: dict[str, int], reserved: set[str]) -> tuple[str, int]:
    """Pick an unused special token to serve as the draft mask token.

    ``special_tokens`` maps token string -> id; ``reserved`` holds tokens the
    chat template (or eos/bos/pad roles) actually use and which therefore must
    not be repurposed.
    """
    candidates = {t: i for t, i in special_tokens.items() if t not in reserved}
    if not candidates:
        raise ValueError("tokenizer has no unused special token to repurpose as mask token")
    for keyword in _MASK_TOKEN_KEYWORDS:
        matches = sorted((t for t in candidates if keyword in t.lower()), key=candidates.get)
        if matches:
            return matches[0], candidates[matches[0]]
    token = sorted(candidates, key=candidates.get)[0]
    return token, candidates[token]


def recommend_target_layer_ids(num_target_layers: int, k: int = 5) -> list[int]:
    """Evenly spaced draft input layers, mirroring DeepSpec's shipped configs.

    DeepSpec's Qwen3-4B config uses [1, 9, 17, 25, 33] for 36 layers: start at
    layer 1, end a couple of layers before the top (the final layer is rejected
    by ``assert_no_final_target_layer``), evenly spaced.
    """
    if num_target_layers < k + 3:
        raise ValueError(f"target has too few layers ({num_target_layers}) for k={k}")
    first = 1
    last = num_target_layers - 3
    ids = [round(first + (last - first) * i / (k - 1)) for i in range(k)]
    # De-duplicate while preserving strict monotonicity (tiny models).
    out: list[int] = []
    for layer_id in ids:
        if not out or layer_id > out[-1]:
            out.append(layer_id)
    return out


def messages_to_conversations(messages: list[Any]) -> list[dict[str, str]]:
    """Convert WeirdChat ``Message`` rows (or dicts) to DeepSpec's format.

    DeepSpec requires a ``conversations`` list of {role, content} dicts that
    starts with a user turn. WeirdChat transcripts contain only user/assistant
    turns, but we validate rather than assume.
    """
    out: list[dict[str, str]] = []
    for m in messages:
        role = m["role"] if isinstance(m, dict) else m.role
        content = m["content"] if isinstance(m, dict) else m.content
        if role not in ("user", "assistant"):
            raise ValueError(f"unsupported role for DeepSpec conversion: {role!r}")
        out.append({"role": role, "content": content})
    if not out or out[0]["role"] != "user":
        raise ValueError("conversation must start with a user turn")
    return out


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Aggregation math (pure python so it is unit-testable without torch).
# ---------------------------------------------------------------------------


def summarize_token_scores(nll_draft: list[float], nll_target: list[float]) -> dict[str, float]:
    """Per-sample aggregates of token-level surprise scores.

    ``excess`` is draft NLL minus target-proxy NLL: it controls for tokens that
    are merely high-entropy for the target itself (temperature-1 sampling tails)
    and isolates tokens that are atypical *relative to normal behavior*.
    """
    assert len(nll_draft) == len(nll_target)
    n = len(nll_draft)
    if n == 0:
        return {"n_scored_tokens": 0.0}
    excess = [d - t for d, t in zip(nll_draft, nll_target)]
    peak_idx = max(range(n), key=lambda i: excess[i])
    return {
        "n_scored_tokens": float(n),
        "mean_nll_draft": sum(nll_draft) / n,
        "max_nll_draft": max(nll_draft),
        "mean_nll_target": sum(nll_target) / n,
        "mean_excess": sum(excess) / n,
        "max_excess": max(excess),
        "peak_excess_token_idx": float(peak_idx),
    }


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation without scipy. Returns None if undefined."""
    assert len(xs) == len(ys)
    n = len(xs)
    if n < 3:
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5
