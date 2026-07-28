"""Phase 1b (Colab variant) — baseline regeneration via OpenRouter.

Replaces local serving of the 35B target for the baseline corpus: assistant
answers are regenerated through OpenRouter under the WeirdChat protocol
(temperature 1.0, no system prompt, reasoning disabled, 1024 new tokens).
Multi-turn rows are regenerated turn by turn, each answer conditioning the
next, exactly like DeepSpec's local regen script.

Rows containing system turns are skipped (the protocol has none). Progress is
resumable: already-written output/error lines are counted and skipped on
restart. The deterministic held-out split (scoring null) happens at the end.

Caveat (documented in the README): OpenRouter providers may serve a different
quantization than the FP8 checkpoint used for hidden-state capture — the same
replication gap WeirdChat itself measures. Acceptable for the Colab-scale run.

Usage:
    export OPENROUTER_API_KEY=...
    python phase1b_openrouter.py \
        --input data/perfectblend_train.jsonl --output-dir data \
        [--max-samples 3000] [--concurrency 16] [--heldout-frac 0.02]

Any OpenAI-compatible endpoint works via --base-url / --api-key-env /
--extra-body — but only if it serves the *exact* subject model: the baseline
must be on-policy text from qwen3.6-35b-a3b itself, or the draft learns the
wrong model's "normal" and the surprise scores measure the wrong thing.

    # Groq (only if the exact model is in their catalog):
    python phase1b_openrouter.py ... \
        --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY \
        --model <groq model id> --extra-body '{}'

    # local SGLang serving the FP8 checkpoint:
    python phase1b_openrouter.py ... \
        --base-url http://127.0.0.1:30000/v1 --api-key-env NONE \
        --model Qwen/Qwen3.6-35B-A3B-FP8 \
        --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import anyio
from openai import AsyncOpenAI

from surprise_common import (
    SUBJECT_MODEL_SLUG,
    WEIRDCHAT_MAX_NEW_TOKENS,
    WEIRDCHAT_TEMPERATURE,
    is_heldout_line,
    prepare_regen_conversations,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


async def regen_row(client: AsyncOpenAI, args: Any, row: dict[str, Any]) -> dict[str, Any]:
    user_turns = prepare_regen_conversations(row.get("conversations"))
    if user_turns is None:
        return {"status": "skipped", "reason": "system turn or invalid structure"}
    messages: list[dict[str, str]] = []
    try:
        for user_turn in user_turns:
            messages.append(user_turn)
            completion = await client.chat.completions.create(
                model=args.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body=json.loads(args.extra_body),
            )
            reply = completion.choices[0].message.content or ""
            if not reply.strip():
                return {"status": "error", "error": "empty completion"}
            messages.append({"role": "assistant", "content": reply})
    except Exception as e:  # noqa: BLE001 — any provider error fails just this row
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    return {"status": "ok", "conversations": messages}


async def run(args: Any) -> None:
    from weirdchat.concurrency import run_bounded

    out_path = os.path.join(args.output_dir, "baseline_regen_full.jsonl")
    err_path = os.path.join(args.output_dir, "baseline_regen_error.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break

    done = count_lines(out_path) + count_lines(err_path)
    todo = rows[done:]
    print(f"{len(rows)} source rows, {done} already processed, {len(todo)} to go")
    if todo:
        api_key = "EMPTY" if args.api_key_env == "NONE" else os.environ[args.api_key_env]
        client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=api_key,
            timeout=180.0,
            max_retries=3,
        )
        # Batches keep resume granularity: results are appended in source order
        # after each bounded batch completes.
        batch_size = max(args.concurrency * 8, 64)
        with open(out_path, "a", encoding="utf-8") as out_f, open(
            err_path, "a", encoding="utf-8"
        ) as err_f:
            for start in range(0, len(todo), batch_size):
                batch = todo[start : start + batch_size]
                futures = [regen_row(client, args, row) for row in batch]
                results = await run_bounded(futures, args.concurrency, "regen")
                for result in results:
                    if result["status"] == "ok":
                        out_f.write(
                            json.dumps(
                                {"conversations": result["conversations"]},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    else:
                        err_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                err_f.flush()
                print(f"processed {min(start + batch_size, len(todo))}/{len(todo)}")

    # Deterministic held-out split for the scoring null distribution.
    train_path = os.path.join(args.output_dir, "baseline_train.jsonl")
    heldout_path = os.path.join(args.output_dir, "baseline_heldout.jsonl")
    n_train = n_heldout = 0
    with open(out_path, "r", encoding="utf-8") as src, open(
        train_path, "w", encoding="utf-8"
    ) as train_f, open(heldout_path, "w", encoding="utf-8") as heldout_f:
        for line in src:
            line = line.strip()
            if not line:
                continue
            if is_heldout_line(line, args.heldout_frac):
                heldout_f.write(line + "\n")
                n_heldout += 1
            else:
                train_f.write(line + "\n")
                n_train += 1
    print(f"split: {n_train} train / {n_heldout} heldout")
    print(f"  {train_path}\n  {heldout_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="perfectblend_train.jsonl")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--model", default=SUBJECT_MODEL_SLUG)
    parser.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help='env var holding the API key; "NONE" for keyless local endpoints',
    )
    parser.add_argument(
        "--extra-body",
        default='{"reasoning": {"enabled": false}}',
        help="JSON merged into each request body (reasoning-off flag varies by provider)",
    )
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=WEIRDCHAT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=WEIRDCHAT_MAX_NEW_TOKENS)
    parser.add_argument("--heldout-frac", type=float, default=0.02)
    args = parser.parse_args()
    anyio.run(run, args)


if __name__ == "__main__":
    main()
