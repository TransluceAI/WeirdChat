# Draft-surprise interpretation of the WeirdChat Qwen patterns with DeepSpec

This pipeline uses [DeepSpec](https://github.com/erikiss/deepspec) — a training
stack for speculative-decoding draft models (DSpark/DFlash/Eagle3) — as an
*interpretation instrument* for the `qwen/qwen3.6-35b-a3b` slice of WeirdChat.

**Idea.** A DSpark draft trained on the target model's *normal* outputs is a
small, learned model of "what Qwen typically does" — including a confidence
head that predicts, per token, whether the draft will match the target. Running
that draft teacher-forced over WeirdChat transcripts turns speculative-decoding
acceptance into a token-level *atypicality signal*:

- `nll_draft` — how surprised the normal-behavior draft is by each actual token;
- `nll_target` — how unlikely the token was for the target itself (its
  temperature-1 sampling tail), estimated from the same aligned-logits path the
  DSpark training loss distills against;
- `excess = nll_draft − nll_target` — surprise *beyond* ordinary sampling
  entropy: the interpretation signal. High-excess spans mark where a transcript
  departs from the model's usual behavior.

This complements WeirdChat's judge-based metrics (binary match, Elo) with a
continuous, judge-free, token-localized measure — e.g. "at which token does the
model tip into Russian / self-harm instructions?", and "does draft surprise
correlate with the unexpectedness Elo?".

**Quickstart:** [`quickstart_colab.ipynb`](quickstart_colab.ipynb) runs the
CPU phases (0 and 1a) directly in Google Colab, reading `HF_TOKEN` from Colab
Secrets.

## Prerequisites

- A DeepSpec checkout (`DEEPSPEC_ROOT`) with its `requirements.txt` installed.
  The chat template and target config are injected from this directory via
  DeepSpec's `config_path` mechanism; for this target DeepSpec additionally
  needs a small patch (phase 0 confirmed the qwen3.6 MoE wrapper nests its
  config and the template needs `enable_thinking=False`):

  ```bash
  cd $DEEPSPEC_ROOT && git apply /path/to/examples/03_deepspec_draft_surprise/deepspec_qwen36.patch
  ```
- The WeirdChat environment (`uv sync` at the repo root) for phases 1 and 5.
- GPUs for phases 1b–4 (the target is a ~35B-A3B MoE; DeepSpec's defaults
  assume one 8-GPU node). Phase 0 and 5 are CPU-cheap.

## Phases

| phase | script | needs | produces |
|-------|--------|-------|----------|
| 0 | `phase0_feasibility.py` | CPU, HF config access | `phase0_report.json` (gates, layer geometry, recommended `target_layer_ids`) |
| 1a | `phase1_export_weirdchat.py` | WeirdChat env | `weird_transcripts.jsonl` + `weird_meta.jsonl` |
| 1b | `phase1_baseline.sh` | served target model | `baseline_train.jsonl` + `baseline_heldout.jsonl` |
| 2 | DeepSpec `prepare_target_cache.py` with `dspark_qwen36_35b_a3b.py` | GPUs, storage | target cache |
| 3 | DeepSpec `train.sh` with the same config | GPUs | DSpark draft checkpoint |
| 4 | `phase4_score_surprise.py` (run on weird + baseline-heldout) | 1 node | `weird_scores.jsonl`, `baseline_scores.jsonl` |
| 5 | `phase5_analyze.py` | CPU | `report.md` |

### Phase 0 — feasibility gates

```bash
python phase0_feasibility.py --deepspec-root $DEEPSPEC_ROOT [--deep]
```

Hard gates: the architecture loads via `AutoConfig`/`AutoModel` under
DeepSpec's pinned `transformers`; the (possibly `text_config`-nested) config
carries every field the dense draft needs (**not** `intermediate_size` — MoE
targets have none, so the draft's dense FFN width is resolved separately); an
unused special token is *discovered* to serve as the draft mask token; the
chat template injects **no** system prompt and has ChatML headers; DeepSpec's
parser produces a non-empty assistant loss mask. Whether the template emits
`<think>` scaffolding is recorded but **not** fatal (it is empty and
consistent across data; the loss-mask gate proves the data path works).

Verified findings for `Qwen/Qwen3.6-35B-A3B` (Colab runs; all hard gates now
pass): `model_type=qwen3_5_moe`, architecture
`Qwen3_5MoeForConditionalGeneration` with nested `text_config`, 40 layers
(→ `target_layer_ids=[1, 10, 19, 28, 37]`), no dense `intermediate_size` and
fine-grained 512-wide experts (so the draft FFN width is resolved from active
capacity / the 8/3·hidden rule instead), mask token `151669` absent
(→ `<|vision_pad|>`=248055), and a `<think>` scaffold the template emits even
with `enable_thinking=False`. All are handled by `deepspec_qwen36.patch` plus
the phase-0 recommendations. The report emits, for phases 2–3:
`recommended_target_layer_ids` (→ `--opts model.target_layer_ids`),
`recommended_draft_intermediate_size` (→ `WEIRDSPEC_DRAFT_INTERMEDIATE_SIZE`),
and `recommended_mask_token_id` (→ `WEIRDSPEC_MASK_TOKEN_ID`). Set
`WEIRDSPEC_STRIP_THINK=1` to strip the think scaffold from the loss mask
(optional; phase 0 verifies it works on this tokenizer).

### Phase 1 — data

```bash
# 1a: the interpretation set (defaults to judge-matched transcripts only)
uv run python phase1_export_weirdchat.py --output-dir data/

# 1b: the baseline corpus, regenerated under the WeirdChat protocol
#     (temperature 1.0, no system prompt, thinking disabled)
DEEPSPEC_ROOT=... TARGET_MODEL=Qwen/Qwen3.6-35B-A3B \
  bash phase1_baseline.sh 127.0.0.1:30000 127.0.0.1:30001 ...
```

### Phases 2–3 — cache and draft training (inside `$DEEPSPEC_ROOT`)

```bash
# from phase 0's report:
export WEIRDSPEC_MASK_TOKEN_ID=...              # recommended_mask_token_id
export WEIRDSPEC_DRAFT_INTERMEDIATE_SIZE=...    # recommended_draft_intermediate_size
# export WEIRDSPEC_STRIP_THINK=1                # optional: strip <think> scaffold

python scripts/data/prepare_target_cache.py \
    --config /path/to/examples/03_deepspec_draft_surprise/dspark_qwen36_35b_a3b.py \
    --train-data-path /path/to/data/baseline_train.jsonl \
    --output-dir ~/.cache/deepspec/qwen36_35b_a3b_target_cache \
    --opts model.target_layer_ids="[...from phase 0...]"

bash scripts/train/train.sh   # config_path -> dspark_qwen36_35b_a3b.py
```

The draft trains **only on the baseline** — it must model normal behavior, not
the anomalies. Mind DeepSpec's storage warning: the cache stores per-token
hidden states; size the corpus (and `target_layer_ids`) to your disk budget.

### Phase 4 — scoring

```bash
python phase4_score_surprise.py --deepspec-root $DEEPSPEC_ROOT \
    --draft ~/checkpoints/weirdspec/dspark_block7_qwen36_35b_a3b/step_latest \
    --data data/weird_transcripts.jsonl    --output data/weird_scores.jsonl \
    --target-device-map auto
python phase4_score_surprise.py --deepspec-root $DEEPSPEC_ROOT \
    --draft ... --data data/baseline_heldout.jsonl --output data/baseline_scores.jsonl \
    --target-device-map auto
```

Scoring computes target hidden states on the fly (no cache round-trip, so the
line-alignment with `weird_meta.jsonl` survives) and enumerates anchors
exhaustively: DeepSpec's anchor sampler keeps *every* valid candidate whenever
`num_anchors` ≥ the candidate count, so windowed scoring covers each assistant
token exactly once — deterministically, without patching DeepSpec. Scored
positions are all assistant tokens except the first of each turn (anchors must
carry loss mask, the same constraint training has).

### Phase 5 — report

```bash
python phase5_analyze.py \
    --weird-scores data/weird_scores.jsonl --weird-meta data/weird_meta.jsonl \
    --baseline-scores data/baseline_scores.jsonl --output report.md
```

Produces: pattern ranking by excess surprise (z-scored against the baseline
null), Spearman correlations against WeirdChat's Elo axes and match rates,
block-offset lookahead-decay curves, and token traces of the top patterns with
+3σ/+5σ tokens marked.

## Caveats

- **Checkpoint identity.** WeirdChat's Qwen data was generated from a
  quantized checkpoint served with SGLang; this pipeline interprets the bf16
  HF checkpoint. That is the same replication gap WeirdChat itself documents
  via `openrouter_replication` — keep it in mind when reading small effects.
- **MoE support is gated, not assumed.** DeepSpec ships dense Qwen3 targets
  (4B/8B/14B). Phase 0 confirmed the 35B-A3B target (a `qwen3_5_moe` wrapper)
  loads but needs `deepspec_qwen36.patch` (nested `text_config`, wrapper
  architecture load, MoE `intermediate_size`, think-scaffold rendering) plus
  phase-0 recommendations (mask token, layer ids, draft FFN width). Hidden
  states are captured architecture-agnostically via forward hooks, and the
  draft itself is always dense — the draft's FFN width is a hyperparameter
  (defaulting to the target's per-expert `moe_intermediate_size`), not tied to
  the MoE. No released DeepSpec checkpoint covers this target, so the draft
  must be trained (phases 2–3).
- **`nll_target` is a proxy** from the draft's lm-head over the target's last
  hidden state (the quantity DSpark distills against), not the target's own
  lm-head logits. Consistent across weird/baseline, so excess comparisons
  stand, but absolute values are approximate.
- **Optional extension.** Fine-tuning a second draft on the weird transcripts
  and diffing the two drafts gives a per-behavior "fingerprint", and a
  weird-tuned draft speeds up mass resampling of weird patterns
  (better speculative acceptance) for cheaper rate estimation.

## Tests

Offline logic tests (no torch/network; not collected by the repo CI):

```bash
uv run pytest examples/03_deepspec_draft_surprise/tests -v
```
