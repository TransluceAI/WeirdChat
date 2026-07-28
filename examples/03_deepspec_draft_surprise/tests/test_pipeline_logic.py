"""Offline unit tests for the pure logic of the draft-surprise pipeline.

These run in the plain WeirdChat environment (no torch / deepspec / network):
    uv run pytest examples/03_deepspec_draft_surprise/tests -v
They are intentionally outside the repository's `tests/` directory so the
project CI (which runs `pytest tests/`) is unaffected.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surprise_common import (  # noqa: E402
    detect_assistant_prefix,
    messages_to_conversations,
    pick_mask_token,
    recommend_target_layer_ids,
    resolve_draft_intermediate_size,
    spearman,
    summarize_token_scores,
    unwrap_text_config,
)
from phase1_export_weirdchat import export_rows  # noqa: E402
from phase4_score_surprise import anchor_candidates  # noqa: E402
from phase5_analyze import build_pattern_table, null_stats, offset_curve  # noqa: E402


def test_recommend_target_layer_ids_matches_shipped_shape():
    # DeepSpec ships [1, 9, 17, 25, 33] for the 36-layer Qwen3-4B; the
    # recommendation must reproduce that shape: start 1, end num_layers-3.
    assert recommend_target_layer_ids(36) == [1, 9, 17, 25, 33]
    ids48 = recommend_target_layer_ids(48)
    assert ids48[0] == 1 and ids48[-1] == 45 and len(ids48) == 5
    assert ids48 == sorted(set(ids48))


def test_recommend_target_layer_ids_rejects_tiny_models():
    with pytest.raises(ValueError):
        recommend_target_layer_ids(6)


def test_messages_to_conversations_roundtrip_and_validation():
    msgs = [
        SimpleNamespace(role="user", content="hi"),
        SimpleNamespace(role="assistant", content="hello"),
    ]
    assert messages_to_conversations(msgs) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    with pytest.raises(ValueError):
        messages_to_conversations([SimpleNamespace(role="assistant", content="x")])
    with pytest.raises(ValueError):
        messages_to_conversations([SimpleNamespace(role="system", content="x")])


def _fake_pattern_and_transcripts():
    pattern = SimpleNamespace(
        pattern_id="groups/x/behavior-a/method/pg0001",
        behavior_id="behavior-a",
        subject_model="qwen/qwen3.6-35b-a3b",
        checkpoint="Qwen/Qwen3.6-35B-A3B-NVFP4",
        method="evolution",
        metrics=SimpleNamespace(match_rate=0.5),
        openrouter_replication=SimpleNamespace(rate=0.25),
        elo=SimpleNamespace(
            mean=2000.0,
            unexpectedness=SimpleNamespace(elo=2100.0),
            harmfulness=SimpleNamespace(elo=1900.0),
            prompt_naturalness=SimpleNamespace(elo=1500.0),
        ),
    )
    transcripts = [
        SimpleNamespace(
            transcript_id="t-matched",
            prompt_id="p1",
            is_highlight=False,
            judgment=SimpleNamespace(match=True),
            messages=[
                SimpleNamespace(role="user", content="q"),
                SimpleNamespace(role="assistant", content="weird answer"),
            ],
        ),
        SimpleNamespace(
            transcript_id="t-unmatched",
            prompt_id="p1",
            is_highlight=False,
            judgment=SimpleNamespace(match=False),
            messages=[
                SimpleNamespace(role="user", content="q"),
                SimpleNamespace(role="assistant", content="normal answer"),
            ],
        ),
    ]
    return pattern, transcripts


def test_export_rows_matched_only_and_alignment():
    pattern, transcripts = _fake_pattern_and_transcripts()
    data, meta = export_rows([pattern], lambda _p: transcripts, matched_only=True)
    assert len(data) == len(meta) == 1
    assert data[0]["id"] == "t-matched"
    assert data[0]["conversations"][0]["role"] == "user"
    assert meta[0]["pattern_id"] == pattern.pattern_id
    assert meta[0]["elo_unexpectedness"] == 2100.0

    data_all, meta_all = export_rows([pattern], lambda _p: transcripts, matched_only=False)
    assert [m["transcript_id"] for m in meta_all] == ["t-matched", "t-unmatched"]
    assert [m["judge_match"] for m in meta_all] == [True, False]
    assert len(data_all) == 2


def test_anchor_candidates_matches_deepspec_validity_rule():
    # anchor a is valid iff loss_mask[a] and loss_mask[a+1] — the rule in
    # DeepSpec's build_anchor_candidate_mask.
    lm = [0, 0, 1, 1, 1, 0, 1, 1]
    assert anchor_candidates(lm, 0, len(lm) - 2) == [2, 3, 6]
    # windowing: chunks [2,3] and [6] partition the candidate set
    assert anchor_candidates(lm, 0, 3) == [2, 3]
    assert anchor_candidates(lm, 4, len(lm) - 2) == [6]
    # last position can never anchor (no a+1 label)
    assert anchor_candidates([1, 1], 0, 1) == [0]


def test_summarize_token_scores_excess_math():
    agg = summarize_token_scores([1.0, 5.0, 2.0], [1.0, 1.0, 3.0])
    assert agg["n_scored_tokens"] == 3.0
    assert agg["mean_nll_draft"] == pytest.approx(8.0 / 3)
    assert agg["max_excess"] == pytest.approx(4.0)
    assert agg["peak_excess_token_idx"] == 1.0
    assert summarize_token_scores([], []) == {"n_scored_tokens": 0.0}


def test_spearman_basics():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 2], [2, 1]) is None  # too few
    assert spearman([1, 1, 1], [1, 2, 3]) is None  # zero variance


def _score_row(index: int, mean_excess: float, nll_draft=None, nll_target=None):
    nll_draft = nll_draft or [1.0, 2.0]
    nll_target = nll_target or [1.0, 1.0]
    return {
        "index": index,
        "id": f"t{index}",
        "nll_draft": nll_draft,
        "nll_target": nll_target,
        "token_strs": ["a", "b"],
        "offset_mean_nll_draft": [1.0, 2.0],
        "aggregates": {"mean_excess": mean_excess, "peak_excess_token_idx": 0.0},
    }


def test_pattern_table_grouping_and_ranking():
    meta = {
        0: {"pattern_id": "pA", "behavior_id": "b1", "elo_unexpectedness": 2000.0},
        1: {"pattern_id": "pA", "behavior_id": "b1", "elo_unexpectedness": 2000.0},
        2: {"pattern_id": "pB", "behavior_id": "b2", "elo_unexpectedness": 1500.0},
    }
    scores = [_score_row(0, 0.5), _score_row(1, 1.5), _score_row(2, 3.0)]
    rows = build_pattern_table(scores, meta, null_mean=0.0, null_std=1.0)
    assert [r["pattern_id"] for r in rows] == ["pB", "pA"]
    assert rows[1]["n_transcripts"] == 2
    assert rows[1]["mean_excess"] == pytest.approx(1.0)
    assert rows[0]["z_vs_null"] == pytest.approx(3.0)
    # best transcript per pattern is the one with the highest mean excess
    assert rows[1]["best_transcript"]["index"] == 1


def test_null_stats_and_offset_curve():
    baseline = [_score_row(0, 0.0), _score_row(1, 2.0)]
    mean, std = null_stats(baseline, "mean_excess")
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(2.0**0.5)
    assert offset_curve(baseline) == [1.0, 2.0]


def test_unwrap_text_config_nested_and_flat():
    # Nested wrapper config (qwen3_5_moe style): geometry lives in text_config.
    inner = SimpleNamespace(num_hidden_layers=48, hidden_size=4096)
    nested = SimpleNamespace(text_config=inner, num_hidden_layers=None)
    cfg, nesting = unwrap_text_config(nested)
    assert cfg is inner and nesting == "text_config"
    # Flat config stays as-is, even if a text_config attribute exists.
    flat = SimpleNamespace(num_hidden_layers=36, text_config=None)
    cfg, nesting = unwrap_text_config(flat)
    assert cfg is flat and nesting is None
    both = SimpleNamespace(num_hidden_layers=36, text_config=inner)
    cfg, nesting = unwrap_text_config(both)
    assert cfg is both and nesting is None


def test_resolve_draft_intermediate_size_dense_moe_and_derived():
    dense = SimpleNamespace(intermediate_size=18944, hidden_size=4096)
    assert resolve_draft_intermediate_size(dense) == (18944, "intermediate_size")
    # MoE with substantial active capacity (per-expert width x top-k) uses it.
    moe = SimpleNamespace(
        intermediate_size=None,
        moe_intermediate_size=1536,
        num_experts_per_tok=8,
        hidden_size=4096,
    )
    assert resolve_draft_intermediate_size(moe) == (12288, "moe_active_capacity")
    # Fine-grained experts (the qwen3.6-35b-a3b case: 512-wide) would make a
    # crippled dense draft — fall back to the conventional 8/3·hidden width.
    fine = SimpleNamespace(
        intermediate_size=None,
        moe_intermediate_size=512,
        num_experts_per_tok=8,
        hidden_size=4096,
    )
    size, source = resolve_draft_intermediate_size(fine)
    assert source == "derived_from_hidden_size"
    assert size % 128 == 0 and size >= 4096 * 8 // 3 - 128
    # Neither present: Qwen-style 8/3 * hidden, rounded to a multiple of 128.
    bare = SimpleNamespace(intermediate_size=None, moe_intermediate_size=None, hidden_size=3072)
    size, source = resolve_draft_intermediate_size(bare)
    assert source == "derived_from_hidden_size"
    assert size % 128 == 0 and 7000 < size < 9000


class _FakeTokenizer:
    """Minimal apply_chat_template stand-in that emits a <think> scaffold."""

    def __init__(self, scaffold: str):
        self.scaffold = scaffold

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        assert not tokenize
        parts = []
        for m in messages:
            if m["role"] == "user":
                parts.append(f"<|im_start|>user\n{m['content']}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>assistant\n{self.scaffold}{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


def test_detect_assistant_prefix_extracts_scaffold():
    tok = _FakeTokenizer("<think>\n\n</think>\n\n")
    assert detect_assistant_prefix(tok) == "<think>\n\n</think>\n\n"
    # No scaffold -> empty prefix.
    assert detect_assistant_prefix(_FakeTokenizer("")) == ""


def test_pick_mask_token_prefers_pad_like_and_skips_reserved():
    specials = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|endoftext|>": 151643,
        "<|fim_pad|>": 151662,
        "<|quad_start|>": 151650,
    }
    reserved = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
    name, token_id = pick_mask_token(specials, reserved)
    assert (name, token_id) == ("<|fim_pad|>", 151662)
    # Without a pad-like candidate, any unused special is acceptable.
    name2, _ = pick_mask_token(
        {"<|im_start|>": 1, "<|quad_start|>": 2}, {"<|im_start|>"}
    )
    assert name2 == "<|quad_start|>"
    with pytest.raises(ValueError):
        pick_mask_token({"<|im_start|>": 1}, {"<|im_start|>"})
