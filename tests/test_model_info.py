"""Tests for mlx_kld.model_info — pure-Python, no MLX/Metal needed."""

from __future__ import annotations

from mlx_kld.model_info import _estimate_param_count


# Dense Qwen3.6-27B config (verbatim values from a real on-disk config.json)
DENSE_CFG = {
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "vocab_size": 248320,
    "intermediate_size": 17408,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
}

# MoE Qwen3.6-A3B config (verbatim values from a real on-disk config.json)
MOE_CFG = {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "vocab_size": 248320,
    "intermediate_size": None,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
}


def test_dense_param_count_in_expected_range():
    n = _estimate_param_count(DENSE_CFG)
    assert n is not None
    # Qwen3.6-27B advertises ~27B params. Our estimate ignores LayerNorms,
    # tied weights, and the SSM hybrid layers (worth a few B), so anything in
    # the 22-30B range is a healthy approximation for bpw computation.
    assert 22e9 < n < 30e9, f"got {n / 1e9:.2f}B"


def test_moe_param_count_matches_advertised_total():
    # Marketing name "35B-A3B" → ~35B total params.
    n = _estimate_param_count(MOE_CFG)
    assert n is not None
    assert 32e9 < n < 36e9, f"got {n / 1e9:.2f}B"


def test_moe_estimator_uses_num_experts():
    # Halving the expert count should roughly halve the MoE MLP contribution.
    half = dict(MOE_CFG, num_experts=128)
    n_full = _estimate_param_count(MOE_CFG)
    n_half = _estimate_param_count(half)
    assert n_full is not None and n_half is not None
    # MoE MLP dominates; expect close to 2× ratio.
    ratio = n_full / n_half
    assert 1.7 < ratio < 2.0, f"ratio={ratio:.2f}"


def test_moe_without_shared_expert_still_estimates():
    no_shared = dict(MOE_CFG, shared_expert_intermediate_size=None)
    n_no_shared = _estimate_param_count(no_shared)
    n_full = _estimate_param_count(MOE_CFG)
    assert n_no_shared is not None and n_full is not None
    # Tiny reduction vs full config (shared expert is ~0.01% of total).
    assert n_no_shared < n_full


def test_returns_none_when_neither_dense_nor_moe():
    cfg = {"hidden_size": 1024, "num_hidden_layers": 8, "vocab_size": 32000}
    assert _estimate_param_count(cfg) is None


def test_returns_none_on_missing_core_fields():
    assert _estimate_param_count({}) is None
    assert _estimate_param_count({"hidden_size": 1024}) is None


def test_gqa_attention_smaller_than_full_attention():
    full_attn = dict(DENSE_CFG, num_key_value_heads=DENSE_CFG["num_attention_heads"])
    gqa = DENSE_CFG  # 8 KV heads vs 40 query heads
    n_full = _estimate_param_count(full_attn)
    n_gqa = _estimate_param_count(gqa)
    assert n_full is not None and n_gqa is not None
    # GQA should be smaller (less K, V projection params per layer).
    assert n_gqa < n_full
