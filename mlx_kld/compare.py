"""Core comparison logic: load models sequentially, collect logits, compute KLD."""

import gc
import sys
import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load as mlx_load

from .metrics import ComparisonResult, PromptResult, compute_kld


def _log(msg: str) -> None:
    """Print a status message to stderr (keeps stdout clean for piping)."""
    print(msg, file=sys.stderr, flush=True)


def _prepare_prompts(
    tokenizer,
    prompts: list[str],
    use_chat_template: bool = True,
) -> list[dict]:
    """Tokenize prompts, optionally applying the chat template.

    Returns a list of dicts with 'prompt', 'token_ids', 'token_strings'.
    """
    prepared = []
    for prompt in prompts:
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            token_ids = tokenizer.encode(prompt)

        token_strings = [tokenizer.decode([tid]) for tid in token_ids]

        prepared.append(
            {
                "prompt": prompt,
                "token_ids": token_ids,
                "token_strings": token_strings,
            }
        )

    return prepared


def _collect_log_probs(
    model: nn.Module,
    token_ids: list[int],
) -> np.ndarray:
    """Run a forward pass and return log-softmax probabilities as numpy.

    Args:
        model: The MLX language model.
        token_ids: List of token IDs for the prompt.

    Returns:
        Log-softmax probabilities as numpy array, shape (seq_len, vocab_size).
    """
    tokens = mx.array(token_ids)[None]  # (1, seq_len)
    logits = model(tokens)  # (1, seq_len, vocab_size)
    logits = logits[0]  # (seq_len, vocab_size) -- drop batch dim

    # Log-softmax for numerical stability, in float32 for numpy compat
    logits = logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Force evaluation and convert to numpy to free MLX memory
    mx.eval(log_probs)
    result = np.array(log_probs, copy=False)

    return result


def _unload_model() -> None:
    """Best-effort cleanup of MLX model memory."""
    gc.collect()
    mx.clear_cache()


def compare(
    reference: str,
    comparison: str,
    prompts: list[str],
    use_chat_template: bool = True,
) -> ComparisonResult:
    """Compare two models by measuring KL divergence on their output distributions.

    Models are loaded sequentially to minimize peak memory usage.

    Args:
        reference: Path or HF repo for the reference model (typically fp16).
        comparison: Path or HF repo for the comparison model (typically quantized).
        prompts: List of prompt strings to evaluate.
        use_chat_template: Whether to apply the model's chat template.

    Returns:
        ComparisonResult with per-prompt and aggregate KLD metrics.
    """
    # --- Phase 1: Reference model ---
    _log(f"Loading reference model: {reference}")
    t0 = time.time()
    ref_model, ref_tokenizer = mlx_load(reference)
    load_time = time.time() - t0
    peak_mem = mx.get_peak_memory() / 1e9
    _log(f"  Loaded in {load_time:.1f}s (peak memory: {peak_mem:.1f} GB)")

    prepared = _prepare_prompts(ref_tokenizer, prompts, use_chat_template)

    _log(f"Collecting reference logits for {len(prompts)} prompt(s)...")
    ref_log_probs_list: list[np.ndarray] = []
    for i, p in enumerate(prepared):
        log_probs = _collect_log_probs(ref_model, p["token_ids"])
        ref_log_probs_list.append(log_probs)
        _log(f"  [{i + 1}/{len(prompts)}] {len(p['token_ids'])} tokens")

    _log("Unloading reference model...")
    del ref_model
    del ref_tokenizer
    _unload_model()

    # --- Phase 2: Comparison model ---
    _log(f"\nLoading comparison model: {comparison}")
    mx.reset_peak_memory()
    t0 = time.time()
    cmp_model, cmp_tokenizer = mlx_load(comparison)
    load_time = time.time() - t0
    peak_mem = mx.get_peak_memory() / 1e9
    _log(f"  Loaded in {load_time:.1f}s (peak memory: {peak_mem:.1f} GB)")

    # Re-tokenize with comparison tokenizer (should be identical but be safe)
    cmp_prepared = _prepare_prompts(cmp_tokenizer, prompts, use_chat_template)

    _log(f"Computing KL divergence for {len(prompts)} prompt(s)...")
    prompt_results: list[PromptResult] = []
    for i, (ref_lp, p) in enumerate(zip(ref_log_probs_list, cmp_prepared)):
        cmp_log_probs = _collect_log_probs(cmp_model, p["token_ids"])

        # Verify shapes match (same tokenizer should produce same lengths)
        if ref_lp.shape != cmp_log_probs.shape:
            _log(
                f"  WARNING: Token count mismatch for prompt {i + 1}: "
                f"ref={ref_lp.shape[0]}, cmp={cmp_log_probs.shape[0]}. "
                f"Truncating to shorter."
            )
            min_len = min(ref_lp.shape[0], cmp_log_probs.shape[0])
            ref_lp = ref_lp[:min_len]
            cmp_log_probs = cmp_log_probs[:min_len]

        per_token_kld = compute_kld(ref_lp, cmp_log_probs)
        _log(
            f"  [{i + 1}/{len(prompts)}] {len(per_token_kld)} tokens, "
            f"mean KLD: {np.mean(per_token_kld):.6f}"
        )

        prompt_results.append(
            PromptResult(
                prompt=p["prompt"],
                num_tokens=len(per_token_kld),
                per_token_kld=per_token_kld,
                token_ids=np.array(p["token_ids"][: len(per_token_kld)]),
                token_strings=p["token_strings"][: len(per_token_kld)],
            )
        )

    _log("Unloading comparison model...")
    del cmp_model
    del cmp_tokenizer
    _unload_model()

    return ComparisonResult(
        reference_model=reference,
        compare_model=comparison,
        prompt_results=prompt_results,
    )
