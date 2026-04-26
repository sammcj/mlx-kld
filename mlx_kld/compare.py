"""Core comparison logic: load models sequentially, collect logits, compute KLD."""

import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load as mlx_load

from .metrics import (
    ComparisonResult,
    PromptResult,
    SparseLogProbs,
    compute_kld_auto,
    sparsify_log_probs,
)
from .model_info import ModelInfo, extract_model_info


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


def _logits_to_log_probs_np(logits: mx.array) -> np.ndarray:
    """log_softmax in float32, materialise to numpy."""
    logits = logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(log_probs)
    return np.array(log_probs, copy=False)


@dataclass
class _ForwardTiming:
    """Accumulator for prefill throughput measurement."""

    total_tokens: int = 0
    total_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.total_seconds if self.total_seconds > 0 else 0.0


def _collect_log_probs(
    model: nn.Module,
    token_ids: list[int],
    chunk_tokens: int = 0,
    timing: Optional["_ForwardTiming"] = None,
) -> np.ndarray:
    """Run a forward pass and return log-softmax probabilities as numpy.

    When ``chunk_tokens`` is 0 (default), runs a single forward pass.

    When ``chunk_tokens`` > 0, splits the prompt into chunks and feeds them
    sequentially through the model with a shared prompt cache. The cache
    carries the KV state across chunks, so position N inside chunk K attends
    over chunks 0..K-1 exactly as it would in an un-chunked forward pass.
    The output is therefore numerically equivalent to the un-chunked path
    (within MLX kernel-launch noise), only the peak activation memory differs.

    Args:
        model: The MLX language model.
        token_ids: List of token IDs for the prompt.
        chunk_tokens: If > 0, split the forward pass into chunks of this many
            tokens to limit peak activation memory.

    Returns:
        Log-softmax probabilities as numpy array, shape (seq_len, vocab_size).
    """
    if chunk_tokens <= 0 or chunk_tokens >= len(token_ids):
        tokens = mx.array(token_ids)[None]
        t0 = time.perf_counter()
        result = _logits_to_log_probs_np(model(tokens)[0])
        if timing is not None:
            timing.total_tokens += len(token_ids)
            timing.total_seconds += time.perf_counter() - t0
        return result

    # Lazy import — only needed in the chunked path, and lets the rest of the
    # module keep working with older mlx-lm versions.
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    pieces: list[np.ndarray] = []
    for start in range(0, len(token_ids), chunk_tokens):
        chunk = token_ids[start : start + chunk_tokens]
        tokens = mx.array(chunk)[None]
        t0 = time.perf_counter()
        # Passing the cache makes the forward pass attend to all previous
        # chunks via the accumulated KV state.
        logits = model(tokens, cache=cache)[0]
        pieces.append(_logits_to_log_probs_np(logits))
        if timing is not None:
            timing.total_tokens += len(chunk)
            timing.total_seconds += time.perf_counter() - t0
        mx.clear_cache()
    return np.concatenate(pieces, axis=0)


def _unload_model() -> None:
    """Best-effort cleanup of MLX model memory."""
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Reference cache: save / load
# ---------------------------------------------------------------------------

RefEntry = "np.ndarray | SparseLogProbs"


def save_reference(
    path: str | Path,
    reference_model: str,
    prepared: list[dict],
    ref_entries: list,  # list of np.ndarray or SparseLogProbs
    sparse_k: int = 0,
    use_mmap: bool = False,
    reference_info: dict | None = None,
) -> None:
    """Save reference logits and prompt metadata to an .npz file.

    Two formats are supported:

    Dense (sparse_k=0):
        - meta.json   : reference model path + per-prompt metadata + format='dense'
        - logprobs_N  : full (seq_len, vocab_size) log-prob array for prompt N

    Sparse top-K (sparse_k > 0):
        - meta.json     : ... + format='sparse_topk', sparse_k, vocab_size
        - lp_N          : (seq_len, k) float32 top-K log-probs
        - idx_N         : (seq_len, k) int32 vocab indices
        - tail_N        : (seq_len,)  float32 log of remaining mass

    Sparse format reduces a 248k-vocab Qwen reference cache from ~8 GB per
    8K-token prompt down to ~8 MB — a ~1000x reduction with negligible KL
    approximation error for quant ranking purposes.

    Args:
        use_mmap: If True, save uncompressed (np.savez). Slightly larger
            on disk but enables mmap loading on the compare side.
    """
    path = Path(path)
    is_sparse = sparse_k > 0

    meta = {
        "reference_model": reference_model,
        "format": "sparse_topk" if is_sparse else "dense",
        "sparse_k": int(sparse_k),
        "reference_info": reference_info,
        "prompts": [
            {
                "prompt": p["prompt"],
                "token_ids": p["token_ids"],
                "token_strings": p["token_strings"],
            }
            for p in prepared
        ],
    }
    if is_sparse and ref_entries:
        first = ref_entries[0]
        meta["vocab_size"] = int(first.vocab_size if isinstance(first, SparseLogProbs) else first.shape[-1])

    arrays: dict[str, np.ndarray] = {}
    if is_sparse:
        for i, e in enumerate(ref_entries):
            assert isinstance(e, SparseLogProbs), "expected SparseLogProbs entries"
            arrays[f"lp_{i}"] = e.log_probs
            arrays[f"idx_{i}"] = e.indices
            arrays[f"tail_{i}"] = e.tail_log_mass
    else:
        for i, e in enumerate(ref_entries):
            assert isinstance(e, np.ndarray), "expected dense np.ndarray entries"
            arrays[f"logprobs_{i}"] = e
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)

    if use_mmap:
        # Uncompressed so the compare side can mmap_mode='r'.
        np.savez(str(path), **arrays)  # type: ignore[call-overload]
    else:
        np.savez_compressed(str(path), **arrays)  # type: ignore[call-overload]

    npz_path = path if str(path).endswith(".npz") else f"{path}.npz"
    size_mb = Path(npz_path).stat().st_size / 1e6
    fmt = f"sparse top-{sparse_k}" if is_sparse else "dense"
    _log(f"Reference saved: {npz_path} ({fmt}, {size_mb:.1f} MB)")


def load_reference(
    path: str | Path,
    use_mmap: bool = True,
) -> tuple[str, list[dict], list, dict | None]:
    """Load reference data from a previously saved .npz file.

    Returns:
        (reference_model_path, prepared_prompts, ref_entries, reference_info)
        where ref_entries is a list of np.ndarray (dense format) or
        SparseLogProbs (sparse format), and reference_info is the optional
        ModelInfo dict captured at cache-build time.
    """
    path = Path(path)
    npz_path = path.with_suffix(".npz") if path.suffix != ".npz" else path

    # mmap_mode is only honoured for uncompressed .npz; np.load silently
    # ignores it for compressed files. Try mmap first, fall back to in-RAM.
    try:
        data = np.load(npz_path, allow_pickle=False, mmap_mode="r" if use_mmap else None)
    except (TypeError, ValueError):
        data = np.load(npz_path, allow_pickle=False)

    meta = json.loads(bytes(data["meta_json"]).decode())
    reference_model = meta["reference_model"]
    prepared = meta["prompts"]
    fmt = meta.get("format", "dense")
    reference_info = meta.get("reference_info")

    ref_entries: list = []
    if fmt == "sparse_topk":
        vocab_size = int(meta["vocab_size"])
        i = 0
        while f"lp_{i}" in data:
            ref_entries.append(
                SparseLogProbs(
                    log_probs=data[f"lp_{i}"],
                    indices=data[f"idx_{i}"],
                    tail_log_mass=data[f"tail_{i}"],
                    vocab_size=vocab_size,
                )
            )
            i += 1
        _log(f"Loaded sparse-{meta.get('sparse_k')} reference from {npz_path} "
             f"({i} prompts, model: {reference_model})")
    else:
        i = 0
        while f"logprobs_{i}" in data:
            ref_entries.append(data[f"logprobs_{i}"])
            i += 1
        _log(f"Loaded dense reference from {npz_path} "
             f"({i} prompts, model: {reference_model})")

    return reference_model, prepared, ref_entries, reference_info


# ---------------------------------------------------------------------------
# Main compare entry point
# ---------------------------------------------------------------------------

def compare(
    reference: str,
    comparisons: list[str],
    prompts: list[str],
    use_chat_template: bool = True,
    save_ref: str | Path | None = None,
    load_ref: str | Path | None = None,
    sparse_k: int = 0,
    chunk_tokens: int = 0,
) -> list[ComparisonResult]:
    """Compare one or more models against a reference by measuring KL divergence.

    Reference logits are collected once (or loaded from disk) and reused for
    every comparison model, so the reference model is never loaded twice.

    Args:
        reference: Path or HF repo for the reference model. Ignored if load_ref
            is provided.
        comparisons: List of paths/HF repos for the models to compare.
        prompts: List of prompt strings to evaluate.
        use_chat_template: Whether to apply the model's chat template.
        save_ref: If provided, save reference logits to this path after
            collecting them (so future runs can skip the reference model).
        load_ref: If provided, load previously saved reference logits from
            this path instead of running the reference model.
        sparse_k: If > 0, store the reference cache as top-K sparse log-probs
            with K=sparse_k. Reduces cache size by ~vocab_size/k (typically
            1000x) with negligible KL approximation error.
        chunk_tokens: If > 0, split forward passes into chunks of this many
            tokens to reduce peak activation memory on long prompts.

    Returns:
        List of ComparisonResult, one per comparison model, in input order.
    """
    # --- Phase 1: Reference logits (run model or load from disk) ---
    reference_info: dict | None = None
    if load_ref is not None:
        reference, prepared, ref_entries, reference_info = load_reference(load_ref)
        # Prompts are baked into the cache; warn if caller also passed prompts
        if prompts:
            _log(
                "  Note: --load-reference includes its own prompts; "
                "--prompts / --prompts-file will be ignored."
            )
    else:
        _log(f"Loading reference model: {reference}")
        ref_info_obj = extract_model_info(reference)
        reference_info = ref_info_obj.to_dict()
        _log(f"  {ref_info_obj.short_summary()}")
        t0 = time.time()
        ref_model, ref_tokenizer, *_ = mlx_load(reference)
        load_time = time.time() - t0
        peak_mem = mx.get_peak_memory() / 1e9
        _log(f"  Loaded in {load_time:.1f}s (peak memory: {peak_mem:.1f} GB)")

        prepared = _prepare_prompts(ref_tokenizer, prompts, use_chat_template)

        _log(f"Collecting reference logits for {len(prompts)} prompt(s)...")
        ref_timing = _ForwardTiming()
        ref_entries: list = []
        for i, p in enumerate(prepared):
            log_probs = _collect_log_probs(
                ref_model, p["token_ids"],
                chunk_tokens=chunk_tokens, timing=ref_timing,
            )
            if sparse_k > 0:
                vocab = log_probs.shape[-1]
                k_eff = min(sparse_k, vocab - 1)
                ref_entries.append(sparsify_log_probs(log_probs, k_eff))
                # Free the dense array immediately — it's the big one
                del log_probs
            else:
                ref_entries.append(log_probs)
            _log(f"  [{i + 1}/{len(prompts)}] {len(p['token_ids'])} tokens")
        _log(f"  Reference prefill: {ref_timing.total_tokens} tokens "
             f"in {ref_timing.total_seconds:.1f}s = {ref_timing.tokens_per_second:.1f} tok/s")

        _log("Unloading reference model...")
        del ref_model
        del ref_tokenizer
        _unload_model()

        if save_ref is not None:
            save_reference(
                save_ref, reference, prepared, ref_entries,
                sparse_k=sparse_k, use_mmap=True,
                reference_info=reference_info,
            )

    # --- Phase 2: Loop over comparison models ---
    results: list[ComparisonResult] = []

    for cmp_idx, comparison in enumerate(comparisons):
        _log(f"\nLoading comparison model [{cmp_idx + 1}/{len(comparisons)}]: {comparison}")
        cmp_info_obj = extract_model_info(comparison)
        compare_info = cmp_info_obj.to_dict()
        _log(f"  {cmp_info_obj.short_summary()}")
        mx.reset_peak_memory()
        t0 = time.time()
        cmp_model, cmp_tokenizer, *_ = mlx_load(comparison)
        load_time = time.time() - t0
        peak_mem = mx.get_peak_memory() / 1e9
        _log(f"  Loaded in {load_time:.1f}s (peak memory: {peak_mem:.1f} GB)")

        # Use the *reference* token ids verbatim. Re-tokenising with the
        # comparison tokenizer can produce a different sequence (different
        # chat template, vocab quirks, etc.) which silently misaligns
        # positions and turns KL into noise. The whole point of the
        # comparison is to feed both models the same input.
        cmp_prepared = prepared

        _log(f"Computing KL divergence for {len(prepared)} prompt(s)...")
        cmp_timing = _ForwardTiming()
        prompt_results: list[PromptResult] = []
        for i, (ref_entry, p) in enumerate(zip(ref_entries, cmp_prepared)):
            cmp_log_probs = _collect_log_probs(
                cmp_model, p["token_ids"],
                chunk_tokens=chunk_tokens, timing=cmp_timing,
            )

            ref_seq_len = (
                ref_entry.seq_len
                if isinstance(ref_entry, SparseLogProbs)
                else ref_entry.shape[0]
            )
            if ref_seq_len != cmp_log_probs.shape[0]:
                _log(
                    f"  WARNING: Token count mismatch for prompt {i + 1}: "
                    f"ref={ref_seq_len}, cmp={cmp_log_probs.shape[0]}. "
                    f"Truncating to shorter."
                )
                min_len = min(ref_seq_len, cmp_log_probs.shape[0])
                cmp_log_probs = cmp_log_probs[:min_len]
                if isinstance(ref_entry, SparseLogProbs):
                    ref_entry = SparseLogProbs(
                        log_probs=ref_entry.log_probs[:min_len],
                        indices=ref_entry.indices[:min_len],
                        tail_log_mass=ref_entry.tail_log_mass[:min_len],
                        vocab_size=ref_entry.vocab_size,
                    )
                else:
                    ref_entry = ref_entry[:min_len]

            per_token_kld = compute_kld_auto(ref_entry, cmp_log_probs)
            _log(
                f"  [{i + 1}/{len(prepared)}] {len(per_token_kld)} tokens, "
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

        _log(f"  Compare prefill: {cmp_timing.total_tokens} tokens "
             f"in {cmp_timing.total_seconds:.1f}s = {cmp_timing.tokens_per_second:.1f} tok/s")
        _log(f"Unloading comparison model: {comparison}")
        del cmp_model
        del cmp_tokenizer
        _unload_model()

        results.append(
            ComparisonResult(
                reference_model=reference,
                compare_model=comparison,
                prompt_results=prompt_results,
                reference_info=reference_info,
                compare_info=compare_info,
                prefill_tokens_per_second=cmp_timing.tokens_per_second,
                prefill_seconds=cmp_timing.total_seconds,
            )
        )

    return results
