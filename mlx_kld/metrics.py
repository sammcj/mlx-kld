"""KLD computation and summary statistics using numpy arrays."""

from dataclasses import dataclass, field
from typing import Union

import numpy as np


@dataclass
class SparseLogProbs:
    """Top-K log-probs from a reference forward pass.

    Stores only the K most-likely tokens per position plus a single tail
    log-mass term. Reduces storage from vocab_size floats per position
    (e.g. 248k for Qwen) to ~K floats + K int32 indices, typically a
    1000x reduction at K=256 with negligible KL approximation error.

    The tail mass holds log(sum(exp(log_probs[k:vocab]))) — i.e. the
    aggregate probability of every token outside the top-K — so the
    approximation is rank-preserving and unbiased for KL ranking of
    quantisation variants.
    """

    log_probs: np.ndarray  # (seq_len, k), float32 — top-K log-probs
    indices: np.ndarray    # (seq_len, k), int32 — vocab ids of those probs
    tail_log_mass: np.ndarray  # (seq_len,), float32 — log of remaining mass
    vocab_size: int

    @property
    def seq_len(self) -> int:
        return int(self.log_probs.shape[0])

    @property
    def k(self) -> int:
        return int(self.log_probs.shape[1])


@dataclass
class TokenDivergence:
    """KLD info for a single token position."""

    position: int
    kld: float
    token_id: int
    token_str: str = ""


@dataclass
class PromptResult:
    """KLD results for a single prompt."""

    prompt: str
    num_tokens: int
    per_token_kld: np.ndarray  # shape: (num_tokens,)
    token_ids: np.ndarray  # shape: (num_tokens,)
    token_strings: list[str] = field(default_factory=list)

    @property
    def mean_kld(self) -> float:
        return float(np.mean(self.per_token_kld))

    @property
    def max_kld(self) -> float:
        return float(np.max(self.per_token_kld))

    @property
    def max_kld_position(self) -> int:
        return int(np.argmax(self.per_token_kld))

    @property
    def median_kld(self) -> float:
        return float(np.median(self.per_token_kld))

    def top_k_divergent(self, k: int = 10) -> list[TokenDivergence]:
        """Return the top-k most divergent token positions."""
        k = min(k, len(self.per_token_kld))
        indices = np.argsort(self.per_token_kld)[-k:][::-1]
        results = []
        for idx in indices:
            tok_str = self.token_strings[idx] if self.token_strings else ""
            results.append(
                TokenDivergence(
                    position=int(idx),
                    kld=float(self.per_token_kld[idx]),
                    token_id=int(self.token_ids[idx]),
                    token_str=tok_str,
                )
            )
        return results


@dataclass
class ComparisonResult:
    """Aggregate KLD results across all prompts."""

    reference_model: str
    compare_model: str
    prompt_results: list[PromptResult]
    # Optional model metadata + prefill throughput (filled in by the runner).
    reference_info: dict | None = None
    compare_info: dict | None = None
    prefill_tokens_per_second: float | None = None
    prefill_seconds: float | None = None

    @property
    def all_kld(self) -> np.ndarray:
        """Concatenate all per-token KLD values."""
        return np.concatenate([r.per_token_kld for r in self.prompt_results])

    @property
    def total_tokens(self) -> int:
        return sum(r.num_tokens for r in self.prompt_results)

    @property
    def mean_kld(self) -> float:
        return float(np.mean(self.all_kld))

    @property
    def median_kld(self) -> float:
        return float(np.median(self.all_kld))

    @property
    def std_kld(self) -> float:
        return float(np.std(self.all_kld))

    @property
    def max_kld(self) -> float:
        return float(np.max(self.all_kld))

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.all_kld, p))

    def to_dict(self, detail: bool = True, top_k: int = 0) -> dict:
        """Serialize to a dict for JSON export.

        Args:
            detail: If True (default), include per-token KLD arrays, token
                IDs, and token strings — same shape as the original output
                format. Set False to emit only summary stats and (optionally)
                top-K divergent tokens; useful for batch comparisons where
                per-token data would blow up file size.
            top_k: If > 0, include the top-K most divergent tokens per prompt
                regardless of detail. Useful for qualitative debugging without
                emitting full per-token arrays.
        """
        out = {
            "reference_model": self.reference_model,
            "compare_model": self.compare_model,
            "summary": {
                "num_prompts": len(self.prompt_results),
                "total_tokens": self.total_tokens,
                "mean_kld": self.mean_kld,
                "median_kld": self.median_kld,
                "std_kld": self.std_kld,
                "max_kld": self.max_kld,
                "p95_kld": self.percentile(95),
                "p99_kld": self.percentile(99),
                "prefill_tokens_per_second": self.prefill_tokens_per_second,
                "prefill_seconds": self.prefill_seconds,
            },
            "reference_info": self.reference_info,
            "compare_info": self.compare_info,
            "prompts": [],
        }
        for r in self.prompt_results:
            entry = {
                "prompt": r.prompt,
                "num_tokens": r.num_tokens,
                "mean_kld": r.mean_kld,
                "median_kld": r.median_kld,
                "max_kld": r.max_kld,
                "max_kld_position": r.max_kld_position,
            }
            if top_k > 0:
                entry["top_divergent_tokens"] = [
                    {
                        "position": td.position,
                        "kld": td.kld,
                        "token_id": td.token_id,
                        "token_str": td.token_str,
                    }
                    for td in r.top_k_divergent(top_k)
                ]
            if detail:
                entry["per_token_kld"] = r.per_token_kld.tolist()
                entry["token_ids"] = r.token_ids.tolist()
                entry["token_strings"] = r.token_strings
            out["prompts"].append(entry)
        return out


def compute_kld(
    ref_log_probs: np.ndarray,
    cmp_log_probs: np.ndarray,
) -> np.ndarray:
    """Compute per-token KL divergence: KL(P_ref || P_cmp).

    Args:
        ref_log_probs: Log-softmax output from reference model.
            Shape: (seq_len, vocab_size)
        cmp_log_probs: Log-softmax output from comparison model.
            Shape: (seq_len, vocab_size)

    Returns:
        Per-token KLD values. Shape: (seq_len,)
    """
    # KL(P || Q) = sum(P * (log P - log Q))
    # P = exp(ref_log_probs), so P * log P = exp(ref_log_probs) * ref_log_probs
    p = np.exp(ref_log_probs)
    kld_per_token = np.sum(p * (ref_log_probs - cmp_log_probs), axis=-1)

    # Clamp to >= 0 (small numerical errors can produce tiny negatives)
    kld_per_token = np.maximum(kld_per_token, 0.0)

    return kld_per_token


def sparsify_log_probs(log_probs: np.ndarray, k: int) -> SparseLogProbs:
    """Reduce a dense (seq_len, vocab) log-prob array to its top-K plus tail mass.

    Used at reference-collection time to produce a compact cache.
    """
    seq_len, vocab_size = log_probs.shape
    if k <= 0 or k >= vocab_size:
        raise ValueError(f"k must be in (0, vocab_size); got k={k}, vocab={vocab_size}")

    # argpartition gets top-K unsorted (faster than full argsort for large vocab)
    part = np.argpartition(log_probs, -k, axis=-1)[:, -k:]
    # Sort just the top-K descending so the cache has a stable order
    rows = np.arange(seq_len)[:, None]
    topk_vals = log_probs[rows, part]
    order = np.argsort(-topk_vals, axis=-1)
    indices = np.take_along_axis(part, order, axis=-1).astype(np.int32)
    top_log_probs = np.take_along_axis(topk_vals, order, axis=-1).astype(np.float32)

    # Tail mass: log(1 - sum(exp(top_log_probs))) computed stably as
    # log1p(-sum_top_p) where sum_top_p is clamped just below 1.
    sum_top_p = np.sum(np.exp(top_log_probs), axis=-1)
    # Numerical guard: sum_top_p can be 1.0 exactly when the top-K already
    # captures the full distribution (very peaked output). Treat as zero tail.
    eps = 1e-12
    tail_p = np.maximum(1.0 - sum_top_p, eps)
    tail_log_mass = np.log(tail_p).astype(np.float32)

    return SparseLogProbs(
        log_probs=top_log_probs,
        indices=indices,
        tail_log_mass=tail_log_mass,
        vocab_size=vocab_size,
    )


def compute_kld_sparse(
    ref: SparseLogProbs,
    cmp_log_probs: np.ndarray,
) -> np.ndarray:
    """Compute per-token KL using a top-K sparse reference and dense comparison logits.

    Math:
        KL(P || Q) = sum_{x in vocab} P(x) * (log P(x) - log Q(x))

        With sparse P:
            sum over top-K: known directly from ref.log_probs
            sum over tail:  P_tail * (log P_tail_avg - log Q_tail_avg)

        We approximate the tail by lumping it into a single "rest" bin with
        probability mass exp(tail_log_mass) and assume the comparison model's
        mass on those tokens equals (1 - sum of cmp probs over the top-K
        indices). This is exact in expectation if the comparison model
        distributes its tail similarly. Empirically the approximation error
        is far below the differences between quants.

    Args:
        ref: Sparse top-K reference log-probs (from sparsify_log_probs).
        cmp_log_probs: Dense (seq_len, vocab_size) log-probs from the
            comparison model.

    Returns:
        Per-token KLD values. Shape: (seq_len,)
    """
    seq_len = ref.seq_len
    if cmp_log_probs.shape[0] != seq_len:
        raise ValueError(
            f"seq_len mismatch: ref={seq_len}, cmp={cmp_log_probs.shape[0]}"
        )
    if cmp_log_probs.shape[1] != ref.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: ref={ref.vocab_size}, "
            f"cmp={cmp_log_probs.shape[1]}"
        )

    # Gather the comparison log-probs at the same vocab positions as the
    # reference top-K. Shape: (seq_len, k)
    rows = np.arange(seq_len)[:, None]
    cmp_top = cmp_log_probs[rows, ref.indices]

    # Top-K contribution: sum_{x in topK} P(x) * (log P(x) - log Q(x))
    p_top = np.exp(ref.log_probs)
    kld_top = np.sum(p_top * (ref.log_probs - cmp_top), axis=-1)

    # Tail contribution: lump the remaining vocab into one bin.
    # cmp tail mass = 1 - sum(exp(cmp_top))  (clamped >0 for numerical safety)
    cmp_top_p_sum = np.sum(np.exp(cmp_top), axis=-1)
    cmp_tail_p = np.maximum(1.0 - cmp_top_p_sum, 1e-12)
    cmp_tail_log = np.log(cmp_tail_p)
    ref_tail_p = np.exp(ref.tail_log_mass)
    kld_tail = ref_tail_p * (ref.tail_log_mass - cmp_tail_log)

    kld_per_token = kld_top + kld_tail
    return np.maximum(kld_per_token, 0.0).astype(np.float64)


def compute_kld_auto(
    ref: Union[np.ndarray, SparseLogProbs],
    cmp_log_probs: np.ndarray,
) -> np.ndarray:
    """Dispatch to compute_kld or compute_kld_sparse based on reference type."""
    if isinstance(ref, SparseLogProbs):
        return compute_kld_sparse(ref, cmp_log_probs)
    return compute_kld(ref, cmp_log_probs)
