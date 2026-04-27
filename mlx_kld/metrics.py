"""KLD computation and summary statistics using numpy arrays."""

from dataclasses import dataclass, field
from typing import Union

import numpy as np


# In short-prompt scoring we drop the first few token positions when
# aggregating into the summary mean/percentiles. Position 0..SKIP_FIRST-1
# have very little left-context, so per-token KL there is dominated by the
# unconditioned-distribution prior over sentence-openers — noise that
# doesn't reflect quantisation behaviour. The unscored positions are still
# kept in the per-token detail array for diagnostics.
SKIP_FIRST_TOKENS_SHORT = 8

# Mode tags used to stratify summaries. "short" = list of discrete prompts,
# scored from SKIP_FIRST_TOKENS_SHORT. "long" = streamed corpus chunks at
# fixed n_ctx, scored only from n_ctx/2 onwards (llama.cpp convention).
MODE_SHORT = "short"
MODE_LONG = "long"


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
    """KLD results for a single prompt or chunk.

    ``score_start`` is the first position whose KL counts toward the summary
    statistics. Positions [0, score_start) are kept in ``per_token_kld``
    for diagnostics but excluded from mean/percentile aggregation. For short
    prompts this is SKIP_FIRST_TOKENS_SHORT; for streamed long-mode chunks
    it is n_ctx/2.
    """

    prompt: str
    num_tokens: int
    per_token_kld: np.ndarray  # shape: (num_tokens,)
    token_ids: np.ndarray  # shape: (num_tokens,)
    token_strings: list[str] = field(default_factory=list)
    score_start: int = 0
    mode: str = MODE_SHORT

    @property
    def scored_kld(self) -> np.ndarray:
        """Per-token KL restricted to positions that count toward summaries."""
        if self.score_start <= 0:
            return self.per_token_kld
        return self.per_token_kld[self.score_start:]

    @property
    def num_scored_tokens(self) -> int:
        return int(self.scored_kld.shape[0])

    @property
    def mean_kld(self) -> float:
        s = self.scored_kld
        return float(np.mean(s)) if s.size else 0.0

    @property
    def max_kld(self) -> float:
        s = self.scored_kld
        return float(np.max(s)) if s.size else 0.0

    @property
    def max_kld_position(self) -> int:
        """Position (in the original token sequence) of the max scored KL."""
        s = self.scored_kld
        if not s.size:
            return 0
        return int(np.argmax(s)) + self.score_start

    @property
    def median_kld(self) -> float:
        s = self.scored_kld
        return float(np.median(s)) if s.size else 0.0

    def top_k_divergent(self, k: int = 10) -> list[TokenDivergence]:
        """Return the top-k most divergent token positions among scored ones."""
        s = self.scored_kld
        k = min(k, len(s))
        if k <= 0:
            return []
        indices = np.argsort(s)[-k:][::-1]
        results = []
        for idx in indices:
            absolute_pos = int(idx) + self.score_start
            tok_str = (
                self.token_strings[absolute_pos]
                if self.token_strings and absolute_pos < len(self.token_strings)
                else ""
            )
            tok_id = (
                int(self.token_ids[absolute_pos])
                if absolute_pos < len(self.token_ids)
                else -1
            )
            results.append(
                TokenDivergence(
                    position=absolute_pos,
                    kld=float(s[idx]),
                    token_id=tok_id,
                    token_str=tok_str,
                )
            )
        return results


@dataclass
class ComparisonResult:
    """Aggregate KLD results across all prompts.

    ``prompt_results`` may mix entries from quick and long modes. Summary
    properties default to the *primary* mode: long if any long entries are
    present, otherwise quick. Per-mode summaries are exposed via the
    ``stats_for_mode`` helper and the JSON ``summary_by_mode`` block.
    """

    reference_model: str
    compare_model: str
    prompt_results: list[PromptResult]
    # Optional model metadata + prefill throughput (filled in by the runner).
    reference_info: dict | None = None
    compare_info: dict | None = None
    prefill_tokens_per_second: float | None = None
    prefill_seconds: float | None = None

    # Modes whose stats came from a previous invocation's JSON (not run this
    # time). Lets a `--long`-only run preserve the `--short` numbers from a
    # prior run when writing JSON / rendering reports.
    external_mode_stats: dict = field(default_factory=dict)
    # Same idea for the per-prompt entries from the other-mode prior run, so
    # detail JSON survives across separate invocations. Each entry is the
    # raw prompt-entry dict as it appeared in the previous JSON.
    external_prompt_entries: list = field(default_factory=list)

    @property
    def primary_mode(self) -> str:
        """Long if any long data is present (run-now or external), else quick."""
        has_long = (
            any(r.mode == MODE_LONG for r in self.prompt_results)
            or self.external_mode_stats.get(MODE_LONG) is not None
        )
        return MODE_LONG if has_long else MODE_SHORT

    def _results_for_mode(self, mode: str) -> list[PromptResult]:
        return [r for r in self.prompt_results if r.mode == mode]

    def _scored_kld_for_mode(self, mode: str) -> np.ndarray:
        results = self._results_for_mode(mode)
        if not results:
            return np.array([], dtype=np.float64)
        return np.concatenate([r.scored_kld for r in results])

    @property
    def all_kld(self) -> np.ndarray:
        """Scored per-token KL values for the primary mode (in-memory only)."""
        return self._scored_kld_for_mode(self.primary_mode)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all results (sum of all per-prompt token counts)."""
        return sum(r.num_tokens for r in self.prompt_results)

    def _primary_stat(self, key: str) -> float:
        """Return ``key`` from the primary mode's stats.

        Computes from in-memory results when present; otherwise falls back to
        external_mode_stats (carried in from a prior invocation's JSON). This
        keeps the headline ComparisonResult.mean_kld / .max_kld / etc. properties
        meaningful even when only the *other* mode ran fresh this time.
        """
        stats = self.stats_for_mode(self.primary_mode)
        if not stats:
            return 0.0
        return float(stats.get(key, 0.0))

    @property
    def mean_kld(self) -> float:
        return self._primary_stat("mean_kld")

    @property
    def median_kld(self) -> float:
        return self._primary_stat("median_kld")

    @property
    def std_kld(self) -> float:
        return self._primary_stat("std_kld")

    @property
    def max_kld(self) -> float:
        return self._primary_stat("max_kld")

    def percentile(self, p: float) -> float:
        # Only p95/p99 are pre-computed in stats; for others fall back to in-memory.
        if int(p) == 95:
            return self._primary_stat("p95_kld")
        if int(p) == 99:
            return self._primary_stat("p99_kld")
        a = self.all_kld
        return float(np.percentile(a, p)) if a.size else 0.0

    def stats_for_mode(self, mode: str) -> dict | None:
        """Return summary stats for ``mode``.

        Falls back to ``external_mode_stats`` (loaded from a prior invocation's
        JSON) if the current run produced no results in that mode.
        """
        results = self._results_for_mode(mode)
        if results:
            scored = self._scored_kld_for_mode(mode)
            if scored.size:
                return {
                    "num_prompts": len(results),
                    "total_tokens": sum(r.num_tokens for r in results),
                    "scored_tokens": int(scored.size),
                    "mean_kld": float(np.mean(scored)),
                    "median_kld": float(np.median(scored)),
                    "std_kld": float(np.std(scored)),
                    "max_kld": float(np.max(scored)),
                    "p95_kld": float(np.percentile(scored, 95)),
                    "p99_kld": float(np.percentile(scored, 99)),
                }
        # No fresh results in this mode — fall back to data carried in from
        # a prior invocation's JSON (preserves cross-invocation outputs).
        return self.external_mode_stats.get(mode)

    def merge_external_modes_from_dict(self, existing: dict) -> None:
        """Splice other-mode data from a previous run's JSON into this result.

        Modes also produced by the current run are NOT overwritten — the
        fresh in-memory data wins. Used to preserve, for example, the quick
        summary block when re-running only --long.
        """
        if not existing or existing.get("compare_model") != self.compare_model:
            return
        existing_by_mode = existing.get("summary_by_mode") or {}
        current_modes = {r.mode for r in self.prompt_results}
        for mode in (MODE_SHORT, MODE_LONG):
            if mode in current_modes:
                continue
            stats = existing_by_mode.get(mode)
            if stats:
                self.external_mode_stats[mode] = stats
        # Carry forward the per-prompt detail entries for modes not in the
        # current run, so the merged JSON keeps both runs' prompt detail.
        for prompt_entry in existing.get("prompts", []) or []:
            entry_mode = prompt_entry.get("mode", MODE_SHORT)
            if entry_mode in current_modes:
                continue
            self.external_prompt_entries.append(prompt_entry)

    def to_dict(self, detail: bool = True, top_k: int = 0) -> dict:
        """Serialize to a dict for JSON export.

        Args:
            detail: If True (default), include per-token KLD arrays, token
                IDs, and token strings. Set False to emit only summary stats
                and (optionally) top-K divergent tokens; useful for batch
                comparisons where per-token data would blow up file size.
            top_k: If > 0, include the top-K most divergent tokens per prompt
                regardless of detail.

        Output shape:
            - ``summary``: stats for the primary mode (long if both ran).
              Existing tools that read ``summary.mean_kld`` keep working.
            - ``summary_by_mode``: { quick: stats|None, long: stats|None }
              for stratified analysis when both modes ran.
        """
        primary = self.primary_mode
        primary_stats = self.stats_for_mode(primary) or {
            "num_prompts": 0, "total_tokens": 0, "scored_tokens": 0,
            "mean_kld": 0.0, "median_kld": 0.0, "std_kld": 0.0, "max_kld": 0.0,
            "p95_kld": 0.0, "p99_kld": 0.0,
        }
        summary = dict(primary_stats)
        summary["mode"] = primary
        summary["prefill_tokens_per_second"] = self.prefill_tokens_per_second
        summary["prefill_seconds"] = self.prefill_seconds

        out = {
            "reference_model": self.reference_model,
            "compare_model": self.compare_model,
            "summary": summary,
            "summary_by_mode": {
                MODE_SHORT: self.stats_for_mode(MODE_SHORT),
                MODE_LONG: self.stats_for_mode(MODE_LONG),
            },
            "reference_info": self.reference_info,
            "compare_info": self.compare_info,
            "prompts": [],
        }
        for r in self.prompt_results:
            entry = {
                "prompt": r.prompt,
                "mode": r.mode,
                "num_tokens": r.num_tokens,
                "score_start": r.score_start,
                "num_scored_tokens": r.num_scored_tokens,
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
        # Merge in prompt entries carried forward from a prior invocation
        # (modes that weren't re-run this time).
        for ext_entry in self.external_prompt_entries:
            out["prompts"].append(dict(ext_entry))
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
