"""KLD computation and summary statistics using numpy arrays."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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

    def to_dict(self) -> dict:
        """Serialize to a dict for JSON export."""
        return {
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
            },
            "prompts": [
                {
                    "prompt": r.prompt,
                    "num_tokens": r.num_tokens,
                    "mean_kld": r.mean_kld,
                    "median_kld": r.median_kld,
                    "max_kld": r.max_kld,
                    "max_kld_position": r.max_kld_position,
                    "per_token_kld": r.per_token_kld.tolist(),
                    "token_ids": r.token_ids.tolist(),
                    "token_strings": r.token_strings,
                }
                for r in self.prompt_results
            ],
        }


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
