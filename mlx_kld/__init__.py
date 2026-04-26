"""mlx-kld: KL divergence measurement for MLX language models."""

# Re-export pure-numpy metrics unconditionally so tests can import without mlx.
from .metrics import (
    ComparisonResult,
    PromptResult,
    SparseLogProbs,
    compute_kld,
    compute_kld_sparse,
    sparsify_log_probs,
)

# Lazy-import compare (which needs mlx) so the metrics namespace is usable
# in environments without mlx installed (tests, docs, etc.).
def __getattr__(name):
    if name == "compare":
        from .compare import compare as _compare
        return _compare
    raise AttributeError(f"module 'mlx_kld' has no attribute {name!r}")

__all__ = [
    "compare",
    "compute_kld",
    "compute_kld_sparse",
    "sparsify_log_probs",
    "ComparisonResult",
    "PromptResult",
    "SparseLogProbs",
]
