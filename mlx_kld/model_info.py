"""Extract quantisation config and on-disk size from an MLX model directory.

Used by the KLD comparison runner to attach contextual model metadata to
each comparison result, so reports can show what each quant actually is
(bits, group_size, mixed-precision overrides, file size) alongside its KLD.

Pure-stdlib — no MLX dependency, so it can be imported and tested anywhere.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelInfo:
    """Compact summary of a quantised MLX model on disk."""

    path: str
    size_bytes: int
    architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    num_hidden_layers: Optional[int] = None
    vocab_size: Optional[int] = None

    # Quantisation
    is_quantised: bool = False
    base_bits: Optional[int] = None
    group_size: Optional[int] = None
    mode: Optional[str] = None
    num_overrides: int = 0
    bit_distribution: dict[int, int] = field(default_factory=dict)
    override_categories: dict[str, int] = field(default_factory=dict)
    quant_family: str = "unknown"  # "rtn", "ud", "oq", "dwq", "autoround", etc.

    # Derived
    effective_bpw: Optional[float] = None  # size_bytes * 8 / param_count

    # Versioning — lets you tell whether two runs were against the same
    # bytes-on-disk even when authors re-publish under the same model name.
    # weights_mtime: filesystem modified time of the newest weight file.
    # weights_index_sha: sha256 of model.safetensors.index.json (12-char prefix).
    weights_mtime_iso: Optional[str] = None
    weights_mtime_unix: Optional[float] = None
    weights_index_sha: Optional[str] = None
    hf_revision: Optional[str] = None  # populated if cached via hf_hub

    def short_summary(self) -> str:
        """One-line human-readable summary suitable for terminal display."""
        size_gb = self.size_bytes / 1e9
        ver = self._version_tag()
        if not self.is_quantised:
            return f"{size_gb:.1f} GB, unquantised{ver}"
        bpw = f"~{self.effective_bpw:.2f} bpw" if self.effective_bpw else "?bpw"
        bits_str = self._format_bit_distribution()
        return (
            f"{size_gb:.1f} GB, {bpw}, {self.quant_family}, "
            f"base={self.base_bits}b/gs{self.group_size} {bits_str}{ver}"
        )

    def _version_tag(self) -> str:
        """Compact 'v=<sha or date>' suffix for the short summary."""
        bits = []
        if self.weights_index_sha:
            bits.append(self.weights_index_sha)
        if self.weights_mtime_iso:
            bits.append(self.weights_mtime_iso[:10])  # date only
        if self.hf_revision:
            bits.append(f"hf={self.hf_revision[:8]}")
        return f" [v: {' / '.join(bits)}]" if bits else ""

    def _format_bit_distribution(self) -> str:
        if not self.bit_distribution:
            return "(uniform)"
        parts = [f"{c}@{b}b" for b, c in sorted(self.bit_distribution.items())]
        return f"overrides: {', '.join(parts)}"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / 1e9, 2),
            "architectures": self.architectures,
            "model_type": self.model_type,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
            "is_quantised": self.is_quantised,
            "base_bits": self.base_bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "quant_family": self.quant_family,
            "num_overrides": self.num_overrides,
            "bit_distribution": self.bit_distribution,
            "override_categories": self.override_categories,
            "effective_bpw": (
                round(self.effective_bpw, 3) if self.effective_bpw is not None else None
            ),
            "weights_mtime_iso": self.weights_mtime_iso,
            "weights_mtime_unix": self.weights_mtime_unix,
            "weights_index_sha": self.weights_index_sha,
            "hf_revision": self.hf_revision,
        }


def _weight_versioning(path: Path) -> tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    """Return (mtime_iso, mtime_unix, index_sha, hf_revision) for a model dir.

    - mtime: latest filesystem modified time across the safetensors files.
      Useful as a quick "did this change?" signal even when authors don't
      version their releases.
    - index_sha: 12-char prefix of sha256 over model.safetensors.index.json.
      The index file deterministically encodes the weight layout (filenames,
      tensor → file mapping, total size) and is stable across re-downloads
      of the same release. Cheap to hash (~200 KB).
    - hf_revision: best-effort lookup. If the model dir is the HF hub cache,
      the path includes the snapshot SHA; otherwise None.
    """
    mtime: Optional[float] = None
    for p in path.glob("*.safetensors"):
        m = p.stat().st_mtime
        if mtime is None or m > mtime:
            mtime = m
    mtime_iso = (
        _dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
        if mtime is not None else None
    )

    index_sha: Optional[str] = None
    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        try:
            index_sha = hashlib.sha256(index_file.read_bytes()).hexdigest()[:12]
        except OSError:
            pass

    # HF cache path layout: .../models--ORG--NAME/snapshots/<sha>/...
    hf_revision: Optional[str] = None
    parts = path.resolve().parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            hf_revision = parts[idx + 1]

    return mtime_iso, mtime, index_sha, hf_revision


def _dir_size_bytes(path: Path) -> int:
    """Sum sizes of all weight-bearing files in a model directory.

    Counts safetensors + model.safetensors.index.json. Ignores tokenizers,
    chat templates, and other small-but-irrelevant files so the number
    reflects the actual quantised weight footprint.
    """
    total = 0
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith(".safetensors") or name.endswith(".npz"):
            total += p.stat().st_size
        elif name == "model.safetensors.index.json":
            total += p.stat().st_size
    if total == 0:
        # Fall back to total directory size if no safetensors found
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


_OVERRIDE_CATEGORIES = (
    ("embed_tokens", lambda k: "embed" in k),
    ("lm_head", lambda k: "lm_head" in k),
    ("self_attn", lambda k: any(s in k for s in ("q_proj", "k_proj", "v_proj", "o_proj"))),
    ("linear_attn(SSM)", lambda k: "linear_attn" in k),
    ("shared_expert", lambda k: "shared_expert" in k),
    ("mlp", lambda k: any(s in k for s in ("gate_proj", "up_proj", "down_proj"))),
    ("vision", lambda k: "visual" in k or "vision" in k),
    ("router", lambda k: "router" in k or k.endswith(".gate")),
)


def _categorise_overrides(override_keys: list[str]) -> dict[str, int]:
    cats: Counter[str] = Counter()
    for k in override_keys:
        for label, pred in _OVERRIDE_CATEGORIES:
            if pred(k):
                cats[label] += 1
                break
        else:
            cats["other"] += 1
    return dict(cats)


def _detect_quant_family(model_path: Path, quant_cfg: dict, num_overrides: int) -> str:
    """Best-effort identification of which quant tool produced the model.

    Heuristic — looks at directory naming conventions and override patterns.
    """
    name = model_path.name.lower()
    if "oq" in name or "/jundot/" in str(model_path).lower():
        return "oQ"
    if "ud-mlx" in name or "ud_mlx" in name or "/unsloth/" in str(model_path).lower():
        return "Unsloth Dynamic"
    if "dwq" in name:
        return "DWQ"
    if "autoround" in name:
        return "AutoRound"
    if "dflash" in name:
        return "DFlash"
    if quant_cfg.get("mode") == "affine" and num_overrides == 0:
        return "RTN affine (uniform)"
    if num_overrides > 0:
        return "RTN affine (mixed)"
    return "unknown"


def _estimate_param_count(text_cfg: dict) -> Optional[int]:
    """Rough parameter count from architecture config.

    Covers attention (with GQA) + MLP (dense or MoE) + embedding parameters.
    Used only for computing effective bpw, so a few percent error from
    LayerNorms, tied weights, hybrid SSM layers, etc. is fine.
    """
    h = text_cfg.get("hidden_size")
    n = text_cfg.get("num_hidden_layers")
    v = text_cfg.get("vocab_size")
    if not (h and n and v):
        return None

    # Attention block: respect grouped-query attention via num_key_value_heads.
    num_heads = text_cfg.get("num_attention_heads") or 1
    num_kv = text_cfg.get("num_key_value_heads") or num_heads
    head_dim = text_cfg.get("head_dim") or (h // num_heads if num_heads else h)
    q_dim = num_heads * head_dim
    kv_dim = num_kv * head_dim
    attn_per_layer = h * q_dim + 2 * h * kv_dim + q_dim * h  # Q, K, V, O

    # MLP block: dense (intermediate_size) or MoE (num_experts × moe_intermediate_size,
    # plus optional shared expert and router gate).
    inter = text_cfg.get("intermediate_size")
    num_experts = text_cfg.get("num_experts")
    moe_inter = text_cfg.get("moe_intermediate_size")
    shared_inter = text_cfg.get("shared_expert_intermediate_size")
    if inter:
        mlp_per_layer = 3 * h * inter
    elif num_experts and moe_inter:
        mlp_per_layer = num_experts * 3 * h * moe_inter
        if shared_inter:
            mlp_per_layer += 3 * h * shared_inter
        mlp_per_layer += h * num_experts  # router gate
    else:
        return None

    embed = v * h
    return embed + n * (attn_per_layer + mlp_per_layer) + v * h  # + lm_head (untied)


def extract_model_info(model_path: str) -> ModelInfo:
    """Parse config.json + measure directory to build a ModelInfo."""
    path = Path(model_path).expanduser()
    cfg_file = path / "config.json"
    info = ModelInfo(path=str(path), size_bytes=_dir_size_bytes(path))
    (info.weights_mtime_iso, info.weights_mtime_unix,
     info.weights_index_sha, info.hf_revision) = _weight_versioning(path)

    if not cfg_file.exists():
        return info

    try:
        cfg = json.loads(cfg_file.read_text())
    except (OSError, json.JSONDecodeError):
        return info

    info.architectures = list(cfg.get("architectures", []))
    info.model_type = cfg.get("model_type", "")

    text_cfg = cfg.get("text_config", cfg)
    info.num_hidden_layers = text_cfg.get("num_hidden_layers")
    info.vocab_size = text_cfg.get("vocab_size") or cfg.get("vocab_size")

    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    if quant:
        info.is_quantised = True
        info.base_bits = quant.get("bits")
        info.group_size = quant.get("group_size")
        info.mode = quant.get("mode") or "affine"
        # Per-tensor overrides are dict-valued entries in the quant config
        overrides = {k: v for k, v in quant.items() if isinstance(v, dict)}
        info.num_overrides = len(overrides)
        info.bit_distribution = {
            int(b): c
            for b, c in Counter(
                v.get("bits") for v in overrides.values() if v.get("bits") is not None
            ).items()
            if b is not None
        }
        info.override_categories = _categorise_overrides(list(overrides.keys()))
        info.quant_family = _detect_quant_family(path, quant, info.num_overrides)
    else:
        # bf16/fp16 reference with no quantization config — surface this
        # explicitly so reports and charts don't fall back to "unknown".
        info.quant_family = "unquantised"

    n_params = _estimate_param_count(text_cfg)
    if n_params and info.size_bytes:
        info.effective_bpw = (info.size_bytes * 8) / n_params

    return info
