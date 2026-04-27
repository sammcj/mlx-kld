"""Markdown report generation for mlx-kld results.

Three layers, all in this module so they can be imported independently by
a future web UI without pulling in matplotlib or the comparison runner:

- Data: ``ReportData`` + ``ReportRow`` dataclasses, built by
  ``report_data_from_results`` from either ``ComparisonResult`` objects
  or their JSON dict form.
- Renderers: pure string-returning functions (``render_markdown`` plus per-
  table helpers). No I/O, no formatting decisions hidden in the data layer.
- Disk wrapper: two convenience functions (``render_markdown_from_results``
  and ``render_markdown_from_json_paths``) that mirror ``chart.py``.

stdlib-only on purpose.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# Single source of truth for column-direction logic. cli._print_summary_table
# imports these so terminal and markdown stay consistent.
HIGHER_IS_BETTER: set[str] = {"tok_s"}
NO_WINNER: set[str] = {"size_gb", "bpw"}

_KLD_COLUMNS: tuple[str, ...] = (
    "mean_kld", "median_kld", "std_kld", "p95_kld", "p99_kld", "max_kld",
)
# Every column the headline table marks a winner on
_RANKED_COLUMNS: tuple[str, ...] = _KLD_COLUMNS + ("tok_s",)

_EMPTY = "—"


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

@dataclass
class ReportRow:
    """One comparison model's entry in the report.

    Headline KLD fields (``mean_kld`` etc.) reflect the primary mode for that
    row — long if long-mode data is present, otherwise short. Per-mode raw
    stats are kept in ``short_stats`` / ``long_stats`` so the renderer can
    show stratified tables when both modes ran.
    """

    name: str
    full_path: str
    family: str
    size_gb: float | None
    bpw: float | None
    base_bits: int | None
    group_size: int | None
    num_overrides: int
    bit_distribution: dict[int, int]
    override_categories: dict[str, int]
    weights_index_sha: str | None
    weights_mtime_iso: str | None
    mean_kld: float
    median_kld: float
    std_kld: float
    p95_kld: float
    p99_kld: float
    max_kld: float
    tok_s: float | None
    quality_per_bit: float | None
    tok_s_per_gb: float | None
    primary_mode: str = "short"
    short_stats: dict | None = None
    long_stats: dict | None = None


@dataclass
class ReproductionSpec:
    reference_path: str | None = None
    load_reference_path: str | None = None
    save_reference_path: str | None = None
    compare_paths: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportData:
    reference_model: str | None
    ref_index_sha: str | None
    ref_mtime_iso: str | None
    ref_size_gb: float | None
    total_tokens: int | None
    rows: list[ReportRow]
    best_per_column: dict[str, int]
    files: list[tuple[str, str]]
    chart_path: str | None
    reproduction: ReproductionSpec
    run_date_iso: str = ""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _result_to_dict(r: Any) -> dict:
    """Accept either a ``ComparisonResult`` or a raw dict."""
    if isinstance(r, dict):
        return r
    if hasattr(r, "to_dict"):
        return r.to_dict(detail=False, top_k=0)
    raise TypeError(f"Unsupported result type: {type(r).__name__}")


def _coerce_bit_distribution(raw: Any) -> dict[int, int]:
    """JSON serialises int keys as strings; coerce back."""
    if not raw:
        return {}
    out: dict[int, int] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _creator_slash_model(path: str) -> str:
    """Format a model path as ``creator/model`` (e.g. ``mlx-community/Qwen3.6-27B-6bit``).

    Falls back to just the model name when there's no meaningful parent
    directory (e.g. for bare names or root-level paths).
    """
    if not path:
        return "(unknown)"
    p = Path(path)
    name = p.name
    if not name:
        return "(unknown)"
    parent = p.parent.name
    if parent and parent not in ("", "/", "."):
        return f"{parent}/{name}"
    return name


def _row_from_result(d: dict) -> ReportRow | None:
    summary = d.get("summary") or {}
    cmp_info = d.get("compare_info") or {}
    mean = summary.get("mean_kld")
    if mean is None:
        return None
    full_path = d.get("compare_model") or cmp_info.get("path") or ""
    name = _creator_slash_model(full_path)
    bpw = cmp_info.get("effective_bpw")
    size_gb = cmp_info.get("size_gb")
    tok_s = summary.get("prefill_tokens_per_second")
    sha = cmp_info.get("weights_index_sha")
    mtime = cmp_info.get("weights_mtime_iso")
    by_mode = d.get("summary_by_mode") or {}
    short_stats = by_mode.get("short")
    long_stats = by_mode.get("long")
    primary_mode = summary.get("mode") or ("long" if long_stats else "short")
    return ReportRow(
        name=name,
        full_path=full_path,
        family=cmp_info.get("quant_family") or "unknown",
        size_gb=float(size_gb) if size_gb is not None else None,
        bpw=float(bpw) if bpw is not None else None,
        base_bits=cmp_info.get("base_bits"),
        group_size=cmp_info.get("group_size"),
        num_overrides=int(cmp_info.get("num_overrides") or 0),
        bit_distribution=_coerce_bit_distribution(cmp_info.get("bit_distribution")),
        override_categories=dict(cmp_info.get("override_categories") or {}),
        weights_index_sha=(sha[:12] if isinstance(sha, str) else None),
        weights_mtime_iso=(mtime[:10] if isinstance(mtime, str) else None),
        mean_kld=float(mean),
        median_kld=float(summary.get("median_kld") or 0.0),
        std_kld=float(summary.get("std_kld") or 0.0),
        p95_kld=float(summary.get("p95_kld") or 0.0),
        p99_kld=float(summary.get("p99_kld") or 0.0),
        max_kld=float(summary.get("max_kld") or 0.0),
        tok_s=float(tok_s) if tok_s is not None else None,
        quality_per_bit=(float(mean) / float(bpw)) if bpw else None,
        tok_s_per_gb=(float(tok_s) / float(size_gb)) if (tok_s and size_gb) else None,
        primary_mode=primary_mode,
        short_stats=short_stats,
        long_stats=long_stats,
    )


def _compute_best_per_column(rows: list[ReportRow]) -> dict[str, int]:
    """Index of the winning row for each ranked column. Skipped if <2 rows."""
    if len(rows) < 2:
        return {}
    best: dict[str, int] = {}
    for col in _RANKED_COLUMNS:
        vals = [(getattr(r, col), i) for i, r in enumerate(rows) if getattr(r, col) is not None]
        if not vals:
            continue
        if col in HIGHER_IS_BETTER:
            best[col] = max(vals)[1]
        else:
            best[col] = min(vals)[1]
    # Quality-per-bit: lower is better
    qpb = [(r.quality_per_bit, i) for i, r in enumerate(rows) if r.quality_per_bit is not None]
    if qpb:
        best["quality_per_bit"] = min(qpb)[1]
    # tok_s per GB: higher is better
    tpg = [(r.tok_s_per_gb, i) for i, r in enumerate(rows) if r.tok_s_per_gb is not None]
    if tpg:
        best["tok_s_per_gb"] = max(tpg)[1]
    return best


def _expected_json_paths(output_prefix: str, rows: list[ReportRow]) -> list[str]:
    """Reproduce cli.py's per-model JSON naming for the files-generated section."""
    if not output_prefix:
        return []
    prefix = Path(output_prefix)
    if len(rows) == 1:
        single = prefix.with_suffix(".json") if prefix.suffix != ".json" else prefix
        return [str(single)]
    # cli.py's _model_slug uses Path(model_path).name (just the leaf), not the
    # display name which now includes the creator prefix.
    return [
        str(prefix.parent / f"{prefix.name}_{Path(r.full_path).name}.json")
        for r in rows
    ]


def _build_files_section(
    *,
    rows: list[ReportRow],
    output_prefix: str | None,
    chart_path: str | None,
    save_reference_path: str | None,
    load_reference_path: str | None,
    markdown_dir: Path | None,
) -> tuple[list[tuple[str, str]], str | None]:
    """Return ((label, displayed_path) list, relative_chart_path_for_embed)."""
    files: list[tuple[str, str]] = []

    ref_cache = save_reference_path or load_reference_path
    if ref_cache:
        files.append(("Reference cache", _maybe_relativise(ref_cache, markdown_dir)))

    rel_chart = None
    if chart_path:
        rel_chart = _maybe_relativise(chart_path, markdown_dir)
        files.append(("Chart", rel_chart))

    for p in _expected_json_paths(output_prefix or "", rows):
        files.append(("Per-model JSON", _maybe_relativise(p, markdown_dir)))

    return files, rel_chart


def _maybe_relativise(path_str: str, base: Path | None) -> str:
    """Make ``path_str`` relative to ``base`` when both share a common root."""
    if not base:
        return path_str
    try:
        rel = os.path.relpath(path_str, start=str(base))
    except ValueError:
        # Different drives on Windows — fall back to absolute.
        return path_str
    # Don't return things like ../../foo unless that's actually shorter than absolute.
    if rel.startswith(".."):
        return path_str
    if not rel.startswith("."):
        rel = "./" + rel
    return rel


def report_data_from_results(
    results: Iterable[Any],
    *,
    chart_path: str | None = None,
    output_prefix: str | None = None,
    save_reference_path: str | None = None,
    load_reference_path: str | None = None,
    reference_path: str | None = None,
    extra_flags: dict[str, Any] | None = None,
    markdown_dir: str | os.PathLike | None = None,
    run_date_iso: str | None = None,
) -> ReportData:
    """Build a ``ReportData`` from comparison results.

    Accepts both ``ComparisonResult`` objects and the dicts they serialise to
    (so render-only paths and live runs share one builder).
    """
    dicts = [_result_to_dict(r) for r in results]

    all_rows = [r for r in (_row_from_result(d) for d in dicts) if r is not None]
    # A self-comparison (--compare X X for sanity, mean_kld == 0) would
    # otherwise win every star and dominate the table. Drop it from the
    # rendered report — but only if at least one non-self row remains, so a
    # report consisting purely of sanity checks doesn't render empty.
    non_self = [r for r in all_rows if r.mean_kld != 0.0]
    if non_self and len(non_self) < len(all_rows):
        rows = non_self
        print(
            f"  note: dropped {len(all_rows) - len(non_self)} self-comparison "
            f"row(s) (mean KLD == 0) from the markdown report.",
            file=sys.stderr,
        )
    else:
        rows = all_rows
    rows.sort(key=lambda r: r.mean_kld)

    # Reference info: take the first non-null reference_info we see.
    ref_model: str | None = None
    ref_info: dict = {}
    for d in dicts:
        if d.get("reference_model") and ref_model is None:
            ref_model = d.get("reference_model")
        if d.get("reference_info"):
            ref_info = d["reference_info"]
            break

    # Each model evaluates the same prompts, so total_tokens is per-comparison,
    # not summed across rows (otherwise an N-model run would inflate by N×).
    total_tokens: int | None = None
    for d in dicts:
        t = (d.get("summary") or {}).get("total_tokens")
        if t is not None:
            total_tokens = int(t)
            break

    md_dir = Path(markdown_dir) if markdown_dir is not None else None
    files, rel_chart = _build_files_section(
        rows=rows,
        output_prefix=output_prefix,
        chart_path=chart_path,
        save_reference_path=save_reference_path,
        load_reference_path=load_reference_path,
        markdown_dir=md_dir,
    )

    # Render-only mode: fall back to the reference_model field captured in
    # the JSON when the caller didn't supply an explicit --reference path.
    repro_reference = reference_path
    if not repro_reference and not load_reference_path:
        repro_reference = ref_model
    repro = ReproductionSpec(
        reference_path=repro_reference,
        load_reference_path=load_reference_path,
        save_reference_path=save_reference_path,
        compare_paths=[r.full_path for r in rows if r.full_path],
        flags=dict(extra_flags or {}),
    )

    if run_date_iso is None:
        run_date_iso = datetime.now(timezone.utc).astimezone().date().isoformat()

    ref_sha = ref_info.get("weights_index_sha") if ref_info else None
    ref_mtime = ref_info.get("weights_mtime_iso") if ref_info else None
    ref_size_gb = ref_info.get("size_gb") if ref_info else None

    return ReportData(
        reference_model=ref_model,
        ref_index_sha=ref_sha[:12] if isinstance(ref_sha, str) else None,
        ref_mtime_iso=ref_mtime[:10] if isinstance(ref_mtime, str) else None,
        ref_size_gb=float(ref_size_gb) if ref_size_gb is not None else None,
        total_tokens=total_tokens,
        rows=rows,
        best_per_column=_compute_best_per_column(rows),
        files=files,
        chart_path=rel_chart,
        reproduction=repro,
        run_date_iso=run_date_iso,
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _fmt_kld(v: float | None) -> str:
    return f"{v:.5f}" if v is not None else _EMPTY


def _fmt_size(v: float | None) -> str:
    return f"{v:.2f} GB" if v is not None else _EMPTY


def _fmt_bpw(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else _EMPTY


def _fmt_tok_s(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else _EMPTY


def _star(value_str: str, is_best: bool) -> str:
    return f"**{value_str}** ★" if is_best else value_str


def _bit_distribution_str(dist: dict[int, int]) -> str:
    if not dist:
        return _EMPTY
    return ", ".join(f"{b}-bit×{c}" for b, c in sorted(dist.items()))


def _override_categories_str(cats: dict[str, int]) -> str:
    if not cats:
        return _EMPTY
    return ", ".join(f"{k}×{v}" for k, v in cats.items())


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table. Pads columns for raw-readability."""
    if not rows:
        widths = [len(h) for h in headers]
    else:
        widths = [
            max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)
        ]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    body_lines = [
        "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def _render_header(data: ReportData) -> str:
    ref_name = Path(data.reference_model).name if data.reference_model else "(reference unknown)"
    lines = [f"# mlx-kld report — {ref_name}", ""]
    meta: list[str] = []
    if data.reference_model:
        meta.append(f"- Reference model: `{data.reference_model}`")
    if data.ref_index_sha:
        meta.append(f"- Reference weights sha[:12]: `{data.ref_index_sha}`")
    if data.ref_mtime_iso:
        meta.append(f"- Reference mtime: {data.ref_mtime_iso}")
    if data.ref_size_gb is not None:
        meta.append(f"- Reference size: {data.ref_size_gb:.2f} GB")
    if data.total_tokens is not None:
        meta.append(f"- Tokens per comparison: {data.total_tokens}")
    meta.append(f"- Report date: {data.run_date_iso}")
    lines.extend(meta)
    return "\n".join(lines) + "\n"


def _render_chart_link(data: ReportData) -> str:
    if not data.chart_path:
        return ""
    ref = Path(data.reference_model).name if data.reference_model else "reference"
    alt = f"Quality comparison: KL divergence vs {ref}"
    return f"![{alt}]({data.chart_path})\n"


def _render_summary_table(data: ReportData) -> str:
    """Render the headline KLD summary.

    When both short and long modes have data, render two stacked tables (one
    per mode) so the reader can see how each quant ranks under each regime.
    Otherwise render a single combined table — the original layout — using
    the per-row primary mode's numbers.
    """
    has_short = any(r.short_stats for r in data.rows)
    has_long = any(r.long_stats for r in data.rows)

    if has_short and has_long:
        out = [
            "## Headline summary",
            "",
            (
                "Two evaluation modes ran. Short-mode and long-mode means are "
                "not directly comparable — they sample different regimes — but "
                "rank-ordering across quants should agree if the quant has no "
                "long-context-specific failure mode."
            ),
            "",
            _render_per_mode_table(data, "short"),
            "",
            _render_per_mode_table(data, "long"),
        ]
        return "\n".join(out) + "\n"

    # Single-mode case (the original layout).
    headers = [
        "Model", "Size", "Eff. bpw",
        "Mean KLD", "Median KLD", "P95 KLD", "P99 KLD", "Max KLD",
        "Prefill tok/s",
    ]
    body: list[list[str]] = []
    best = data.best_per_column
    for i, r in enumerate(data.rows):
        row = [
            f"`{r.name}`",
            _fmt_size(r.size_gb),
            _fmt_bpw(r.bpw),
            _star(_fmt_kld(r.mean_kld), best.get("mean_kld") == i),
            _star(_fmt_kld(r.median_kld), best.get("median_kld") == i),
            _star(_fmt_kld(r.p95_kld), best.get("p95_kld") == i),
            _star(_fmt_kld(r.p99_kld), best.get("p99_kld") == i),
            _star(_fmt_kld(r.max_kld), best.get("max_kld") == i),
            _star(_fmt_tok_s(r.tok_s), best.get("tok_s") == i),
        ]
        body.append(row)
    out = ["## Headline summary", "", _md_table(headers, body)]
    if best:
        out.append("")
        out.append("★ = best on that metric. (Lowest KLD is best; highest tok/s is best.)")
    return "\n".join(out) + "\n"


def _render_per_mode_table(data: ReportData, mode: str) -> str:
    """Render a KLD table for a single mode (``"short"`` or ``"long"``)."""
    label = "Short mode" if mode == "short" else "Long mode"
    headers = [
        "Model", "Size", "Eff. bpw",
        "Mean KLD", "Median KLD", "P95 KLD", "P99 KLD", "Max KLD",
    ]
    # Best-per-column for this mode.
    rows_with_stats = [
        (i, r) for i, r in enumerate(data.rows)
        if (r.short_stats if mode == "short" else r.long_stats)
    ]
    if not rows_with_stats:
        return f"### {label}\n\n(no data)\n"
    best: dict[str, int] = {}
    for col in ("mean_kld", "median_kld", "p95_kld", "p99_kld", "max_kld"):
        candidates = [
            (s.get(col), i) for i, r in rows_with_stats
            for s in [r.short_stats if mode == "short" else r.long_stats]
            if s and s.get(col) is not None
        ]
        if candidates:
            best[col] = min(candidates)[1]
    body: list[list[str]] = []
    for i, r in rows_with_stats:
        s = r.short_stats if mode == "short" else r.long_stats
        if not s:
            continue
        body.append([
            f"`{r.name}`",
            _fmt_size(r.size_gb),
            _fmt_bpw(r.bpw),
            _star(_fmt_kld(s.get("mean_kld")), best.get("mean_kld") == i),
            _star(_fmt_kld(s.get("median_kld")), best.get("median_kld") == i),
            _star(_fmt_kld(s.get("p95_kld")), best.get("p95_kld") == i),
            _star(_fmt_kld(s.get("p99_kld")), best.get("p99_kld") == i),
            _star(_fmt_kld(s.get("max_kld")), best.get("max_kld") == i),
        ])
    return f"### {label}\n\n{_md_table(headers, body)}\n"


def _render_quality_per_bit_table(data: ReportData) -> str:
    headers = ["Model", "Mean KLD / bpw"]
    body: list[list[str]] = []
    best = data.best_per_column.get("quality_per_bit")
    # Sort ascending by quality_per_bit for this view; rows missing bpw go last.
    indexed = list(enumerate(data.rows))
    indexed.sort(key=lambda iv: (iv[1].quality_per_bit is None, iv[1].quality_per_bit or 0.0))
    for i, r in indexed:
        body.append([
            f"`{r.name}`",
            _star(_fmt_kld(r.quality_per_bit), best == i),
        ])
    return "\n".join(["## Quality per bit", "", _md_table(headers, body)]) + "\n"


def _render_arch_table(data: ReportData) -> str:
    headers = [
        "Model", "Quant family", "Base bits", "Overrides",
        "Tensors elevated", "sha[:12]", "mtime",
    ]
    body: list[list[str]] = []
    for r in data.rows:
        bits = r.base_bits if r.base_bits is not None else _EMPTY
        gs = r.group_size
        base_str = f"{bits}" + (f"/gs{gs}" if gs is not None else "")
        if r.num_overrides == 0:
            overrides = "none"
        else:
            overrides = f"{r.num_overrides} ({_bit_distribution_str(r.bit_distribution)})"
        body.append([
            f"`{r.name}`",
            r.family,
            base_str,
            overrides,
            _override_categories_str(r.override_categories),
            f"`{r.weights_index_sha}`" if r.weights_index_sha else _EMPTY,
            r.weights_mtime_iso or _EMPTY,
        ])
    return "\n".join(["## Quant architecture", "", _md_table(headers, body)]) + "\n"


def _render_inference_table(data: ReportData) -> str:
    headers = ["Model", "Size", "tok/s", "tok/s per GB"]
    indexed = list(enumerate(data.rows))
    # Sort by tok/s descending; rows without tok/s go last.
    indexed.sort(key=lambda iv: (iv[1].tok_s is None, -(iv[1].tok_s or 0.0)))
    best_tok = data.best_per_column.get("tok_s")
    best_tpg = data.best_per_column.get("tok_s_per_gb")
    body: list[list[str]] = []
    for i, r in indexed:
        body.append([
            f"`{r.name}`",
            _fmt_size(r.size_gb),
            _star(_fmt_tok_s(r.tok_s), best_tok == i),
            _star(_fmt_tok_s(r.tok_s_per_gb), best_tpg == i),
        ])
    return "\n".join(["## Inference speed", "", _md_table(headers, body)]) + "\n"


def _render_versioning_table(data: ReportData) -> str:
    headers = ["Model", "weights_index_sha[:12]", "mtime"]
    body: list[list[str]] = []
    if data.ref_index_sha or data.ref_mtime_iso:
        ref_label = (Path(data.reference_model).name if data.reference_model else "reference")
        body.append([
            f"`{ref_label}` (reference)",
            f"`{data.ref_index_sha}`" if data.ref_index_sha else _EMPTY,
            data.ref_mtime_iso or _EMPTY,
        ])
    for r in data.rows:
        body.append([
            f"`{r.name}`",
            f"`{r.weights_index_sha}`" if r.weights_index_sha else _EMPTY,
            r.weights_mtime_iso or _EMPTY,
        ])
    return "\n".join(["## Versioning", "", _md_table(headers, body)]) + "\n"


def _render_files_section(data: ReportData) -> str:
    if not data.files:
        return ""
    lines = ["## Files generated", ""]
    for label, path in data.files:
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def _render_reproduction(data: ReportData) -> str:
    spec = data.reproduction
    parts = ["mlx-kld"]
    if spec.load_reference_path:
        parts.append(f"  --load-reference {spec.load_reference_path}")
    elif spec.reference_path:
        parts.append(f"  --reference {spec.reference_path}")
    if spec.save_reference_path:
        parts.append(f"  --save-reference {spec.save_reference_path}")
    if spec.compare_paths:
        parts.append("  --compare")
        for p in spec.compare_paths:
            parts.append(f"    {p}")
    flag_lines = _format_flags(spec.flags)
    parts.extend(flag_lines)
    body = " \\\n".join(parts) if len(parts) > 1 else parts[0]
    return "\n".join([
        "## Reproducing",
        "",
        "```bash",
        body,
        "```",
    ]) + "\n"


def _format_flags(flags: dict[str, Any]) -> list[str]:
    """Render any truthy flag using the snake_case → kebab-case CLI convention.

    True booleans become bare ``--flag``; numeric/string values become
    ``--flag value``. Falsy entries (None / 0 / "" / False) are skipped, so
    new flags wired into ``extra_flags`` show up in the reproduction block
    without needing to extend this function.
    """
    out: list[str] = []
    for key, value in flags.items():
        if not value:
            continue
        cli_name = "--" + key.replace("_", "-")
        if value is True:
            out.append(f"  {cli_name}")
        else:
            out.append(f"  {cli_name} {value}")
    return out


def _render_notes_stub() -> str:
    return "## Notes\n\n<!-- Add your interpretation here -->\n"


def render_markdown(data: ReportData) -> str:
    """Render a complete markdown report from ``ReportData``."""
    sections = [
        _render_header(data),
        _render_chart_link(data),
        _render_summary_table(data),
        _render_quality_per_bit_table(data),
        _render_arch_table(data),
        _render_inference_table(data),
        _render_versioning_table(data),
        _render_files_section(data),
        _render_reproduction(data),
        _render_notes_stub(),
    ]
    return "\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# Disk wrappers
# ---------------------------------------------------------------------------

def _warn_missing_chart(chart_path: str | None) -> None:
    if chart_path and not Path(chart_path).exists():
        print(
            f"  warning: chart path '{chart_path}' does not exist yet; linking anyway.",
            file=sys.stderr,
        )


def render_markdown_from_results(
    results: Iterable[Any],
    output_path: str,
    *,
    chart_path: str | None = None,
    output_prefix: str | None = None,
    save_reference_path: str | None = None,
    load_reference_path: str | None = None,
    reference_path: str | None = None,
    extra_flags: dict[str, Any] | None = None,
) -> str:
    out = Path(output_path)
    if out.suffix.lower() != ".md":
        out = out.with_suffix(".md")
    _warn_missing_chart(chart_path)
    data = report_data_from_results(
        results,
        chart_path=chart_path,
        output_prefix=output_prefix,
        save_reference_path=save_reference_path,
        load_reference_path=load_reference_path,
        reference_path=reference_path,
        extra_flags=extra_flags,
        markdown_dir=out.resolve().parent,
    )
    out.write_text(render_markdown(data))
    return str(out.resolve())


def render_markdown_from_json_paths(
    json_paths: list[str],
    output_path: str,
    *,
    chart_path: str | None = None,
    save_reference_path: str | None = None,
    load_reference_path: str | None = None,
    reference_path: str | None = None,
    extra_flags: dict[str, Any] | None = None,
) -> str:
    dicts: list[dict] = []
    for p in json_paths:
        try:
            dicts.append(json.loads(Path(p).read_text()))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  warning: skipping {p} ({e})", file=sys.stderr)
    if not dicts:
        raise ValueError("No valid result JSONs to render.")
    return render_markdown_from_results(
        dicts,
        output_path,
        chart_path=chart_path,
        save_reference_path=save_reference_path,
        load_reference_path=load_reference_path,
        reference_path=reference_path,
        extra_flags=extra_flags,
    )
