"""Chart generation for mlx-kld results.

Renders a single comparison chart that shows the most important KLD metric
(mean KL divergence vs reference) per model, with quant-family colour and
marker, log-scale x-axis, and bpw + tok/s annotations.

matplotlib is an optional dependency (install with ``pip install mlx-kld[chart]``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


# Categorical palette: chosen for colour-blind safety (Wong / IBM tweak) and
# distinguishability from typical "rainbow" academic charts. Each quant
# family also gets a distinct marker shape so the chart stays readable
# when printed in greyscale.
FAMILY_STYLE: dict[str, dict] = {
    "RTN affine (uniform)": {"color": "#0072B2", "marker": "o", "label": "RTN (uniform)"},
    "RTN affine (mixed)":   {"color": "#56B4E9", "marker": "o", "label": "RTN (mixed)"},
    "oQ":                   {"color": "#009E73", "marker": "D", "label": "oQ"},
    "Unsloth Dynamic":      {"color": "#E69F00", "marker": "s", "label": "Unsloth Dynamic"},
    "AutoRound":            {"color": "#CC79A7", "marker": "^", "label": "AutoRound"},
    "DWQ":                  {"color": "#8E44AD", "marker": "v", "label": "DWQ"},
    "DFlash":               {"color": "#34495E", "marker": "<", "label": "DFlash"},
    "unquantised":          {"color": "#444444", "marker": "*", "label": "Unquantised"},
    "unknown":              {"color": "#999999", "marker": "x", "label": "Other"},
}


def _require_matplotlib():
    """Import matplotlib lazily and emit a useful error if it's missing."""
    try:
        import matplotlib
        # Headless-safe backend; only set if no GUI backend has been configured.
        try:
            matplotlib.get_backend()
        except Exception:  # pragma: no cover
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError as e:
        raise ImportError(
            "Chart generation requires matplotlib. "
            "Install with: pip install 'mlx-kld[chart]'"
        ) from e


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


def _pick_label_offsets(plottable: list[dict], ax) -> list[tuple]:
    """Choose a label offset per point that avoids overlapping any other marker.

    Tries right → left → above → below in order; falls back to right if all
    four positions collide. Uses display (pixel) coords so it works with the
    log-scaled x-axis. Approximates label width from character count and
    fontsize=9; marker radius is generous to account for the s=180 dot.
    """
    marker_disp = ax.transData.transform(
        [(r["mean_kld"], r["size_gb"]) for r in plottable]
    )
    marker_radius_px = 14
    char_width_px = 6.0
    label_height_px = 12
    candidates: list[tuple[tuple[int, int], str, str]] = [
        ((10, 0), "left", "center"),
        ((-10, 0), "right", "center"),
        ((0, 12), "center", "bottom"),
        ((0, -12), "center", "top"),
    ]
    chosen: list[tuple] = []
    for i, r in enumerate(plottable):
        label_w = max(40, len(r["name"]) * char_width_px)
        my_x, my_y = marker_disp[i]
        for offset, ha, va in candidates:
            lx, ly = my_x + offset[0], my_y + offset[1]
            x0 = (lx - label_w if ha == "right"
                  else lx - label_w / 2 if ha == "center"
                  else lx)
            x1 = x0 + label_w
            y0 = (ly - label_height_px if va == "top"
                  else ly - label_height_px / 2 if va == "center"
                  else ly)
            y1 = y0 + label_height_px
            collides = False
            for j, (jx, jy) in enumerate(marker_disp):
                if j == i:
                    continue
                if not (x1 < jx - marker_radius_px or x0 > jx + marker_radius_px
                        or y1 < jy - marker_radius_px or y0 > jy + marker_radius_px):
                    collides = True
                    break
            if not collides:
                chosen.append((offset, ha, va))
                break
        else:
            chosen.append(((10, 0), "left", "center"))
    return chosen


def _row_from_result_dict(d: dict) -> Optional[dict]:
    """Extract chartable fields from a result JSON or in-memory ComparisonResult dict."""
    summary = d.get("summary") or {}
    cmp_info = d.get("compare_info") or {}
    ref_info = d.get("reference_info") or {}
    mean = summary.get("mean_kld")
    if mean is None:
        return None
    cmp_path = d.get("compare_model") or cmp_info.get("path") or ""
    by_mode = d.get("summary_by_mode") or {}
    return {
        "name": _creator_slash_model(cmp_path),
        "mean_kld": float(mean),
        "median_kld": summary.get("median_kld"),
        "p99_kld": summary.get("p99_kld"),
        "max_kld": summary.get("max_kld"),
        # Per-mode stats so the chart can plot the more rigorous mode when
        # both ran.
        "mode": summary.get("mode") or ("long" if by_mode.get("long") else "short"),
        "short_stats": by_mode.get("short"),
        "long_stats": by_mode.get("long"),
        # Comparison-model metadata (annotations beside each point)
        "size_gb": cmp_info.get("size_gb"),
        "bpw": cmp_info.get("effective_bpw"),
        "family": cmp_info.get("quant_family") or "unknown",
        "tok_s": summary.get("prefill_tokens_per_second"),
        # Reference-model metadata (used by title/subtitle, identical across rows
        # but stored on every row for convenience)
        "reference_model": d.get("reference_model"),
        "ref_index_sha": (ref_info.get("weights_index_sha") or "")[:12],
        "ref_mtime_iso": (ref_info.get("weights_mtime_iso") or "")[:10],
    }


def _rows_from_json_files(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        try:
            d = json.loads(Path(p).read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  warning: skipping {p} ({e})")
            continue
        row = _row_from_result_dict(d)
        if row is not None:
            rows.append(row)
    return rows


def render_quality_chart(
    rows: list[dict],
    output_path: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> str:
    """Write a Pareto-style quality vs size scatter for the comparison set.

    Both axes carry data: x = mean KL divergence (log, lower is better),
    y = file size on disk in GB (linear, lower is less memory & ~faster
    decode on Apple Silicon since decode is memory-bandwidth bound).
    Models are coloured and marker-coded by quant family; each point is
    labelled with the model name. The lower-left of the chart is the
    Pareto-efficient region; dominated quants (worse KL *and* larger size)
    appear up-and-to-the-right of a comparable model.

    Prefill tok/s is intentionally not shown — it doesn't reflect decode
    latency, which is the user-felt cost. Size is the better speed proxy
    on this hardware class and is already encoded by the y-axis.

    Args:
        rows: List of dicts with keys: name, mean_kld, family, size_gb,
            and optionally reference_model and ref_* fields. Use the
            output of ``_rows_from_json_files`` or ``_row_from_result_dict``.
        output_path: Destination file path. Extension picks the format
            (``.png``, ``.svg``, ``.pdf``); defaults to ``.png`` if absent.
        title: Optional override for the chart title.
        subtitle: Optional override for the small grey subtitle line.

    Returns:
        Absolute path of the file written.
    """
    if not rows:
        raise ValueError("No rows to plot.")

    # Drop rows missing either axis or with non-positive KLD (log-scale x can't
    # plot those; mean_kld == 0 typically means a self-comparison sanity check
    # that doesn't belong on a quality chart).
    plottable = [
        r for r in rows
        if r.get("mean_kld") is not None
        and r["mean_kld"] > 0
        and r.get("size_gb")
    ]
    if not plottable:
        raise ValueError(
            "No rows have positive mean_kld and a size_gb — cannot render scatter."
        )

    _, plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    fig.subplots_adjust(left=0.10, right=0.78, top=0.86, bottom=0.13)

    # Sort once by KLD so collision detection has a stable order.
    plottable.sort(key=lambda r: r["mean_kld"])

    # Pass 1: plot markers only. Labels need final axis limits to compute
    # display-space collision boxes, so we annotate after limits are set.
    for r in plottable:
        family = r.get("family", "unknown")
        style = FAMILY_STYLE.get(family, FAMILY_STYLE["unknown"])
        ax.scatter(
            r["mean_kld"], r["size_gb"],
            s=180, c=style["color"], marker=style["marker"],
            edgecolors="#222222", linewidths=0.7,
            zorder=3,
        )

    # Axes
    ax.set_xscale("log")
    ax.set_xlabel(
        "Mean KL divergence vs reference  (log scale, lower is better)",
        fontsize=10, labelpad=8,
    )
    ax.set_ylabel("Size on disk (GB)  ·  higher = more memory & likely slower",
                  fontsize=10, labelpad=8)
    ax.tick_params(axis="both", labelsize=9)

    # Subtle grid on both axes for a scatter
    ax.grid(True, which="major", linestyle=":", color="#cccccc", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Headroom so annotations don't get clipped on the right
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 0.7, xmax * 5.0)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0, ymin - 2), ymax + 2)

    # Pass 2: place labels using collision-aware offsets so a label never sits
    # on top of another marker. Tries right → left → above → below in order.
    label_offsets = _pick_label_offsets(plottable, ax)
    for r, (offset, ha, va) in zip(plottable, label_offsets):
        ax.annotate(
            r["name"],
            (r["mean_kld"], r["size_gb"]),
            xytext=offset, textcoords="offset points",
            ha=ha, va=va, fontsize=9, color="#222222",
        )

    # Pareto-region cue: faint shaded triangle in the lower-left
    ax.text(
        0.02, 0.04,
        "← better quality   ·   ↓ smaller / faster",
        transform=ax.transAxes, fontsize=9, color="#888888",
        ha="left", va="bottom", style="italic",
    )

    # Title — include the evaluation mode so a long-mode chart is never
    # mistaken for a short-mode one (or vice versa).
    ref_name = Path(plottable[0].get("reference_model") or "").name or "reference"
    mode = plottable[0].get("mode") or "short"
    mode_label = "long-mode (WikiText-2 streamed)" if mode == "long" else "short-mode prompts"
    if title is None:
        title = f"Quantisation quality vs size — {ref_name}  ·  {mode_label}"
    ax.set_title(title, fontsize=13, weight="bold", loc="left", pad=18)

    # Subtitle (date/sha if reference_info was captured at run time)
    if subtitle is None:
        sub_bits = []
        ref_sha = rows[0].get("ref_index_sha") or ""
        ref_date = rows[0].get("ref_mtime_iso") or ""
        if ref_sha:
            sub_bits.append(f"ref sha {ref_sha[:8]}")
        if ref_date:
            sub_bits.append(ref_date)
        subtitle = "  ·  ".join(sub_bits) if sub_bits else None
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle,
            transform=ax.transAxes, fontsize=9, color="#666666",
            ha="left", va="bottom",
        )

    # Legend: one entry per family that actually appears in the plot
    # (use ``plottable``, not the raw ``rows`` — otherwise filtered-out
    # self-comparison rows still pollute the legend with an "Other" entry).
    families_present = []
    for r in plottable:
        f = r.get("family", "unknown")
        if f not in families_present:
            families_present.append(f)
    handles = [
        plt.Line2D(
            [0], [0],
            marker=FAMILY_STYLE.get(f, FAMILY_STYLE["unknown"])["marker"],
            color="white",
            markerfacecolor=FAMILY_STYLE.get(f, FAMILY_STYLE["unknown"])["color"],
            markeredgecolor="#222222", markersize=10,
            label=FAMILY_STYLE.get(f, FAMILY_STYLE["unknown"])["label"],
            linewidth=0,
        )
        for f in families_present
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, fontsize=9, title="Quant family", title_fontsize=9,
    )

    # Resolve output path (default to .png)
    out = Path(output_path)
    if out.suffix.lower() not in (".png", ".svg", ".pdf"):
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out.resolve())


def render_chart_from_json_paths(
    json_paths: list[str], output_path: str, **kwargs: Any
) -> str:
    """Convenience: load result JSONs, render a chart in one call."""
    rows = _rows_from_json_files(json_paths)
    return render_quality_chart(rows, output_path, **kwargs)


def render_chart_from_results(
    results: list, output_path: str, **kwargs: Any
) -> str:
    """Convenience: render directly from in-memory ComparisonResult objects."""
    rows = []
    for r in results:
        d = r.to_dict(detail=False, top_k=0)
        row = _row_from_result_dict(d)
        if row is not None:
            rows.append(row)
    return render_quality_chart(rows, output_path, **kwargs)
