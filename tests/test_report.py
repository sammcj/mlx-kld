"""Tests for mlx_kld.report — pure-Python, no MLX/Metal needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_kld.report import (
    HIGHER_IS_BETTER,
    NO_WINNER,
    _render_arch_table,
    _render_inference_table,
    _render_quality_per_bit_table,
    _render_summary_table,
    _render_versioning_table,
    render_markdown,
    render_markdown_from_json_paths,
    report_data_from_results,
)


def _result_dict(
    *,
    name: str,
    mean_kld: float,
    median_kld: float = 0.001,
    std_kld: float = 0.01,
    p95_kld: float = 0.005,
    p99_kld: float = 0.05,
    max_kld: float = 0.5,
    tok_s: float | None = 100.0,
    size_gb: float | None = 10.0,
    bpw: float | None = 4.0,
    family: str = "RTN affine (uniform)",
    base_bits: int | None = 4,
    group_size: int | None = 64,
    num_overrides: int = 0,
    bit_distribution: dict | None = None,
    override_categories: dict | None = None,
    sha: str | None = "abcdef0123456789",
    mtime: str | None = "2026-04-25T12:00:00",
    reference_info: dict | None = None,
    total_tokens: int = 1000,
) -> dict:
    return {
        "reference_model": "/models/reference",
        "compare_model": f"/models/{name}",
        "summary": {
            "num_prompts": 10,
            "total_tokens": total_tokens,
            "mean_kld": mean_kld,
            "median_kld": median_kld,
            "std_kld": std_kld,
            "max_kld": max_kld,
            "p95_kld": p95_kld,
            "p99_kld": p99_kld,
            "prefill_tokens_per_second": tok_s,
            "prefill_seconds": 1.0,
        },
        "reference_info": reference_info,
        "compare_info": {
            "path": f"/models/{name}",
            "size_gb": size_gb,
            "is_quantised": True,
            "base_bits": base_bits,
            "group_size": group_size,
            "mode": "affine",
            "quant_family": family,
            "num_overrides": num_overrides,
            "bit_distribution": bit_distribution or {},
            "override_categories": override_categories or {},
            "effective_bpw": bpw,
            "weights_index_sha": sha,
            "weights_mtime_iso": mtime,
        },
        "prompts": [],
    }


@pytest.fixture
def three_models() -> list[dict]:
    return [
        _result_dict(name="A", mean_kld=0.005, tok_s=300.0, bpw=8.0, size_gb=20.0),
        _result_dict(name="B", mean_kld=0.020, tok_s=350.0, bpw=4.0, size_gb=10.0),
        _result_dict(name="C", mean_kld=0.012, tok_s=320.0, bpw=6.0, size_gb=15.0),
    ]


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def test_report_data_from_results_extracts_rows(three_models):
    data = report_data_from_results(three_models)
    assert len(data.rows) == 3
    # Display name is now ``creator/model`` from the cmp path's parent + leaf
    names = {r.name for r in data.rows}
    assert names == {"models/A", "models/B", "models/C"}
    by_name = {r.name: r for r in data.rows}
    assert by_name["models/A"].mean_kld == 0.005
    assert by_name["models/B"].tok_s == 350.0
    assert by_name["models/C"].bpw == 6.0


def test_report_data_sorted_by_mean_kld_ascending(three_models):
    data = report_data_from_results(three_models)
    means = [r.mean_kld for r in data.rows]
    assert means == sorted(means)
    # Best (lowest mean_kld) is A → "models/A" with the new naming
    assert data.rows[0].name == "models/A"


def test_best_per_column_lowest_kld_wins(three_models):
    data = report_data_from_results(three_models)
    # Row index 0 is A after sorting (lowest mean_kld)
    assert data.best_per_column["mean_kld"] == 0


def test_best_per_column_highest_tok_s_wins(three_models):
    data = report_data_from_results(three_models)
    # B has the highest tok/s (350)
    by_name_idx = {r.name: i for i, r in enumerate(data.rows)}
    assert data.best_per_column["tok_s"] == by_name_idx["models/B"]


def test_best_per_column_no_winner_for_size_and_bpw(three_models):
    data = report_data_from_results(three_models)
    assert "size_gb" not in data.best_per_column
    assert "bpw" not in data.best_per_column
    assert "size_gb" in NO_WINNER
    assert "bpw" in NO_WINNER
    assert "tok_s" in HIGHER_IS_BETTER


def test_quality_per_bit_computed_and_ranked(three_models):
    data = report_data_from_results(three_models)
    by_name = {r.name: r for r in data.rows}
    # A: 0.005 / 8.0 = 0.000625; B: 0.020 / 4.0 = 0.005; C: 0.012 / 6.0 = 0.002
    assert by_name["models/A"].quality_per_bit == pytest.approx(0.000625)
    assert by_name["models/B"].quality_per_bit == pytest.approx(0.005)
    by_name_idx = {r.name: i for i, r in enumerate(data.rows)}
    assert data.best_per_column["quality_per_bit"] == by_name_idx["models/A"]


def test_bit_distribution_string_keys_coerced_to_int(three_models):
    # Real JSON has string keys for bit_distribution because JSON requires strings
    rows = list(three_models)
    rows[0]["compare_info"]["bit_distribution"] = {"5": 108, "6": 2}
    rows[0]["compare_info"]["num_overrides"] = 110
    data = report_data_from_results(rows)
    by_name = {r.name: r for r in data.rows}
    assert by_name["models/A"].bit_distribution == {5: 108, 6: 2}


# ---------------------------------------------------------------------------
# Renderer layer
# ---------------------------------------------------------------------------

def test_render_markdown_contains_required_sections(three_models):
    data = report_data_from_results(three_models)
    md = render_markdown(data)
    for heading in (
        "# mlx-kld report",
        "## Headline summary",
        "## Quality per bit",
        "## Quant architecture",
        "## Inference speed",
        "## Versioning",
        "## Reproducing",
        "## Notes",
    ):
        assert heading in md, f"missing heading: {heading}"
    assert "<!-- Add your interpretation here -->" in md


def test_render_markdown_handles_missing_tok_s(three_models):
    rows = list(three_models)
    rows[1]["summary"]["prefill_tokens_per_second"] = None
    data = report_data_from_results(rows)
    md = render_markdown(data)
    # No exception, model B's tok/s cell renders as the empty marker
    assert "—" in md


def test_render_markdown_handles_missing_compare_info():
    minimal = {
        "reference_model": "/ref",
        "compare_model": "/m1",
        "summary": {
            "mean_kld": 0.01, "median_kld": 0.001, "std_kld": 0.01,
            "max_kld": 0.5, "p95_kld": 0.005, "p99_kld": 0.05,
            "total_tokens": 100, "prefill_tokens_per_second": None,
        },
        "reference_info": None,
        "compare_info": None,
    }
    data = report_data_from_results([minimal])
    md = render_markdown(data)
    assert "## Headline summary" in md
    # Empty markers used where compare_info would have populated
    assert "—" in md


def test_render_markdown_chart_link_relative_path(tmp_path, three_models):
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"fake png")
    data = report_data_from_results(
        three_models, chart_path=str(chart), markdown_dir=tmp_path,
    )
    md = render_markdown(data)
    assert "![Quality comparison" in md
    assert "(./chart.png)" in md


def test_render_markdown_skips_chart_when_none(three_models):
    data = report_data_from_results(three_models, chart_path=None)
    md = render_markdown(data)
    assert "![" not in md


def test_render_markdown_single_model_omits_stars(three_models):
    data = report_data_from_results([three_models[0]])
    md = render_markdown(data)
    assert "★" not in md


def test_individual_table_renderers_are_self_contained(three_models):
    data = report_data_from_results(three_models)
    arch = _render_arch_table(data)
    assert arch.startswith("## Quant architecture")
    assert "RTN affine (uniform)" in arch
    summary = _render_summary_table(data)
    assert summary.startswith("## Headline summary")
    qpb = _render_quality_per_bit_table(data)
    assert qpb.startswith("## Quality per bit")
    inf = _render_inference_table(data)
    assert inf.startswith("## Inference speed")
    ver = _render_versioning_table(data)
    assert ver.startswith("## Versioning")


# ---------------------------------------------------------------------------
# Disk wrappers
# ---------------------------------------------------------------------------

def test_render_markdown_from_json_paths_roundtrip(tmp_path, three_models):
    json_paths = []
    for m in three_models:
        p = tmp_path / f"result_{Path(m['compare_model']).name}.json"
        p.write_text(json.dumps(m))
        json_paths.append(str(p))
    out_md = tmp_path / "report.md"
    written = render_markdown_from_json_paths(json_paths, str(out_md))
    assert Path(written).exists()
    text = Path(written).read_text()
    for name in ("A", "B", "C"):
        assert name in text


def test_reproduction_block_uses_load_reference_when_set(three_models):
    data = report_data_from_results(
        three_models,
        load_reference_path="/cache/ref.npz",
        extra_flags={"top_k_cache": 256},
    )
    md = render_markdown(data)
    assert "--load-reference /cache/ref.npz" in md
    assert "--top-k-cache 256" in md
    assert "--reference " not in md  # not when load is present


def test_reproduction_block_uses_reference_when_no_load(three_models):
    data = report_data_from_results(
        three_models, reference_path="/models/ref",
    )
    md = render_markdown(data)
    assert "--reference /models/ref" in md
    assert "--load-reference" not in md


def test_files_section_lists_chart_cache_and_jsons(tmp_path, three_models):
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"x")
    data = report_data_from_results(
        three_models,
        chart_path=str(chart),
        save_reference_path="/cache/ref.npz",
        output_prefix=str(tmp_path / "run"),
        markdown_dir=tmp_path,
    )
    md = render_markdown(data)
    assert "Reference cache" in md
    assert "Chart" in md
    assert "Per-model JSON" in md
    # The per-model JSON paths should follow cli.py's "{prefix}_{slug}.json" rule
    assert "run_A.json" in md
    assert "run_B.json" in md
    assert "run_C.json" in md


def test_versioning_table_includes_reference_when_present():
    rows = [
        _result_dict(
            name="A", mean_kld=0.005,
            reference_info={
                "weights_index_sha": "deadbeefcafe1234",
                "weights_mtime_iso": "2026-04-26T08:00:00",
                "size_gb": 55.5,
            },
        ),
    ]
    data = report_data_from_results(rows)
    ver = _render_versioning_table(data)
    assert "(reference)" in ver
    assert "deadbeefcafe" in ver


def test_total_tokens_is_per_comparison_not_summed(three_models):
    # Each model evaluates the same prompts; the header should show the
    # per-comparison count, not 3x it.
    for m in three_models:
        m["summary"]["total_tokens"] = 1500
    data = report_data_from_results(three_models)
    assert data.total_tokens == 1500


def test_repro_falls_back_to_reference_model_in_render_only(three_models):
    # No explicit reference_path/load_reference_path supplied — the renderer
    # should use the reference_model captured in the JSON.
    data = report_data_from_results(three_models)
    md = render_markdown(data)
    assert "--reference /models/reference" in md


def test_repro_includes_save_reference_when_set(three_models):
    data = report_data_from_results(
        three_models,
        reference_path="/models/ref",
        save_reference_path="/cache/ref.npz",
    )
    md = render_markdown(data)
    assert "--save-reference /cache/ref.npz" in md


def test_self_comparison_row_is_dropped(three_models, capsys):
    # Add a self-compare row with mean_kld == 0; it should be dropped from
    # the rendered tables, with a stderr note.
    rows = list(three_models)
    rows.append(_result_dict(name="REF", mean_kld=0.0, tok_s=400.0))
    data = report_data_from_results(rows)
    md = render_markdown(data)
    assert "`REF`" not in md
    err = capsys.readouterr().err
    assert "self-comparison" in err


def test_self_comparison_kept_when_all_rows_are_self():
    # If every row is a self-compare (e.g. --compare X X Y Y where X==reference
    # and Y==reference), keep them all rather than rendering an empty report.
    rows = [
        _result_dict(name="A", mean_kld=0.0),
        _result_dict(name="B", mean_kld=0.0),
    ]
    data = report_data_from_results(rows)
    md = render_markdown(data)
    assert "`models/A`" in md
    assert "`models/B`" in md


def test_self_comparison_kept_when_only_row():
    # If a single mean_kld==0 result is the only row, keep it (single-model
    # render shouldn't be silently empty).
    only_self = _result_dict(name="REF", mean_kld=0.0, tok_s=400.0)
    data = report_data_from_results([only_self])
    md = render_markdown(data)
    assert "`models/REF`" in md


def test_format_flags_handles_arbitrary_keys(three_models):
    # _format_flags should auto-emit any truthy entry without code changes.
    data = report_data_from_results(
        three_models,
        reference_path="/r",
        extra_flags={
            "top_k_cache": 256,                # numeric
            "no_chat_template": True,          # bool
            "json_summary_only": False,        # falsy: skipped
            "future_invented_flag": "abc",     # string, no code change needed
            "another_zero": 0,                 # falsy: skipped
        },
    )
    md = render_markdown(data)
    assert "--top-k-cache 256" in md
    assert "--no-chat-template" in md
    assert "--future-invented-flag abc" in md
    assert "--json-summary-only" not in md
    assert "--another-zero" not in md


def test_inference_table_sorted_by_tok_s_desc(three_models):
    data = report_data_from_results(three_models)
    inf = _render_inference_table(data)
    lines = inf.splitlines()
    # Body lines start after header + separator (2 metadata lines + 2 table lines)
    body = [l for l in lines if l.startswith("| `")]
    # Order should be B (350), C (320), A (300)
    assert "`models/B`" in body[0]
    assert "`models/C`" in body[1]
    assert "`models/A`" in body[2]
