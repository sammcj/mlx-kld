"""CLI entry point for mlx-kld."""

import argparse
import json
import sys
from pathlib import Path

from .report import (
    HIGHER_IS_BETTER as _REPORT_HIGHER_IS_BETTER,
    NO_WINNER as _REPORT_NO_WINNER,
    _creator_slash_model as _display_name,
)
# .compare is imported lazily inside main() so render-only paths
# (--render-chart-from / --render-markdown-from) don't trigger MLX/Metal init.


def _model_slug(model_path: str) -> str:
    """Turn a model path or HF repo into a short filename-safe slug."""
    # Use the last path component (works for both local paths and HF repos)
    return Path(model_path).name


def _merge_existing_into_result(result, out_path: Path) -> None:
    """If ``out_path`` is a previous run's JSON for the same model, merge any
    modes the current run *didn't* cover into ``result`` so writing the new
    JSON preserves them.

    Modes covered by the current run win — we never overwrite fresh numbers
    with stale ones.
    """
    if not out_path.exists():
        return
    try:
        existing = json.loads(out_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    result.merge_external_modes_from_dict(existing)


def _filter_dflash(paths: list[str], *, kind: str) -> list[str]:
    """Drop any path whose leaf name contains 'dflash' (case-insensitive).

    DFlash variants are speculative-decoding draft models — prefill KLD against
    a regular reference doesn't measure anything meaningful for them.
    """
    kept, dropped = [], []
    for p in paths:
        if "dflash" in Path(p).name.lower():
            dropped.append(p)
        else:
            kept.append(p)
    if dropped:
        print(
            f"  note: skipping {len(dropped)} DFlash model(s) from {kind} "
            f"(speculative-decoding drafts; prefill KLD doesn't apply): "
            + ", ".join(Path(p).name for p in dropped),
            file=sys.stderr,
        )
    return kept


def _short_info(info: dict) -> str:
    """Compact one-line summary of a model_info dict for terminal display."""
    if not info:
        return ""
    size_gb = info.get("size_gb")
    if not info.get("is_quantised"):
        return f"{size_gb} GB, unquantised" if size_gb else "unquantised"
    bpw = info.get("effective_bpw")
    family = info.get("quant_family", "?")
    base = info.get("base_bits")
    gs = info.get("group_size")
    n_over = info.get("num_overrides", 0)
    bits_dist = info.get("bit_distribution") or {}
    bd_str = (
        ", ".join(f"{c}@{b}b" for b, c in sorted(bits_dist.items())) if bits_dist else "uniform"
    )
    bpw_str = f"~{bpw:.2f} bpw" if bpw else ""
    return (
        f"{size_gb} GB, {bpw_str}, {family}, base={base}b/gs{gs}, "
        f"overrides={n_over} [{bd_str}]"
    )


def _print_results(result, top_k: int = 0) -> None:
    """Print formatted results for a single comparison to stdout."""
    from .metrics import MODE_LONG, MODE_SHORT

    s = result

    print(f"\n{'=' * 50}")
    print(f"  KL Divergence Results")
    print(f"{'=' * 50}")
    print(f"  Reference: {s.reference_model}")
    if s.reference_info:
        print(f"             {_short_info(s.reference_info)}")
    print(f"  Compare:   {s.compare_model}")
    if s.compare_info:
        print(f"             {_short_info(s.compare_info)}")
    print(f"{'=' * 50}")
    print(f"  Prompts:     {len(s.prompt_results)}")
    print(f"  Tokens:      {s.total_tokens}")
    if s.prefill_tokens_per_second:
        print(f"  Prefill:     {s.prefill_tokens_per_second:.1f} tok/s "
              f"({s.prefill_seconds:.1f}s on {s.total_tokens} tokens)")
    print(f"{'─' * 50}")

    short_stats = s.stats_for_mode(MODE_SHORT)
    long_stats = s.stats_for_mode(MODE_LONG)
    if short_stats and long_stats:
        # Both modes present: show per-mode tables. Don't print a combined
        # summary number — averaging different sample regimes is misleading.
        for label, stats in (("Short mode", short_stats), ("Long mode", long_stats)):
            print(f"  {label}  ({stats['scored_tokens']} scored tokens)")
            print(f"    Mean KLD:   {stats['mean_kld']:.6f}")
            print(f"    Median KLD: {stats['median_kld']:.6f}")
            print(f"    P95 KLD:    {stats['p95_kld']:.6f}")
            print(f"    P99 KLD:    {stats['p99_kld']:.6f}")
            print(f"    Max KLD:    {stats['max_kld']:.6f}")
    else:
        print(f"  Mode:        {s.primary_mode}")
        print(f"  Mean KLD:    {s.mean_kld:.6f}")
        print(f"  Median KLD:  {s.median_kld:.6f}")
        print(f"  Std KLD:     {s.std_kld:.6f}")
        print(f"  P95 KLD:     {s.percentile(95):.6f}")
        print(f"  P99 KLD:     {s.percentile(99):.6f}")
        print(f"  Max KLD:     {s.max_kld:.6f}")

    # Find which prompt/token has the max in this run's data.
    if s.prompt_results:
        max_prompt_idx = 0
        max_pos = 0
        max_val = 0.0
        for i, pr in enumerate(s.prompt_results):
            if pr.max_kld > max_val:
                max_val = pr.max_kld
                max_prompt_idx = i
                max_pos = pr.max_kld_position

        max_pr = s.prompt_results[max_prompt_idx]
        max_tok = (
            max_pr.token_strings[max_pos]
            if max_pr.token_strings and max_pos < len(max_pr.token_strings)
            else "?"
        )
        print(
            f"             (prompt {max_prompt_idx + 1}, "
            f'position {max_pos}, token "{max_tok.strip()}")'
        )
    print(f"{'=' * 50}")

    # Per-prompt breakdown
    if len(s.prompt_results) > 1:
        print(f"\n  Per-prompt breakdown:")
        print(f"  {'#':<4} {'Tokens':<8} {'Mean':>10} {'Max':>10}  Prompt")
        print(f"  {'─' * 60}")
        for i, pr in enumerate(s.prompt_results):
            prompt_preview = pr.prompt[:40] + ("..." if len(pr.prompt) > 40 else "")
            print(
                f"  {i + 1:<4} {pr.num_tokens:<8} "
                f"{pr.mean_kld:>10.6f} {pr.max_kld:>10.6f}  "
                f"{prompt_preview}"
            )
        print()

    # Top-K divergent tokens
    if top_k > 0:
        print(f"\n  Top-{top_k} most divergent tokens (across all prompts):")
        print(f"  {'Prompt':<8} {'Pos':<6} {'KLD':>10}  Token")
        print(f"  {'─' * 50}")

        all_tokens = []
        for pi, pr in enumerate(s.prompt_results):
            for td in pr.top_k_divergent(top_k):
                all_tokens.append((pi, td))

        all_tokens.sort(key=lambda x: x[1].kld, reverse=True)
        for pi, td in all_tokens[:top_k]:
            print(
                f"  {pi + 1:<8} {td.position:<6} "
                f'{td.kld:>10.6f}  "{td.token_str.strip()}"'
            )
        print()


def _strip_common_prefix(names: list[str]) -> tuple[list[str], str]:
    """Return ``(stripped_names, prefix)``. Only strips when the shared prefix
    is at least 3 chars long and stripping leaves every name non-empty —
    so "Qwen3.6-35B-4bit" / "Qwen3.6-35B-oQ6" / ... become "4bit" / "oQ6" /
    … with the common prefix surfaced in the table caption instead.
    """
    import os
    if len(names) < 2:
        return list(names), ""
    prefix = os.path.commonprefix(names)
    if len(prefix) < 3 or any(len(n) - len(prefix) < 1 for n in names):
        return list(names), ""
    return [n[len(prefix):] for n in names], prefix


def _print_summary_table(results) -> None:
    """Print a side-by-side summary table for multiple comparison models."""
    if len(results) < 2:
        return

    # Display the same creator/model form the chart and markdown report use,
    # so a multi-creator comparison (e.g. mlx-community/X vs Jundot/X) shows
    # who built each quant rather than ambiguous leaf names.
    raw_names = [_display_name(r.compare_model) for r in results]
    slugs, common_prefix = _strip_common_prefix(raw_names)
    # Width sized to the longest name. Cap at 40 so a rogue path doesn't blow
    # the table off the side of the terminal — anything longer is truncated
    # with an ellipsis. Common-prefix stripping above already shortens
    # same-creator runs (e.g. shared "Qwen3.6-35B-" prefix).
    col = max(12, min(40, max(len(s) for s in slugs)))
    label_col = 30
    header_parts = [f"{'Model':<{label_col}}"]
    divider_parts = [f"{'─' * label_col}"]
    # Map terminal label -> ReportData column key, so the higher_is_better /
    # no_winner sets in report.py stay the single source of truth.
    label_to_key = {
        "Mean KLD":     "mean_kld",
        "Median KLD":   "median_kld",
        "Std KLD":      "std_kld",
        "P95 KLD":      "p95_kld",
        "P99 KLD":      "p99_kld",
        "Max KLD":      "max_kld",
        "Size (GB)":    "size_gb",
        "Eff. bpw":     "bpw",
        "Prefill tok/s":"tok_s",
    }
    metric_rows = {label: [] for label in label_to_key}

    for r, slug in zip(results, slugs):
        if len(slug) > col:
            slug = slug[: col - 1] + "…"
        header_parts.append(f"{slug:>{col}}")
        divider_parts.append("─" * col)
        metric_rows["Mean KLD"].append(r.mean_kld)
        metric_rows["Median KLD"].append(r.median_kld)
        metric_rows["Std KLD"].append(r.std_kld)
        metric_rows["P95 KLD"].append(r.percentile(95))
        metric_rows["P99 KLD"].append(r.percentile(99))
        metric_rows["Max KLD"].append(r.max_kld)
        info = r.compare_info or {}
        metric_rows["Size (GB)"].append(info.get("size_gb") or 0.0)
        metric_rows["Eff. bpw"].append(info.get("effective_bpw") or 0.0)
        metric_rows["Prefill tok/s"].append(r.prefill_tokens_per_second or 0.0)

    sep = "  "
    # When both modes ran, the per-row stats fall back to the primary mode
    # (long). Label the table so a reader looking at numbers in isolation
    # knows which regime they're from.
    primary_modes = {r.primary_mode for r in results}
    mode_label = (
        f" ({primary_modes.pop()} mode)"
        if len(primary_modes) == 1
        else ""
    )
    print(f"\n{'=' * 50}")
    print(f"  Summary Comparison{mode_label}")
    if common_prefix:
        print(f"  (model names share prefix '{common_prefix}', stripped for readability)")
    print(f"{'=' * 50}")
    print(sep.join(header_parts))
    print(sep.join(divider_parts))

    for label, vals in metric_rows.items():
        key = label_to_key[label]
        if key in _REPORT_NO_WINNER:
            best_idx = -1
        elif key in _REPORT_HIGHER_IS_BETTER:
            best_idx = vals.index(max(vals))
        else:
            best_idx = vals.index(min(vals))

        row = [f"{label:<30}"]
        # Smaller numbers (size/bpw/tok-s) read better with fewer decimals.
        if label in {"Size (GB)", "Eff. bpw"}:
            fmt = "{:.2f}"
        elif label == "Prefill tok/s":
            fmt = "{:.1f}"
        else:
            fmt = "{:.6f}"
        for i, v in enumerate(vals):
            cell = fmt.format(v)
            marker = " *" if i == best_idx else "  "
            row.append(f"{cell:>{col - 2}}{marker}")
        print(sep.join(row))

    print()
    print("  * = best on that metric (lowest KLD, highest tok/s)")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="mlx-kld",
        description="Measure KL divergence between MLX language model output distributions.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Path or HF repo for the reference model. Not needed with --load-reference.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=[],
        metavar="MODEL",
        help="One or more paths/HF repos for comparison models.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[],
        help="One or more prompt strings to evaluate.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help=(
            "Path to a text file with one prompt per line. "
            "Takes precedence over --short / --long when given. "
            "If neither this, --prompts, --short, nor --long is given, "
            "--short is the default and the bundled sample_prompts/prompts.txt loads."
        ),
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help=(
            "Run the short-prompt mode (the bundled sample_prompts/prompts.txt set). "
            "This is the default if no prompt source is specified. "
            "May be combined with --long to run both regimes in one invocation."
        ),
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help=(
            "Run the long-mode streamed-corpus evaluation (WikiText-2 chunked at "
            "n_ctx tokens, scoring only the second half of each chunk per the "
            "llama.cpp convention). Numbers from this mode are methodologically "
            "comparable to published GPTQ/AWQ/llama.cpp tables. May be combined "
            "with --short."
        ),
    )
    parser.add_argument(
        "--long-corpus",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Override the long-mode corpus path. "
            "Defaults to the bundled corpora/wiki.test.raw."
        ),
    )
    parser.add_argument(
        "--long-ctx",
        type=int,
        default=2048,
        metavar="N",
        help=(
            "Long-mode chunk size in tokens. Default 2048 (GPTQ/AWQ academic "
            "convention). Each chunk scores positions [N/2, N)."
        ),
    )
    parser.add_argument(
        "--long-chunks",
        type=int,
        default=32,
        metavar="N",
        help=(
            "Number of long-mode chunks to take from the start of the corpus. "
            "Default 32 (~32k scored tokens at the default n_ctx — tight SE "
            "without bloating runtime)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path prefix for JSON results. With one model: saved as-is. "
            "With multiple models: one file per model using the prefix + model slug "
            "(e.g. --output results → results_Qwen3.5-27b-oQ5.json)."
        ),
    )
    parser.add_argument(
        "--save-reference",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "After running the reference model, save its logits to this path (.npz). "
            "Future runs can skip the reference model entirely with --load-reference."
        ),
    )
    parser.add_argument(
        "--load-reference",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Load previously saved reference logits from this .npz file instead of "
            "running the reference model. Prompts are loaded from the file."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Show the top-K most divergent tokens per model.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Don't apply the model's chat template to prompts.",
    )
    parser.add_argument(
        "--top-k-cache",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Store the reference cache as top-K sparse log-probs (recommended). "
            "Reduces 27B Qwen reference cache from ~8 GB/prompt to ~8 MB/prompt "
            "with negligible KL approximation error. Try K=256 or K=512. "
            "Default 0 = full vocab (legacy)."
        ),
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Split forward passes into chunks of N tokens to reduce peak "
            "activation memory. Useful on memory-constrained Macs with long "
            "(8K+) prompts. Default 0 = single forward pass."
        ),
    )
    parser.add_argument(
        "--json-summary-only",
        action="store_true",
        help=(
            "Emit only summary stats in JSON output (drops per-token KLD, "
            "token IDs, and token strings). Useful for batch comparisons "
            "across many models where per-token data would blow up file size. "
            "--top-k still adds top-K tokens per prompt regardless of this flag."
        ),
    )
    parser.add_argument(
        "--chart",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Render a quality-comparison chart of mean KLD per model to PATH. "
            "Extension picks the format (.png/.svg/.pdf, default .png). "
            "Requires matplotlib: pip install 'mlx-kld[chart]'."
        ),
    )
    parser.add_argument(
        "--render-chart-from",
        nargs="+",
        default=None,
        metavar="JSON",
        help=(
            "Render a chart from previously saved result JSON files without "
            "running any comparison. Use with --chart to specify the output path."
        ),
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Render a markdown report of the comparison to PATH. "
            "If omitted, a report is still written by default to "
            "'{--output prefix}.md' when --output is set, or "
            "'./mlx-kld-report.md' otherwise. Use --no-markdown to disable. "
            "Stdlib-only — no extra dependencies."
        ),
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable the default markdown report. Has no effect with --markdown.",
    )
    parser.add_argument(
        "--render-markdown-from",
        nargs="+",
        default=None,
        metavar="JSON",
        help=(
            "Render a markdown report from previously saved result JSON files "
            "without running any comparison. Use with --markdown to specify the "
            "output path."
        ),
    )

    args = parser.parse_args()

    # --- Render-only mode: build chart from existing JSON, no comparison run ---
    if args.render_chart_from:
        if not args.chart:
            print(
                "Error: --render-chart-from requires --chart to specify the output path.",
                file=sys.stderr,
            )
            sys.exit(1)
        from .chart import render_chart_from_json_paths
        try:
            out = render_chart_from_json_paths(args.render_chart_from, args.chart)
            print(f"Chart saved to {out}", file=sys.stderr)
        except (ImportError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        # If the user also asked for a markdown render in the same invocation,
        # let it fall through to the markdown render-only branch below.
        if not args.render_markdown_from:
            return

    # --- Render-only mode: build markdown from existing JSON, no comparison run ---
    if args.render_markdown_from:
        if not args.markdown:
            print(
                "Error: --render-markdown-from requires --markdown to specify the output path.",
                file=sys.stderr,
            )
            sys.exit(1)
        from .report import render_markdown_from_json_paths
        try:
            out = render_markdown_from_json_paths(
                args.render_markdown_from,
                args.markdown,
                chart_path=args.chart,
            )
            print(f"Markdown report saved to {out}", file=sys.stderr)
        except (OSError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Validate: need either --reference or --load-reference
    if args.load_reference is None and args.reference is None:
        print(
            "Error: provide --reference (or --load-reference to use a saved cache).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.compare:
        print("Error: provide at least one model via --compare.", file=sys.stderr)
        sys.exit(1)

    # DFlash variants are speculative-execution draft models, not standard MLX
    # quants — prefill KLD against a normal reference is a category error, so
    # we always exclude them.
    args.compare = _filter_dflash(args.compare, kind="--compare")
    if args.reference and "dflash" in Path(args.reference).name.lower():
        print(
            f"Error: --reference cannot be a DFlash model ({args.reference}). "
            "DFlash variants are speculative-execution drafts, not regular quants.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.compare:
        print("Error: no comparison models left after DFlash filter.", file=sys.stderr)
        sys.exit(1)

    # Resolve which evaluation modes to run.
    #
    # Precedence:
    #   - explicit --prompts / --prompts-file overrides everything (literal)
    #   - --long alone runs only long mode
    #   - --short or default (no flags) runs only short mode
    #   - --short --long runs both
    #
    # The default is short-only because it's the fast smoke test (~2 min for
    # 5 models). --long is opt-in.
    repo_root = Path(__file__).resolve().parent.parent
    sample_dir = repo_root / "sample_prompts"
    explicit_prompts = bool(args.prompts) or bool(args.prompts_file)
    run_short = args.short or (not args.long and not explicit_prompts)
    run_long = args.long

    prompts = list(args.prompts)
    if args.prompts_file:
        pf = Path(args.prompts_file)
        if not pf.exists():
            print(f"Error: prompts file not found: {pf}", file=sys.stderr)
            sys.exit(1)
        prompts.extend(
            line.strip() for line in pf.read_text().splitlines() if line.strip()
        )
    elif run_short and args.load_reference is None:
        default_pf = sample_dir / "prompts.txt"
        if not default_pf.exists():
            print(
                f"Error: --short requested but {default_pf} is missing.",
                file=sys.stderr,
            )
            sys.exit(1)
        lines = [
            line.strip() for line in default_pf.read_text().splitlines() if line.strip()
        ]
        print(f"  short mode: loaded {len(lines)} prompt(s) from {default_pf.name}")
        prompts.extend(lines)

    # Resolve long-mode corpus path.
    long_corpus_path: str | None = None
    if run_long:
        if args.long_corpus:
            resolved = args.long_corpus
        else:
            default_corpus = repo_root / "corpora" / "wiki.test.raw"
            if not default_corpus.exists():
                print(
                    f"Error: --long requested but bundled corpus is missing: {default_corpus}.",
                    file=sys.stderr,
                )
                sys.exit(1)
            resolved = str(default_corpus)
        long_corpus_path = resolved
        print(
            f"  long mode: corpus={Path(resolved).name}, "
            f"n_ctx={args.long_ctx}, chunks={args.long_chunks}"
        )

    if args.load_reference is None and not prompts and not run_long:
        print(
            "Error: nothing to evaluate. Pass --short, --long, --prompts, or --prompts-file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Run comparisons
    from .compare import compare
    results = compare(
        reference=args.reference or "",
        comparisons=args.compare,
        prompts=prompts,
        use_chat_template=not args.no_chat_template,
        save_ref=args.save_reference,
        load_ref=args.load_reference,
        sparse_k=args.top_k_cache,
        chunk_tokens=args.chunk_tokens,
        long_corpus_path=long_corpus_path,
        long_n_ctx=args.long_ctx,
        long_num_chunks=args.long_chunks,
    )

    # Print results for each model
    for result in results:
        _print_results(result, top_k=args.top_k)

    # Summary table when comparing multiple models
    if len(results) > 1:
        _print_summary_table(results)

    # Save JSON output. Before writing each file, splice in any other-mode
    # data from a prior invocation's JSON so running --short and --long in
    # separate invocations leaves both sets of numbers in the final file.
    if args.output:
        out_prefix = Path(args.output)
        detail = not args.json_summary_only
        if len(results) == 1:
            out_path = out_prefix.with_suffix(".json") if out_prefix.suffix != ".json" else out_prefix
            _merge_existing_into_result(results[0], out_path)
            out_path.write_text(json.dumps(
                results[0].to_dict(detail=detail, top_k=args.top_k),
                indent=2,
            ))
            print(f"Results saved to {out_path}", file=sys.stderr)
        else:
            for result in results:
                slug = _model_slug(result.compare_model)
                out_path = out_prefix.parent / f"{out_prefix.name}_{slug}.json"
                _merge_existing_into_result(result, out_path)
                out_path.write_text(json.dumps(
                    result.to_dict(detail=detail, top_k=args.top_k),
                    indent=2,
                ))
                print(f"Results saved to {out_path}", file=sys.stderr)

    # Generate chart if requested (after comparison run)
    if args.chart:
        from .chart import render_chart_from_results
        try:
            out = render_chart_from_results(results, args.chart)
            print(f"Chart saved to {out}", file=sys.stderr)
        except (ImportError, ValueError) as e:
            print(f"Chart skipped: {e}", file=sys.stderr)

    # Markdown report is on by default; resolve the destination from
    # --markdown if given, then --output's prefix, then a cwd fallback.
    if not args.no_markdown:
        if args.markdown:
            md_path = args.markdown
        elif args.output:
            out_prefix = Path(args.output)
            md_path = str(
                out_prefix if out_prefix.suffix == ".md"
                else out_prefix.with_suffix(".md")
            )
        else:
            md_path = "mlx-kld-report.md"
        from .report import render_markdown_from_results
        try:
            out = render_markdown_from_results(
                results,
                md_path,
                chart_path=args.chart,
                output_prefix=args.output,
                save_reference_path=args.save_reference,
                load_reference_path=args.load_reference,
                reference_path=args.reference,
                extra_flags={
                    "top_k_cache": args.top_k_cache,
                    "chunk_tokens": args.chunk_tokens,
                    "no_chat_template": args.no_chat_template,
                    "json_summary_only": args.json_summary_only,
                    "top_k": args.top_k,
                },
            )
            print(f"Markdown report saved to {out}", file=sys.stderr)
        except (OSError, ValueError) as e:
            print(f"Markdown report skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
