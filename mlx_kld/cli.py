"""CLI entry point for mlx-kld."""

import argparse
import json
import sys
from pathlib import Path

from .compare import compare


def _print_results(result, top_k: int = 0) -> None:
    """Print formatted results to stdout."""
    s = result

    print(f"\n{'=' * 50}")
    print(f"  KL Divergence Results")
    print(f"{'=' * 50}")
    print(f"  Reference: {s.reference_model}")
    print(f"  Compare:   {s.compare_model}")
    print(f"{'=' * 50}")
    print(f"  Prompts:     {len(s.prompt_results)}")
    print(f"  Tokens:      {s.total_tokens}")
    print(f"{'─' * 50}")
    print(f"  Mean KLD:    {s.mean_kld:.6f}")
    print(f"  Median KLD:  {s.median_kld:.6f}")
    print(f"  Std KLD:     {s.std_kld:.6f}")
    print(f"  P95 KLD:     {s.percentile(95):.6f}")
    print(f"  P99 KLD:     {s.percentile(99):.6f}")
    print(f"  Max KLD:     {s.max_kld:.6f}")

    # Find which prompt/token has the max
    max_prompt_idx = 0
    max_pos = 0
    max_val = 0.0
    for i, pr in enumerate(s.prompt_results):
        if pr.max_kld > max_val:
            max_val = pr.max_kld
            max_prompt_idx = i
            max_pos = pr.max_kld_position

    max_pr = s.prompt_results[max_prompt_idx]
    max_tok = max_pr.token_strings[max_pos] if max_pr.token_strings else "?"
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

        # Gather all tokens with their prompt index
        all_tokens = []
        for pi, pr in enumerate(s.prompt_results):
            for td in pr.top_k_divergent(top_k):
                all_tokens.append((pi, td))

        # Sort by KLD descending and take top_k
        all_tokens.sort(key=lambda x: x[1].kld, reverse=True)
        for pi, td in all_tokens[:top_k]:
            print(
                f"  {pi + 1:<8} {td.position:<6} "
                f'{td.kld:>10.6f}  "{td.token_str.strip()}"'
            )
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="mlx-kld",
        description="Measure KL divergence between MLX language model output distributions.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path or HF repo for the reference model (e.g. full-precision).",
    )
    parser.add_argument(
        "--compare",
        required=True,
        help="Path or HF repo for the comparison model (e.g. quantized).",
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
        help="Path to a text file with one prompt per line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed JSON results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Show the top-K most divergent tokens.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Don't apply the model's chat template to prompts.",
    )

    args = parser.parse_args()

    # Collect prompts
    prompts = list(args.prompts)
    if args.prompts_file:
        pf = Path(args.prompts_file)
        if not pf.exists():
            print(f"Error: prompts file not found: {pf}", file=sys.stderr)
            sys.exit(1)
        prompts.extend(
            line.strip() for line in pf.read_text().splitlines() if line.strip()
        )

    if not prompts:
        print(
            "Error: provide at least one prompt via --prompts or --prompts-file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Run comparison
    result = compare(
        reference=args.reference,
        comparison=args.compare,
        prompts=prompts,
        use_chat_template=not args.no_chat_template,
    )

    # Print results
    _print_results(result, top_k=args.top_k)

    # Save JSON if requested
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Results saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
