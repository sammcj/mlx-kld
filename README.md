# mlx-kld

Measure KL divergence between MLX language model output distributions. Compare a reference model (typically full-precision or 8-bit) against one or more quantised variants to rank quant quality, with optional speed and architecture context.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional extras
pip install -e ".[chart]"   # matplotlib for chart rendering
pip install -e ".[dev]"     # pytest for tests
```

## Quick start

Build a reference cache once, compare any number of quants against it later:

```bash
# Stage 1: cache the reference (8-bit is a fine stand-in for bf16; KL between
# them is ~1e-4, ~1000x smaller than typical 4-bit divergence). Pointing
# --compare at the reference doubles as a sanity check (self-KLD must be 0).
mlx-kld \
  --reference ~/.lmstudio/models/mlx-community/Qwen3.6-27B-8bit \
  --compare  ~/.lmstudio/models/mlx-community/Qwen3.6-27B-8bit \
  --prompts-file sample_prompts/prompts.txt \
  --top-k-cache 256 \
  --save-reference qwen27b_ref

# Stage 2: rank as many quants as you like against the cached reference.
# The reference model is never reloaded.
mlx-kld \
  --load-reference qwen27b_ref \
  --compare \
    ~/.lmstudio/models/mlx-community/Qwen3.6-27B-4bit \
    ~/.lmstudio/models/mlx-community/Qwen3.6-27B-6bit \
    ~/.lmstudio/models/Jundot/Qwen3.6-27B-oQ4 \
    ~/.lmstudio/models/Jundot/Qwen3.6-27B-oQ6 \
  --top-k 20 \
  --json-summary-only \
  --chart results/quality.png \
  --output results/run
```

For long-context prompts (8k+) on memory-constrained Macs, add `--chunk-tokens 2048`. KV state carries across chunks so the result is numerically equivalent to an un-chunked pass.

## Options

| Flag                     | Description                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `--reference`            | Path or HF repo for the reference model.                                                             |
| `--compare`              | One or more paths/HF repos for comparison models.                                                    |
| `--prompts`              | One or more prompt strings.                                                                          |
| `--prompts-file`         | Text file with one prompt per line.                                                                  |
| `--no-chat-template`     | Tokenise raw prompts instead of applying the model's chat template.                                  |
| `--save-reference PATH`  | Save reference logits to a `.npz` file after running.                                                |
| `--load-reference PATH`  | Load previously saved reference; skips loading the reference model.                                  |
| `--top-k-cache K`        | Save reference as top-K sparse log-probs (~500-1000x smaller cache, rank-preserving). Try K=256.     |
| `--chunk-tokens N`       | Split forward passes into chunks of N tokens with a shared KV cache. Lower peak memory, same result. |
| `--top-k N`              | Show the N most divergent tokens per model in the terminal.                                          |
| `--output PATH_PREFIX`   | Write per-model JSON results.                                                                        |
| `--json-summary-only`    | Drop per-token arrays from JSON (KB-sized output instead of MB-sized).                               |
| `--chart PATH`           | Render a quality-comparison chart to PATH (`.png`/`.svg`/`.pdf`). Requires `mlx-kld[chart]`.         |
| `--render-chart-from`    | Render a chart from previously saved result JSONs without running a comparison.                      |
| `--markdown PATH`        | Override the default markdown report path. (Markdown is on by default.)                              |
| `--no-markdown`          | Disable the default markdown report.                                                                 |
| `--render-markdown-from` | Render a markdown report from previously saved result JSONs without running a comparison.            |

## What you get

For every comparison the tool reports, in the terminal and JSON output:

- **KLD summary stats**: mean, median, std, P95, P99, max
- **Prefill throughput**: tokens/sec (measured during the comparison forward pass — zero extra cost)
- **Model metadata**: size on disk, effective bpw, quant family (RTN / oQ / Unsloth Dynamic / AutoRound / DWQ / DFlash / unquantised / unknown), base bits, group size, per-tensor override count + bit distribution + categorisation by tensor role
- **Versioning**: SHA-256 prefix of `model.safetensors.index.json` and mtime of the newest weight file — lets you tell whether two runs were against the same bytes-on-disk
- **Top-K most divergent tokens** (with `--top-k`) for qualitative debugging

A multi-model run prints a side-by-side summary table marking the best (lowest KLD, highest tok/s) for each row.

## Markdown report

A human-readable markdown report is written by default after every comparison. It contains the headline summary table, quality-per-bit ranking, quant architecture details, inference-speed table, versioning info, files-generated list, and a reproduction command. Stdlib only (no matplotlib needed). If you also pass `--chart`, the chart is embedded as a relative image link.

Default path resolution:

- `--markdown PATH` is honoured if set.
- Otherwise, if `--output PREFIX` is set, the report is written to `{PREFIX}.md`.
- Otherwise, it lands at `./mlx-kld-report.md`.

Pass `--no-markdown` to disable.

Re-render from existing result JSONs without re-running the comparison:

```bash
mlx-kld --render-markdown-from results/run_*.json --markdown results/report.md
```

## Chart output

`--chart PATH` writes a Pareto-style scatter of quantisation quality vs size:

- X axis: mean KL divergence vs reference (log scale, lower is better)
- Y axis: file size on disk in GB (linear). Smaller = less memory and faster decode on memory-bandwidth-bound Apple Silicon
- Lower-left is the Pareto-efficient region. A quant up-and-to-the-right of another is dominated (worse quality *and* larger)
- Points coloured and shape-coded by quant family, readable in greyscale and colour-blind safe
- Each point labelled with the model in `creator/model` form (e.g. `mlx-community/Qwen3.6-27B-6bit`)
- Self-comparison rows (mean KLD = 0) are filtered out of the plot
- Title shows the reference model; subtitle includes the reference weights SHA and mtime when present

Prefill tok/s is intentionally not plotted. It doesn't reflect decode latency, which is the user-felt cost. On this hardware class file size is the better speed proxy and is already encoded by the y-axis.

Re-render charts from existing results without re-running the comparison:

```bash
mlx-kld --render-chart-from results/run_*.json --chart results/quality.png
```

## Why a top-K sparse cache

A dense reference cache stores the full vocabulary distribution at every token position. For Qwen-class vocabularies (~248k tokens) that's about 8 GB per 8k-token prompt. The top-K sparse cache stores only the K most-likely log-probs per position plus a single tail-mass scalar — roughly 500-1000x smaller.

The approximation lumps the tail vocab into one bin and assumes the comparison model distributes its tail mass similarly. Empirically (see `results/qwen3.6-mlx-results.md`, Phase D) this introduces a ~3-5% systematic underestimate of the absolute KL value but is rank-preserving across comparison models — which is the property quant ranking actually depends on. K=256 is a sensible default; the tests verify rank preservation.

## How it works

1. Load the **reference model**, run a forward pass on each prompt, store the log-softmax outputs as numpy arrays (sparsified if `--top-k-cache` is set).
2. **Unload the reference model** (or skip entirely with `--load-reference`).
3. For each **comparison model**: load it, run the same forward passes using the cached reference token IDs verbatim, compute per-token KL, then unload before loading the next.
4. Aggregate stats, write JSON, optionally render a chart.

Only one model is in memory at a time — important when the reference is large.

KL is computed in the standard direction `KL(P_ref || P_cmp)` — information lost when using the comparison distribution to approximate the reference. With sparse references:

```
KL(P || Q) = sum_topK(P * (log P - log Q))
           + P_tail * (log P_tail - log Q_tail_estimate)
```

## Dependencies

- [mlx](https://github.com/ml-explore/mlx)
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- numpy
- matplotlib (optional, for `--chart`)

---

## Future Enhancements

Ordered by value × feasibility. The first three would meaningfully improve the *credibility* of any claim made with the tool — they let you say "A is significantly better than B" rather than just "A scored lower on this run".

Each of these improvements should be removed from this README.md once implemented in full.

For agents implementing any of the following enhancements, this README + codebase covers most of what and how. The non-obvious context falls into a few buckets:

### Project history that may not be inferred

- The bug fix in compare.py (cache load path silently re-tokenising) is real and recent — they should respect the inline comment explaining why cmp_prepared = prepared. If they "improve" by re-adding tokenisation they'll
break months-old caches. Worth telling them: "the verbatim-token comment is load-bearing, don't relitigate it."
- The sparse-K approximation has a known ~3-5% systematic underestimate on real LLM output (verified Phase D in results/qwen3.6-mlx-results.md). If they see test tolerance loosened to 15% on synthetic Gaussians and
panic-tighten it, the synthetic distributions will fail because real LLM peakedness is much higher than what's testable cheaply. Tell them: "the 15% tolerance is calibrated to synthetic worst-case; real-output error is
~4%."
- The chunked path uses mlx_lm.models.cache.make_prompt_cache and is bit-equivalent to un-chunked. Earlier versions weren't — there's a regression test that pins this. Don't let them "simplify" by dropping the cache.
- The most useful way of surfacing information to the user is in human readable formats such as markdown. JSON is great for data and state but not as useful for human consumption.

#### Operating environment

- The users machine you're developing on is a powerful M5 Max with 128GB of memory, however you should check with the user before doing anything that may use more than 100GB of memory in case you disrupt anything else they may be working on.
- The user uses uv with the Python 3.14.x venv located at `.venv` in the project root.
- The user keeps their MLX models in `~/.lmstudio/models/`

#### External constraints

- Apple Silicon / MLX only. No CUDA fallbacks. Some idiomatic Python performance tricks (e.g. heavy numpy vectorisation across the full-vocab dim) blow up memory because Qwen-class vocab is ~248k. They should stay aware
that a single dense (seq_len, vocab) tensor at 13K tokens is ~13 GB.
- mlx-lm is moving fast and we need to make sure we support the latest models as mlx gets updated to add them. To date we've pinned behaviour by capability, not by version. The make_prompt_cache API has been stable for a while but the model class hierarchy hasn't.

#### Conventions worth surfacing

- All tokenisation flows through one place; cached token IDs are authoritative once a cache exists. If an enhancement involves new prompt-handling, it should live next to _prepare_prompts and respect the same "cache wins"
  rule.
- The model_info.py quant-family detection is heuristic (path-name based). New families (e.g. AWQ, GPTQ-MLX if those land) need a one-line addition and a colour in chart.py's FAMILY_STYLE. Both touch points should be
flagged.
- The chart palette is intentionally Wong/IBM colour-blind safe with distinct shapes. Don't let them switch to viridis or seaborn defaults.
- Tests have a runtime split: pure-numpy tests run anywhere; tests touching mlx.core need Metal access (don't run in headless CI without a real Mac).

#### Decisions that look weird but are deliberate

- --json-summary-only defaults to off (full per-token detail by default) — for backward compat with the upstream we forked from. New flags should follow the same "additive, opt-in" pattern.
- Confidence intervals (Future Enhancement #1) are listed first because every other quality claim depends on them. If an agent picks up the report-generation task (#9) before #1, the markdown report will keep emitting
point estimates for now — that's fine, just don't let them invent fake CIs to fill the column.
- The --render-chart-from and (planned) --render-markdown-from flags exist precisely so re-rendering doesn't require model re-loading. Any new output format should follow the same dual-path pattern: produce-from-results
AND produce-from-saved-JSON.

#### Things not to do

- Don't add backwards-compatibility shims for fields they themselves are introducing.
- Don't generate interpretive prose in auto-emitted output. The "## Notes" stub in the markdown spec is intentional.

### 1. Confidence intervals on summary stats

Currently mean KLD is reported as a point estimate. With ~2k tokens per run a paired bootstrap (a few seconds of compute) would give 95% CIs:

```
Mean KLD: 0.00321 [0.00298, 0.00347]
```

Without CIs you can't tell whether oQ6 (0.00321) is significantly better than RTN-6bit (0.00373) or just numerically lower on this particular run. Cheap to add (~30 lines), high value.

### 2. Pairwise significance tests between quants

For each pair of compared models, run a paired Wilcoxon signed-rank test on per-token KLD differences. Output a small p-value matrix in the summary. Pairs naturally with confidence intervals — answers "is this difference real?".

### 3. Per-prompt-category breakdown

Tag prompts by category (`code`, `prose`, `reasoning`, `math`, `multilingual`, etc.) via a sidecar file or inline `[code]` prefix, and report KLD per category in addition to the aggregate. Reveals **selectively weak quants** — e.g. UD-4bit might be fine on prose but catastrophic on code; the current aggregate hides this.

### 4. HTML report

Self-contained `.html` artifact embedding the chart, summary table, quant architecture details, and the top-K most divergent tokens *with surrounding text context* (so reviewers can read what each model actually got wrong). Easier to share with collaborators than a directory of JSON.

### 5. Generation-mode divergence

Current measurement is **prefill** divergence — at each existing prompt position, how do the two models' next-token distributions disagree? But user-visible quality is about **autoregressive generation** — how fast do outputs diverge as you sample? A second mode that samples N tokens from each model at fixed prefill positions and measures exact-match length and trajectory KLD would capture cumulative quantisation error that prefill KL misses.

### 6. Memory + load-time persistence

`mx.get_peak_memory()` and the load-time wall clock are already measured for terminal logging. Just persist them to the JSON output. Trivial change, useful for tracking actual M5 Max memory budgets across quants where disk size is a poor proxy (KV cache and activations vary).

### 7. Per-layer attribution analysis (research-grade)

For each high-KLD outlier position, hot-swap each layer's weights between reference and comparison to identify which layer's quantisation produced the divergence. Expensive (~N forward passes per outlier) but extremely informative for quant authors — would let oQ/UD/AutoRound designers see exactly which tensor allocations protected what or failed to.

### 8. Direct quant-vs-quant comparison

Currently every comparison routes through a single reference. Sometimes you want to directly compare two quants without going via bf16 — e.g. "how different are oQ4 and RTN-4 from each other, regardless of how either compares to bf16?". Easy to add as a `--reference-is-quant` flag or a separate `--pairwise` mode.

