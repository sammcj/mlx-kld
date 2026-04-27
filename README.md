# mlx-kld

Measure KL divergence between MLX language model output distributions. Compare a reference model (typically full-precision or 8-bit) against one or more quantised variants to rank quant quality, with optional speed and architecture context.

## Install

```bash
uv venv .venv
source .venv/bin/activate
pip install -e . ".[chart]" ".[dev]"
```

## Quick start

The default is **short mode** — a fast smoke test against the bundled discrete prompt set. Add `--long` for a methodologically-comparable streamed-corpus evaluation against WikiText-2 (the corpus everyone else uses).

```bash
# Default = short mode. Bundled sample_prompts/prompts.txt loads automatically.
mlx-kld \
  --reference /path/to/models/mlx-community/Qwen3.6-27B-8bit \
  --compare \
    /path/to/models/mlx-community/Qwen3.6-27B-4bit \
    /path/to/models/mlx-community/Qwen3.6-27B-6bit \
    /path/to/models/Jundot/Qwen3.6-27B-oQ6 \
  --top-k-cache 256 \
  --output results/run

# Add --long to also run the WikiText-2 streamed evaluation. Both summaries
# land in the same JSON; the markdown report shows both tables.
mlx-kld --short --long \
  --reference /path/to/models/mlx-community/Qwen3.6-27B-8bit \
  --compare /path/to/models/mlx-community/Qwen3.6-27B-4bit \
  --top-k-cache 256 \
  --output results/run

# Modes preserve each other across runs. Run --short today, --long next week,
# and the final JSON for each model contains BOTH sets of numbers.
```

Build a reference cache once and reuse it for many comparison runs:

```bash
# Stage 1: cache the reference (8-bit is a fine stand-in for bf16; KL between
# them is ~1e-4, ~1000x smaller than typical 4-bit divergence).
mlx-kld \
  --reference /path/to/models/mlx-community/Qwen3.6-27B-8bit \
  --compare  /path/to/models/mlx-community/Qwen3.6-27B-8bit \
  --top-k-cache 256 \
  --save-reference qwen27b_ref

# Stage 2: rank as many quants as you like against the cached reference.
mlx-kld \
  --load-reference qwen27b_ref \
  --compare \
    /path/to/models/mlx-community/Qwen3.6-27B-4bit \
    /path/to/models/mlx-community/Qwen3.6-27B-6bit \
    /path/to/models/Jundot/Qwen3.6-27B-oQ4 \
    /path/to/models/Jundot/Qwen3.6-27B-oQ6 \
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
| `--short`                | Run the short-prompt mode (default if no prompt source is given).                                    |
| `--long`                 | Run the long-mode WikiText-2 streamed evaluation. May be combined with `--short`.                    |
| `--long-corpus PATH`     | Override the long-mode corpus. Defaults to bundled `corpora/wiki.test.raw`.                          |
| `--long-ctx N`           | Long-mode chunk size in tokens. Default 2048 (GPTQ/AWQ academic convention).                         |
| `--long-chunks N`        | Number of long-mode chunks to take from the start of the corpus. Default 32.                         |
| `--prompts`              | One or more prompt strings (literal; takes precedence over `--short`).                               |
| `--prompts-file`         | Text file with one prompt per line (literal; takes precedence over `--short`).                       |
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

## What KLD measures

KL divergence is a number that says how much the quantised model's next-token predictions disagree with the reference model's, averaged across every position in your prompts. Think of it as "information lost when you use the quant in place of the reference" — `0` means the two models produce identical predictions; bigger means more divergent.

At each token position, both models emit a probability distribution over the full vocabulary (~248k tokens for Qwen-class models). The reference says "the next token is ~80% likely to be `the`, ~5% likely to be `a`, …"; the quant says something slightly different. KLD compares those two distributions per position, then averages.

This is a sharper signal for quant ranking than perplexity or downstream eval scores. Perplexity measures absolute model quality on a corpus, which mixes the model's intrinsic uncertainty with quantisation damage. KLD against a fixed reference isolates the quant damage — you're asking "how much did this quant change the model's behaviour?" rather than "how good is this model overall?". It's also far cheaper than running an eval suite: a few thousand tokens through one prefill pass is enough to rank quants reliably.

Rough scale (in nats, on Qwen-class instruct models):

| Mean KLD | Reads as |
| --- | --- |
| `< 1e-4` | Effectively identical to the reference (e.g. 8-bit vs bf16) |
| `1e-3` to `5e-3` | Very close — typical of well-made 6-bit quants |
| `1e-2` to `5e-2` | Noticeable degradation — typical 4-bit territory |
| `> 1e-1` | Substantial divergence; outputs likely to differ obviously |

Two caveats worth knowing:

- **Prefill only.** This tool measures how much the next-token *distribution* diverges at each existing prompt position. It doesn't measure how much sampled outputs drift over a long generation — small per-token divergences can compound. Lower KLD reliably tracks "subjectively better quant", but isn't a guarantee.
- **Direction matters.** We compute `KL(reference ‖ quant)`, which weights divergence by the *reference's* probabilities — disagreements where the reference is confident count for more than disagreements in the long tail. That's the right direction for "how well does the quant imitate the reference", and it's the same quantity you'd minimise if you were training the quant to copy the reference. The reverse direction (`KL(quant ‖ reference)`) would answer a different question.

The choice of KLD over perplexity for quantisation evaluation follows the argument made by Dutta et al. ([NeurIPS 2024](https://arxiv.org/abs/2407.09141)): a quantised model is meant to be a drop-in replacement for the reference, so the right metric is *distance to the reference*, not absolute capability. Their result is striking — on MCQ tasks, perplexity changes can stay near zero while up to 13.6% of answers flip versus the FP16 reference; KLD against the reference correlates 0.96-0.98 with that flip rate.

## Evaluation modes

Two regimes ship in the box. They measure subtly different things and can be run together.

### Short mode (default, `--short`)

A list of 51 discrete prompts (49 short + 2 synthetic ~4k-token long ones) covering reasoning, code, prose, multilingual, and code-review tasks. The whole set tokenises to roughly 10k tokens, runs in ~2 minutes for 5 models on M5 Max, and is the fast smoke test you reach for during iteration. Per-prompt KL is aggregated from position 8 onwards — the first eight token positions have very little left-context so their per-token KL is dominated by the unconditioned distribution prior over sentence-openers, noise that has nothing to do with quantisation. Skipping them removes that bias at the cost of ~16% of scored tokens.

This mode is *idiosyncratic* relative to the wider quantisation-eval ecosystem. It's task-flavoured rather than corpus-streamed, which is good for noticing breakage but means its absolute numbers aren't directly comparable to llama.cpp / GPTQ / AWQ scoreboards. Treat it as your fast iteration loop.

### Long mode (`--long`)

A streamed evaluation on the WikiText-2 raw test set, mirroring the methodology used by [llama.cpp's `llama-perplexity`](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/evaluation/kl_divergence.py), [GPTQ](https://github.com/IST-DASLab/gptq), and the academic literature ([Frantar et al. 2022](https://arxiv.org/abs/2210.17323), [Lin et al. 2023](https://arxiv.org/abs/2306.00978)). The corpus is tokenised end-to-end with the reference tokeniser, split into back-to-back non-overlapping windows of `--long-ctx` tokens (default 2048), and the first `--long-chunks` chunks (default 32) are scored. Within each chunk only positions ≥ n_ctx/2 contribute to the summary — the llama.cpp convention — so every scored token has at least n_ctx/2 tokens of left-context.

This mode's numbers are methodologically comparable to published tables. The corpus ships in the repo at `corpora/wiki.test.raw` (~1.3 MB) under its original CC BY-SA licence; see `corpora/LICENSE.wikitext` for attribution.

Long-context evaluation matters: [Mekala et al. 2025](https://arxiv.org/abs/2505.20276) showed 4-bit quants can lose up to 59% on 128K retrieval tasks while looking fine on short context. Short-only evaluation systematically underestimates the harm of aggressive quantisation, so when you're picking a quant for production you want both numbers.

### Combining

`--short --long` runs both regimes in one invocation and reports stratified summaries per mode (short and long means aren't directly comparable — they sample different regimes — but rank-order should agree if a quant has no long-context-specific failure). Modes also persist across separate invocations: running `--short` today and `--long` next week leaves both sets of numbers in the per-model JSON; the markdown report and chart pick up whatever is present.

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

KL is computed in the standard direction `KL(P_ref || P_cmp)` — information lost when using the comparison distribution to approximate the reference. Reported in **nats** (natural log via numpy). With sparse references:

```
KL(P || Q) = sum_topK(P * (log P - log Q))
           + P_tail * (log P_tail - log Q_tail_estimate)
```

## Dependencies

- [mlx](https://github.com/ml-explore/mlx)
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- numpy
- matplotlib (optional, for `--chart`)

## References

The methodology here borrows from a few specific places. If you're trying to reproduce numbers or compare against published tables, these are the citations that matter.

- **WikiText-2 corpus** — Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher. [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). ICLR 2017. The bundled `corpora/wiki.test.raw` is the test split of WikiText-2 raw v1, redistributed under CC BY-SA per `corpora/LICENSE.wikitext`. Same archive used by llama.cpp.
- **KL divergence over perplexity for quant evaluation** — Dutta, Krishnan, Kwatra, Ramjee. [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141). NeurIPS 2024. The strongest argument in the literature for treating a quantised model as a *drop-in replacement* and measuring distance to the reference, not absolute capability. PPL can stay flat while 5-13% of MCQ answers flip; KLD against the reference correlates 0.96-0.98 with that flip rate.
- **Long-context quantisation degradation** — Mekala et al. [Does quantization affect models' performance on long-context tasks?](https://arxiv.org/abs/2505.20276) 2025. First systematic study at 8K / 64K / 128K. 4-bit quants can drop up to 59% on 128K retrieval (OneRuler) while looking fine on short context. Motivates the long-mode default.
- **n_ctx=2048 streamed-corpus convention** — Frantar et al. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323). 2022. The convention later inherited by AWQ, SpQR, QuIP, OmniQuant, AutoAWQ, AutoGPTQ, etc.
- **Score-only-the-second-half trick** — [llama.cpp's perplexity tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp), specifically the `kl_divergence()` function. Removes the early-position noise that dominates short-context KL by ignoring positions `[0, n_ctx/2)` of every chunk. The same idea applied to short prompts is the `SKIP_FIRST_TOKENS_SHORT = 8` constant in `mlx_kld/metrics.py`.
- **Activation-aware weight quantisation context** — Lin et al. [AWQ](https://arxiv.org/abs/2306.00978). 2023. For the calibration-vs-evaluation distribution-mismatch experiments that motivate using a separate evaluation corpus rather than re-using calibration data.

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
- The user keeps their MLX models in `/path/to/models/`

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
- The `--render-chart-from` and `--render-markdown-from` flags exist precisely so re-rendering doesn't require model re-loading. Any new output format should follow the same dual-path pattern: produce-from-results AND produce-from-saved-JSON.

#### Things not to do

- Don't add backwards-compatibility shims for fields they themselves are introducing.
- Don't generate interpretive prose in auto-emitted output. The "## Notes" stub in the markdown spec is intentional.

### Enhancements

#### Confidence intervals on summary stats

Currently mean KLD is reported as a point estimate. With ~2k tokens per run a paired bootstrap (a few seconds of compute) would give 95% CIs:

```
Mean KLD: 0.00321 [0.00298, 0.00347]
```

Without CIs you can't tell whether oQ6 (0.00321) is significantly better than RTN-6bit (0.00373) or just numerically lower on this particular run. Cheap to add (~30 lines), high value.

#### Pairwise significance tests between quants

For each pair of compared models, run a paired Wilcoxon signed-rank test on per-token KLD differences. Output a small p-value matrix in the summary. Pairs naturally with confidence intervals — answers "is this difference real?".

#### Per-prompt-category breakdown

Tag prompts by category (`code`, `prose`, `reasoning`, `math`, `multilingual`, etc.) via a sidecar file or inline `[code]` prefix, and report KLD per category in addition to the aggregate. Reveals **selectively weak quants** — e.g. UD-4bit might be fine on prose but catastrophic on code; the current aggregate hides this.

#### Generation-mode divergence

Current measurement is **prefill** divergence — at each existing prompt position, how do the two models' next-token distributions disagree? But user-visible quality is about **autoregressive generation** — how fast do outputs diverge as you sample? A second mode that samples N tokens from each model at fixed prefill positions and measures exact-match length and trajectory KLD would capture cumulative quantisation error that prefill KL misses.

#### Memory + load-time persistence

`mx.get_peak_memory()` and the load-time wall clock are already measured for terminal logging. Just persist them to the JSON output. Trivial change, useful for tracking actual M5 Max memory budgets across quants where disk size is a poor proxy (KV cache and activations vary).

#### Direct quant-vs-quant comparison

Currently every comparison routes through a single reference. Sometimes you want to directly compare two quants without going via bf16 — e.g. "how different are oQ4 and RTN-4 from each other, regardless of how either compares to bf16?". Easy to add as a `--reference-is-quant` flag or a separate `--pairwise` mode.

### Per-layer attribution analysis (research-grade)

For each high-KLD outlier position, hot-swap each layer's weights between reference and comparison to identify which layer's quantisation produced the divergence. Expensive (~N forward passes per outlier) but extremely informative for quant authors — would let oQ/UD/AutoRound designers see exactly which tensor allocations protected what or failed to.

#### HTML report

Self-contained `.html` artifact with embedded chart, summary table, quant architecture details, and the top-K most divergent tokens *with surrounding text context* (so reviewers can read what each model actually got wrong). Easier to share with collaborators than a directory of JSON.
