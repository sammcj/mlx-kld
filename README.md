# mlx-kld

Measure KL divergence between MLX language model output distributions. Compare a reference model (typically full-precision) against a quantized variant to see how much the quantization shifts the model's probability distributions.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
mlx-kld \
  --reference /path/to/full-precision-model \
  --compare /path/to/quantized-model \
  --prompts "What is the capital of France?" "Explain photosynthesis." \
  --top-k 10
```

### Options

| Flag | Description |
|------|-------------|
| `--reference` | Path or HF repo for the reference model (required) |
| `--compare` | Path or HF repo for the comparison model (required) |
| `--prompts` | One or more prompt strings |
| `--prompts-file` | Text file with one prompt per line |
| `--output` | Save detailed per-token results to a JSON file |
| `--top-k N` | Show the N most divergent tokens in the output |
| `--no-chat-template` | Tokenize raw prompts instead of applying the model's chat template |

### Examples

Compare a full-precision model against its 8-bit quant:

```bash
mlx-kld \
  --reference /Volumes/Blackbird/Models/Qwen/Qwen3.5-0.8B-FULL \
  --compare /Volumes/Blackbird/Models/thecraig/Qwen3.5-0.8B-oQ8 \
  --prompts "What is gravity?" \
  --top-k 5 \
  --output results.json
```

Use a file of prompts for batch evaluation:

```bash
mlx-kld \
  --reference mlx-community/Llama-3.2-3B-Instruct \
  --compare mlx-community/Llama-3.2-3B-Instruct-4bit \
  --prompts-file prompts.txt \
  --output results.json
```

## How it works

1. Loads the **reference model**, runs a forward pass on each prompt, and stores the log-softmax output distributions as numpy arrays
2. **Unloads the reference model** to free memory
3. Loads the **comparison model**, runs the same forward passes, and computes per-token KL divergence against the stored reference distributions
4. Reports summary statistics and optionally saves detailed per-token results

Models are loaded sequentially so only one is in memory at a time. This matters for large full-precision models.

### What's measured

KL divergence at each token position in the prefill (forward pass on the prompt tokens):

```
KL(P_ref || P_cmp) = sum(P_ref * (log(P_ref) - log(P_cmp)))
```

This tells you how much information is lost at each token position by using the comparison model instead of the reference.

### Output

```
==================================================
  KL Divergence Results
==================================================
  Reference: Qwen/Qwen3.5-0.8B-FULL
  Compare:   thecraig/Qwen3.5-0.8B-oQ8
==================================================
  Prompts:     2
  Tokens:      37
--------------------------------------------------
  Mean KLD:    0.001821
  Median KLD:  0.000715
  Std KLD:     0.003890
  P95 KLD:     0.006803
  P99 KLD:     0.017617
  Max KLD:     0.017784
             (prompt 1, position 13, token "assistant")
==================================================
```

The `--output` JSON includes per-token KLD values, token IDs, and decoded token strings for every prompt.

## Dependencies

- [mlx](https://github.com/ml-explore/mlx)
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- numpy
