# Real Model Smoke Test

Model: `random_tiny_gpt2`

Device: `cpu`

This is a CPU/debug-scale smoke test on a tiny text corpus, not a final perplexity benchmark.

- `fp32`: loss `4.8407`, ppl `126.56`, delta loss `0.0000`, modules `0`
- `rtn`: loss `4.8407`, ppl `126.56`, delta loss `0.0000`, modules `6`
- `admm_gptq`: loss `4.8407`, ppl `126.56`, delta loss `-0.0000`, modules `6`
