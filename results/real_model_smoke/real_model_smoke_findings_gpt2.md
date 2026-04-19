# Real Model Smoke Test

Model: `gpt2`

Device: `cuda`

Hugging Face cache: `/dataMeR2/yutong/hf_cache`

This is a CPU/debug-scale smoke test on a tiny text corpus, not a final perplexity benchmark.

- `fp32`: loss `4.7312`, ppl `113.43`, delta loss `0.0000`, modules `0`
- `rtn`: loss `4.7681`, ppl `117.69`, delta loss `0.0369`, modules `16`
- `admm_gptq`: loss `4.7514`, ppl `115.75`, delta loss `0.0202`, modules `16`
