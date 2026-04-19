# Real Model Smoke Test

Model: `sshleifer/tiny-gpt2`

Device: `cuda`

Hugging Face cache: `/dataMeR2/yutong/hf_cache`

This is a CPU/debug-scale smoke test on a tiny text corpus, not a final perplexity benchmark.

- `fp32`: loss `10.8237`, ppl `50197.51`, delta loss `0.0000`, modules `0`
- `rtn`: loss `10.8237`, ppl `50197.97`, delta loss `0.0000`, modules `8`
- `admm_gptq`: loss `10.8237`, ppl `50197.89`, delta loss `0.0000`, modules `8`
