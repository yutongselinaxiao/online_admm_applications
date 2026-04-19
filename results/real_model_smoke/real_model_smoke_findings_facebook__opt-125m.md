# Real Model Smoke Test

Model: `facebook/opt-125m`

Device: `cuda`

Hugging Face cache: `/dataMeR2/yutong/hf_cache`

This is a CPU/debug-scale smoke test on a tiny text corpus, not a final perplexity benchmark.

- `fp32`: loss `5.0232`, ppl `151.89`, delta loss `0.0000`, modules `0`
- `rtn`: loss `5.0618`, ppl `157.87`, delta loss `0.0386`, modules `24`
- `admm_gptq`: loss `5.0793`, ppl `160.66`, delta loss `0.0561`, modules `24`
