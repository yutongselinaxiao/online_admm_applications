# Real Model PTQ

Model: `random_tiny_gpt2`  
Corpus: `toy`  
Device: `cuda`  
Seeds: `[0, 1]`  
Bits: `[4]`  
Methods: `['fp32', 'rtn', 'gptq_proxy', 'awq_proxy', 'admm_gptq']`  
ADMM iters: `6`  
fp32 ppl: `128.807` on `192` eval tokens

| method | bits | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|
| admm_gptq | 4 | 2 | 128.821 ± 0.008 | +0.014 | 0.0983 |
| awq_proxy | 4 | 2 | 128.836 ± 0.035 | +0.029 | 0.0965 |
| fp32 |  | 2 | 128.807 ± 0.000 | +0.000 | 0.0000 |
| gptq_proxy | 4 | 2 | 128.813 ± 0.032 | +0.006 | 0.1218 |
| rtn | 4 | 2 | 128.857 ± 0.000 | +0.051 | 0.0994 |
