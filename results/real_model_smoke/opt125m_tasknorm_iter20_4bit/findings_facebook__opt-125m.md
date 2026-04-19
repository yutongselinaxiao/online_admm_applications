# Real Model PTQ

Model: `facebook/opt-125m`  
Corpus: `wikitext2`  
Device: `cuda`  
Seeds: `[0, 1, 2]`  
Bits: `[4]`  
Methods: `['fp32', 'gptq_proxy', 'awq_proxy', 'admm_gptq']`  
ADMM controller: `task_norm_magnitude`  
ADMM rho0 list: `[1.0]`  
ADMM iters: `20`  
fp32 ppl: `27.366` on `284672` eval tokens

| method | bits | rho0 | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|---|
| admm_gptq | 4 | 1.0 | 3 | 85.519 ± 4.258 | +58.153 | 0.1467 |
| awq_proxy | 4 |  | 3 | 49.825 ± 0.045 | +22.459 | 0.1337 |
| fp32 |  |  | 3 | 27.366 ± 0.000 | +0.000 | 0.0000 |
| gptq_proxy | 4 |  | 3 | 61.221 ± 2.595 | +33.855 | 0.1306 |
