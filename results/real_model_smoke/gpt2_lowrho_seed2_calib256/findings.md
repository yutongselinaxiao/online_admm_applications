# Real Model PTQ

Model: `gpt2`  
Corpus: `wikitext2`  
Device: `cuda`  
Seeds: `[2]`  
Bits: `[4]`  
Methods: `['fp32', 'admm_gptq']`  
ADMM controller: `task_feasibility`  
ADMM rho0 list: `[0.01]`  
ADMM iters: `30`  
fp32 ppl: `29.010` on `285696` eval tokens

| method | bits | rho0 | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|---|
| admm_gptq | 4 | 0.01 | 1 | 361.599 ± 0.000 | +332.590 | 0.1461 |
| fp32 |  |  | 1 | 29.010 ± 0.000 | +0.000 | 0.0000 |
