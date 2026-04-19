# Real Model PTQ

Model: `gpt2`  
Corpus: `wikitext2`  
Device: `cuda`  
Seeds: `[1, 2, 3, 4]`  
Bits: `[4]`  
Methods: `['fp32', 'admm_gptq']`  
ADMM controller: `task_feasibility`  
ADMM rho0 list: `[0.01]`  
ADMM iters: `30`  
fp32 ppl: `29.010` on `285696` eval tokens

| method | bits | rho0 | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|---|
| admm_gptq | 4 | 0.01 | 4 | 1149.313 ± 874.476 | +1120.303 | 0.1459 |
| fp32 |  |  | 4 | 29.010 ± 0.000 | +0.000 | 0.0000 |
