# Real Model PTQ

Model: `gpt2`  
Corpus: `wikitext2`  
Device: `cuda`  
Seeds: `[0, 1, 2, 3, 4]`  
Bits: `[4]`  
Methods: `['fp32', 'admm_gptq']`  
ADMM controller: `task_norm_magnitude`  
ADMM rho0 list: `[1.0]`  
ADMM iters: `60`  
fp32 ppl: `29.010` on `285696` eval tokens

| method | bits | rho0 | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|---|
| admm_gptq | 4 | 1.0 | 5 | 10154.657 ± 1698.004 | +10125.648 | 0.1752 |
| fp32 |  |  | 5 | 29.010 ± 0.000 | +0.000 | 0.0000 |
