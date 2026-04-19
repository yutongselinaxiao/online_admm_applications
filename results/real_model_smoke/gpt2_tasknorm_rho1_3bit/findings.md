# Real Model PTQ

Model: `gpt2`  
Corpus: `wikitext2`  
Device: `cuda`  
Seeds: `[0, 1, 2, 3, 4]`  
Bits: `[3]`  
Methods: `['fp32', 'admm_gptq']`  
ADMM controller: `task_norm_magnitude`  
ADMM rho0 list: `[1.0]`  
ADMM iters: `30`  
fp32 ppl: `29.010` on `285696` eval tokens

| method | bits | rho0 | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|---|
| admm_gptq | 3 | 1.0 | 5 | 344012550.059 ± 177441649.679 | +344012521.049 | 0.3505 |
| fp32 |  |  | 5 | 29.010 ± 0.000 | +0.000 | 0.0000 |
