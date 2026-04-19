# Real Model PTQ

Model: `gpt2`  
Corpus: `wikitext2`  
Device: `cuda`  
Seeds: `[0]`  
Bits: `[4]`  
Methods: `['fp32', 'admm_gptq']`  
ADMM controller: `task_norm_magnitude`  
ADMM rho0 list: `[1.0]`  
ADMM iters: `30`  
fp32 ppl: `29.010` on `285696` eval tokens

| method | bits | rho0 | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|---|
| admm_gptq | 4 | 1.0 | 1 | 583.009 ± 0.000 | +554.000 | 0.1613 |
| fp32 |  |  | 1 | 29.010 ± 0.000 | +0.000 | 0.0000 |
