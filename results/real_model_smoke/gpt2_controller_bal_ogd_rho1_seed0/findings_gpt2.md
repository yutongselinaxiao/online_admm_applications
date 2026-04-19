# Real Model PTQ

Model: `gpt2`  
Corpus: `wikitext2`  
Device: `cuda`  
Seeds: `[0]`  
Bits: `[4]`  
Methods: `['fp32', 'admm_gptq']`  
ADMM controller: `bal_ogd`  
ADMM rho0 list: `[1.0]`  
ADMM iters: `30`  
fp32 ppl: `29.010` on `285696` eval tokens

| method | bits | rho0 | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|---|
| admm_gptq | 4 | 1.0 | 1 | 3326.326 ± 0.000 | +3297.316 | 0.1768 |
| fp32 |  |  | 1 | 29.010 ± 0.000 | +0.000 | 0.0000 |
