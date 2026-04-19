# Real Model PTQ

Model: `gpt2`  
Corpus: `wikitext2`  
Device: `cuda`  
Seeds: `[0]`  
Bits: `[4]`  
Methods: `['fp32', 'admm_gptq']`  
ADMM controller: `task_feasibility`  
ADMM rho0 list: `[0.01, 0.1, 1.0, 10.0, 100.0]`  
ADMM iters: `30`  
fp32 ppl: `29.010` on `285696` eval tokens

| method | bits | rho0 | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|---|
| admm_gptq | 4 | 0.01 | 1 | 275.021 ± 0.000 | +246.011 | 0.1465 |
| admm_gptq | 4 | 0.1 | 1 | 910.785 ± 0.000 | +881.775 | 0.1794 |
| admm_gptq | 4 | 1.0 | 1 | 7372.980 ± 0.000 | +7343.971 | 0.1761 |
| admm_gptq | 4 | 10.0 | 1 | 1300.589 ± 0.000 | +1271.579 | 0.1761 |
| admm_gptq | 4 | 100.0 | 1 | 1175.925 ± 0.000 | +1146.915 | 0.1751 |
| fp32 |  |  | 1 | 29.010 ± 0.000 | +0.000 | 0.0000 |
