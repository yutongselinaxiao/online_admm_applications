# Real Model PTQ

Model: `random_tiny_gpt2`  
Corpus: `toy`  
Device: `cpu`  
Seeds: `[0]`  
Bits: `[4]`  
Methods: `['fp32', 'admm_gptq']`  
ADMM controller: `task_feasibility`  
ADMM rho0 list: `[0.1, 1.0]`  
ADMM iters: `2`  
fp32 ppl: `128.839` on `64` eval tokens

| method | bits | rho0 | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|---|
| admm_gptq | 4 | 0.1 | 1 | 128.960 ± 0.000 | +0.121 | 0.0883 |
| admm_gptq | 4 | 1.0 | 1 | 128.966 ± 0.000 | +0.127 | 0.0960 |
| fp32 |  |  | 1 | 128.839 ± 0.000 | +0.000 | 0.0000 |
