# Real Model PTQ

Model: `gpt2`  
Corpus: `wikitext2`  
Device: `cuda`  
Seeds: `[0]`  
Bits: `[4]`  
Methods: `['fp32', 'rtn', 'gptq_proxy', 'awq_proxy', 'admm_gptq']`  
ADMM iters: `6`  
fp32 ppl: `36.859` on `2048` eval tokens

| method | bits | seeds | ppl_mean ± std | Δppl | mean recon |
|---|---|---|---|---|---|
| admm_gptq | 4 | 1 | 332.302 ± 0.000 | +295.443 | 0.1721 |
| awq_proxy | 4 | 1 | 209.349 ± 0.000 | +172.490 | 0.1586 |
| fp32 |  | 1 | 36.859 ± 0.000 | +0.000 | 0.0000 |
| gptq_proxy | 4 | 1 | 346.734 ± 0.000 | +309.874 | 0.1284 |
| rtn | 4 | 1 | 149276.045 ± 0.000 | +149239.186 | 0.1752 |
