# Curvature-Aware ADMM Findings

Run configuration: synthetic LLM PTQ, three seeds, 100 ADMM iterations, 4-bit quantization.

## Main Results

- Best non-FP16 method: `gptq_like_sequential` at deploy relative error `0.205`.
- Standalone GPTQ-like proxy: `0.205`.
- ADMM with GPTQ-like `Z` update plus online rho: `0.214`.
- ADMM with Hessian-diagonal `Z` update plus online rho: `0.237`.
- ADMM with the original uniform `Z` update plus online rho: `0.254`.

## Conclusion

This is the strongest evidence so far for the online ADMM direction. Plain ADMM with uniform projection only matches RTN. Once the `Z` step becomes curvature-aware, online ADMM becomes competitive with Hessian-aware PTQ and nearly reaches the standalone GPTQ-like proxy. The remaining gap is small enough to justify a real-model experiment.
