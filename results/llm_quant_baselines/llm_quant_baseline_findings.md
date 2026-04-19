# LLM Quantization Baseline Findings

These are local proxy implementations on the synthetic LLM PTQ calibration problem, not official GPTQ/AWQ/SmoothQuant library runs.

## Mean Deploy Relative Error

- GPTQ-like sequential Hessian compensation: `0.205`
- Hessian diagonal clipping: `0.222`
- RTN per-channel: `0.252`
- ADMM aggressive task-feasibility: `0.254`
- ADMM fixed rho=10: `0.278`

## Conclusion

The current ADMM quantizer does not beat Hessian-aware PTQ. It is roughly competitive with RTN only after aggressive feasibility tuning, but GPTQ-like error compensation is clearly stronger on this calibration objective.

This does not kill the online-rho idea. It says the current projection step is too weak. The strongest next version should combine online ADMM penalty tuning with a Hessian-aware or activation-aware quantized `Z` update, rather than comparing pure uniform projection ADMM against GPTQ/AWQ directly.
