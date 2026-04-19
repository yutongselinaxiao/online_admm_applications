# Real-Model PTQ Runs

This doc covers paper-grade PTQ runs from `run_real_model_smoke.py` on GPU
clusters. The script is a single entry point for both debug-scale smoke tests
(`--random-tiny --corpus toy`) and paper-grade benchmarks (`--corpus wikitext2`
with multiple seeds / bits / methods).

## Environment

The script needs `torch`, `transformers`, `datasets`, `numpy`. The existing
environment at `/data/yutong/envs/jaxgpu` works:

```bash
/data/yutong/envs/jaxgpu/bin/pip install transformers datasets
```

On a fresh cluster:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch transformers datasets numpy
```

For H100 nodes, install the PyTorch wheel matching the cluster's CUDA (CUDA
12.x for H100). Our A40 validation used torch 2.5.1 + CUDA 12.1.

## Architecture notes

- **ADMM hot path runs on GPU** (torch float32). The per-module bottleneck —
  `torch.linalg.solve`, `torch.linalg.inv`, GPTQ sequential update — executes
  on the target device. The penalty controller (`FeasibilityTaskOnlineOGD` by
  default, selectable via `--admm-controller`) stays pure Python.
- **Calibration is sample-free**: we accumulate the Gram matrix
  `xtx = X^T X / N` and per-channel `|x|` mean via forward hooks, never
  materializing raw activations. Memory footprint is `d^2` floats per module
  (max ~40 MB for the largest `opt-125m` Linear).
- **Perplexity eval** is standard sliding-window: non-overlapping
  `block_size`-token windows over the entire WikiText-2 test corpus.
- **Module coverage**: `--max-modules` defaults to unlimited (all supported
  `nn.Linear` / `Conv1D`).
- **ADMM canonical form** matches `../online_admm/online_admm.py::admm_one_step`
  in scaled-dual form; the z-update uses the curvature-aware GPTQ variant from
  ledger §7. Residual definitions, ρ-rescale on change, and stopping
  thresholds all follow the JAX reference.

## Debug-scale smoke (offline, ~1 minute on any GPU/CPU)

```bash
/data/yutong/envs/jaxgpu/bin/python run_real_model_smoke.py \
  --random-tiny --corpus toy \
  --seeds 0 1 --bits-list 4 \
  --max-admm-iter 6 --calib-seqs 3 \
  --eval-max-batches 3 --block-size 64 \
  --output-subdir validate_random_tiny
```

Success criteria: all 5 methods run, `aggregate.csv` and `findings.md` written.

## Paper-grade run (H100 80GB)

### Single model, all methods, 5 seeds × {3,4}-bit

`gpt2` takes ~30–60 min per (seed, bits, method) combo for `admm_gptq`;
`opt-125m` is slower because several Linears are 3072-dim (6.5M GPTQ loop
iterations per module-iter). Budget ~6–12 hours per model on H100.

```bash
HF_HOME=/data/yutong/hf_cache \
/data/yutong/envs/jaxgpu/bin/python run_real_model_smoke.py \
  --model gpt2 \
  --corpus wikitext2 \
  --seeds 0 1 2 3 4 \
  --bits-list 3 4 \
  --max-admm-iter 30 \
  --calib-seqs 128 \
  --block-size 2048 \
  --cache-dir /data/yutong/hf_cache \
  --output-subdir gpt2_paper_grade
```

```bash
HF_HOME=/data/yutong/hf_cache \
/data/yutong/envs/jaxgpu/bin/python run_real_model_smoke.py \
  --model facebook/opt-125m \
  --corpus wikitext2 \
  --seeds 0 1 2 3 4 \
  --bits-list 3 4 \
  --max-admm-iter 30 \
  --calib-seqs 128 \
  --block-size 2048 \
  --cache-dir /data/yutong/hf_cache \
  --output-subdir opt125m_paper_grade
```

### Controller ablation (tests ADMM variants against fixed baselines)

Run the same config with a different `--admm-controller`:

```bash
for ctrl in task_feasibility bal_ogd heuristic spectral task_norm_magnitude fixed; do
  /data/yutong/envs/jaxgpu/bin/python run_real_model_smoke.py \
    --model gpt2 --corpus wikitext2 \
    --seeds 0 1 2 --bits-list 4 \
    --max-admm-iter 30 --calib-seqs 128 --block-size 2048 \
    --methods fp32 admm_gptq \
    --admm-controller $ctrl \
    --output-subdir gpt2_controller_${ctrl}
done
```

### Quick sanity before a long run

```bash
/data/yutong/envs/jaxgpu/bin/python run_real_model_smoke.py \
  --model gpt2 --corpus wikitext2 \
  --seeds 0 --bits-list 4 --max-admm-iter 10 \
  --calib-seqs 16 --eval-max-batches 20 --block-size 1024 \
  --output-subdir gpt2_sanity
```

## What counts as success

- `summary.csv` and `aggregate.csv` written for each invocation.
- `fp32` ppl lands at the published baseline (~29 for gpt2, ~27 for opt-125m
  on WikiText-2; our eval uses the full test corpus).
- `rtn` at 4-bit all-layers degrades noticeably (expected without
  calibration).
- `gptq_proxy`, `awq_proxy`, `admm_gptq` all recover most of the gap to fp32.
- `admm_gptq` at {3,4}-bit is within noise of `gptq_proxy` — or beats it when
  the online ρ gives the feasibility/curvature tradeoff some headroom.

## Output layout

Each run creates `results/real_model_smoke/<output-subdir>/`:

- `summary.csv` — per (seed, bits, method) row with ppl, Δppl, recon error, ρ.
- `aggregate.csv` — means and std across seeds.
- `per_module.csv` — per-module recon error (useful for diagnosing outlier
  layers).
- `findings.md` — markdown table for quick inspection.

## Legacy compatibility

The old `--random-tiny --max-batches 3 --max-modules 6 --max-samples 96 \
--max-admm-iter 6 --block-size 64` invocation still works; flags `--bits`,
`--seed`, `--max-batches`, `--max-samples` map onto the new `--bits-list`,
`--seeds`, `--calib-seqs` API. `--max-samples` is a no-op now (gram matrices
are sample-free).
