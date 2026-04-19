# Cluster / GPU Real-Model Smoke Test

The local CPU smoke test passed with an offline random tiny GPT-2 model. The Hugging Face download for `sshleifer/tiny-gpt2` failed locally because the remote connection was closed, not because the code requires a GPU.

## Files To Transfer

Transfer the whole experiment folder, or at minimum:

- `run_real_model_smoke.py`
- `run_llm_curvature_admm.py`
- `run_llm_quantization_baselines.py`
- `online_admm_experiments/`

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch transformers datasets tqdm numpy pandas
```

For a CUDA cluster, install the PyTorch wheel recommended by the cluster docs.

## CPU/Offline Debug

```bash
python run_real_model_smoke.py \
  --random-tiny \
  --max-batches 3 \
  --max-modules 6 \
  --max-samples 96 \
  --max-admm-iter 6 \
  --block-size 64
```

Expected behavior: it should run `fp32`, `rtn`, and `admm_gptq`, write `results/real_model_smoke/real_model_smoke_summary.csv`, and report that several modules were quantized.

## Real Small Model Smoke

Start with:

```bash
python run_real_model_smoke.py \
  --model sshleifer/tiny-gpt2 \
  --max-batches 4 \
  --max-modules 8 \
  --max-samples 128 \
  --max-admm-iter 8 \
  --block-size 64
```

Then try:

```bash
python run_real_model_smoke.py \
  --model gpt2 \
  --max-batches 8 \
  --max-modules 16 \
  --max-samples 256 \
  --max-admm-iter 10 \
  --block-size 128
```

For OPT/Pythia models, use a GPU if available:

```bash
python run_real_model_smoke.py \
  --model facebook/opt-125m \
  --max-batches 8 \
  --max-modules 24 \
  --max-samples 256 \
  --max-admm-iter 10 \
  --block-size 128
```

## What Counts As Success

- The script loads the model and collects calibration inputs.
- `rtn` and `admm_gptq` quantize a nonzero number of modules.
- The quantized losses are finite.
- `admm_gptq` has comparable or better loss delta than `rtn`.

The current script is still a smoke test, not a final benchmark. For the paper-grade run, replace the built-in toy texts with WikiText-2/C4 calibration and evaluate full perplexity.
