# Rheidae ML Pipeline

Pipeline to preprocess neutrino flux HDF5 data, build training sets, and train the residual FFI model (48-dim input → 24-dim output).

## Layout
- `pre_process/` Julia preprocessing (Box3D, etc.).
- `config/` YAML configs: `pre_process.yaml` (Julia), `model.yaml` (experiment), `train_parms.yaml` (training controls).
- `src/rheidae/` Python package (model, data loader, losses, PCGrad, constraints, weighting schemes).
- `scripts/train.py` training CLI.
- `.pdata/` preprocessed H5 outputs (generated).

## Quickstart
```bash
# create venv + install deps
make setup

# preprocess raw H5s from .data → .pdata
make preprocess

# train (direct F_true target)
make train

# train on residual (F_true - F_box) targets
make train-residual

# quick one-batch smoke test
make smoke
```

## Data and shapes
- Inputs: `F_init_flat` (24) concatenated with `F_box_flat` (24) → $$\mathbb{R}^{48}$$.
- Targets: `F_true_flat` → $$\mathbb{R}^{24}$$. If `predict_residual=true`, target becomes $$F_{\text{true}} - F_{\text{box}}$$.
- Scaling: `config/model.yaml` sets `target_scale: 1e-33`, applied to both inputs and targets in the loader.

## Preprocessing (Julia)
Config: `config/pre_process.yaml` (paths already point to `.data`/`.pdata`).
Command:
```bash
make preprocess
```
Outputs per-sim files in `.pdata/*_preprocessed.h5` and a combined `.pdata/preprocessed_all.h5` with datasets: `input (N,48)`, `target (N,24)`, plus diagnostic arrays.

## Training configs
- `config/model.yaml`: data/model/optimizer/scheduler/training/loss hyperparams.
- `config/train_parms.yaml`: loss weighting (`kendall_gal`|`dwa`|`gradnorm`|`none`), `pcgrad` toggle, `predict_residual` toggle, constraint settings, checkpoint cadence, early stopping (`early_stop.enabled`, `patience`, `min_delta`), and LR scheduler (`lr_scheduler.type`, `factor`, `patience`, `min_lr`, `threshold`).

CLI options (scripts/train.py):
```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --config config/model.yaml \
  --train-parms config/train_parms.yaml \
  [--predict-residual]   # overrides train_parms
```

## Make targets
- `make setup` / `make install`: venv + pip install.
- `make preprocess`: run Julia preprocessing.
- `make train`: run training with current configs.
- `make train-residual`: force residual targets.
- `make smoke`: one-batch forward/backward sanity check (num_workers=0).
- `make clean`: remove `.venv`, `checkpoints/`, `results/`.
- `make slurm-train`: submit the Slurm script at `.slurm/train.sbatch` (run on the cluster where `sbatch` is available).
- `make plot-loss LOG=path/to/log`: parse `Epoch ... train_loss=... val_loss=...` lines and save `results/loss_curve.png`.

## Slurm submission
A template Slurm script lives at `.slurm/train.sbatch`. Adjust `--chdir` to your cluster path and venv path as needed (defaults to `/nfs/home/jmcguig1/venv_Emu`). Logs go to `.slurm/output/train-%j.out|err`. It runs `make install`, checks for `.pdata/preprocessed_all.h5` and runs `make preprocess` if missing, then runs `make train`, honoring `VENV` from the environment so you can override the venv path without editing the Makefile.

## Notes
- Ensure `.pdata/preprocessed_all.h5` exists before training (run `make preprocess`).
- `PYTHONPATH=src` is set in Makefile targets; if running commands manually, set it or install the package in editable mode.
- MPS/CPU: pin_memory warnings on macOS MPS are harmless for this workflow.
