PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PYTHONPATH := src
LOG := slurm/output

.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: setup install preprocess train train-residual smoke clean slurm-train plot-loss

setup: install

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV) requirements.txt
	$(PIP) install -r requirements.txt

preprocess:
	julia --project=pre_process pre_process/preprocess.jl --config config/pre_process.yaml

train: install
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/train.py --config config/model.yaml --train-parms config/train_parms.yaml

train-residual: install
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/train.py --config config/model.yaml --train-parms config/train_parms.yaml --predict-residual

smoke: install
	PYTHONPATH=$(PYTHONPATH) $(PY) -c "from rheidae.config import ExperimentConfig; from rheidae.data_loader import build_dataloaders; from rheidae.model import ResidualFFIModel; from rheidae.losses import compute_flux_losses; from rheidae.pcgrad import PCGrad; import torch; cfg=ExperimentConfig.load('config/model.yaml'); cfg.data.num_workers=0; loaders=build_dataloaders(cfg.data, predict_residual=False); batch=next(iter(loaders['train'])); inputs,targets=batch['input'],batch['target']; model=ResidualFFIModel(input_dim=cfg.model.input_dim, output_dim=cfg.model.output_dim); optim=PCGrad(torch.optim.AdamW(model.parameters(), lr=1e-3)); preds=model(inputs); losses=compute_flux_losses(preds, targets, cfg.loss); optim.zero_grad(); optim.pc_backward(list(losses.values())); optim.step(); print('smoke ok', tuple(inputs.shape), {k: float(v) for k,v in losses.items()})"

clean:
	rm -rf $(VENV) checkpoints results

SLURM_SCRIPT := slurm/train.sbatch

slurm-train:
	@[ -f $(SLURM_SCRIPT) ] || (echo "Missing $(SLURM_SCRIPT); edit or copy it before submitting." && exit 1)
	sbatch $(SLURM_SCRIPT)

plot-loss: install
	@[ -n "$(LOG)" ] || (echo "Set LOG=<path to training log containing 'Epoch ... train_loss=... val_loss=...'>"; exit 1)
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/plot_losses.py --log $(LOG) --out results/loss_curve.png
