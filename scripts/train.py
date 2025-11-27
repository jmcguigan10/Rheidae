from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

from rheidae.config import ExperimentConfig, LossConfig
from rheidae.data_loader import build_dataloaders
from rheidae.losses import compute_flux_losses, compute_constraint_vec
from rheidae.model import ResidualFFIModel
from rheidae.pcgrad import PCGrad
from rheidae.ecm import EqualityConstraintManager
from rheidae.wloss import DWALossWeighter, GradNormWeighter, KendallGalWeighter


def _active_tasks(loss_cfg: LossConfig) -> List[str]:
    tasks = []
    if loss_cfg.w_density > 0:
        tasks.append("density")
    if loss_cfg.w_magnitude > 0:
        tasks.append("magnitude")
    if loss_cfg.w_direction > 0:
        tasks.append("direction")
    if loss_cfg.w_residual_l1 > 0:
        tasks.append("residual_l1")
    if loss_cfg.w_unphysical > 0:
        tasks.append("unphysical")
    if not tasks:
        raise ValueError("No active loss components; enable at least one weight.")
    return tasks


def _build_weighter(weighting_cfg: Dict, task_names: List[str], device) -> Optional[object]:
    scheme = (weighting_cfg or {}).get("scheme", "none")
    params = (weighting_cfg or {}).get("params", {}) or {}

    if scheme == "none":
        return None
    if scheme == "kendall_gal":
        return KendallGalWeighter(task_names)
    if scheme == "dwa":
        T = float(params.get("T", 2.0))
        return DWALossWeighter(task_names, T=T, device=device)
    if scheme == "gradnorm":
        alpha = float(params.get("alpha", 1.5))
        lam = float(params.get("lambda_gradnorm", 0.1))
        return GradNormWeighter(task_names, alpha=alpha, lambda_gradnorm=lam, device=device)
    raise ValueError(f"Unknown weighting scheme '{scheme}'")


def _load_train_parms(path: Path) -> Dict:
    return yaml.safe_load(path.read_text()) or {}


def _maybe_clip_gradients(model: torch.nn.Module, max_norm: float):
    if max_norm and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_cfg: LossConfig,
    weighter,
    pcgrad: Optional[PCGrad],
    optimizer: torch.optim.Optimizer,
    constraint_mgr: Optional[EqualityConstraintManager],
    device,
    grad_clip: float,
) -> float:
    model.train()
    total = 0.0
    n_batches = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        preds = model(inputs)
        task_losses = compute_flux_losses(preds, targets, loss_cfg)

        if weighter is None:
            total_task_loss = sum(task_losses.values())
            per_task_weighted = list(task_losses.values())
        elif isinstance(weighter, GradNormWeighter):
            total_task_loss, per_task_weighted = weighter(
                task_losses, shared_params=model.parameters()
            )
        else:
            total_task_loss, per_task_weighted = weighter(task_losses)

        augmented_loss = total_task_loss
        extra_task = []
        if constraint_mgr is not None:
            c_vec = compute_constraint_vec(preds, loss_cfg)
            augmented_loss = constraint_mgr.augment_loss(total_task_loss, c_vec)
            extra_task = [augmented_loss - total_task_loss]

        optimizer.zero_grad()
        if pcgrad is not None:
            pcgrad.pc_backward(per_task_weighted + extra_task)
            _maybe_clip_gradients(model, grad_clip)
            pcgrad.step()
        else:
            augmented_loss.backward()
            _maybe_clip_gradients(model, grad_clip)
            optimizer.step()

        if constraint_mgr is not None:
            constraint_mgr.update_multipliers(c_vec)

        total += augmented_loss.detach().item()
        n_batches += 1

    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_cfg: LossConfig,
    device,
) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        preds = model(inputs)
        task_losses = compute_flux_losses(preds, targets, loss_cfg)
        total += sum(task_losses.values()).item()
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate_zero_residual(
    loader: torch.utils.data.DataLoader, loss_cfg: LossConfig, device
) -> float:
    """
    Baseline loss if we predict zero residual (i.e., use F_box directly as prediction).
    When predict_residual=True, targets = F_true - F_box, so zero corresponds to Box3D.
    When predict_residual=False, this returns the loss of a zero tensor vs target.
    """
    total = 0.0
    n_batches = 0
    zero_pred = None
    for batch in loader:
        targets = batch["target"].to(device)
        if zero_pred is None:
            zero_pred = torch.zeros_like(targets)
        task_losses = compute_flux_losses(zero_pred, targets, loss_cfg)
        total += sum(task_losses.values()).item()
        n_batches += 1
    return total / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/model.yaml",
        help="Path to experiment config (matches config.py schema).",
    )
    parser.add_argument(
        "--train-parms",
        default="config/train_parms.yaml",
        help="Path to training control parameters.",
    )
    parser.add_argument(
        "--predict-residual",
        action="store_true",
        help="Override train_parms to train on (F_true - F_box).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier; defaults to timestamp YYYYmmdd-HHMMSS.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier; defaults to timestamp YYYYmmdd-HHMMSS.",
    )
    args = parser.parse_args()

    exp_cfg = ExperimentConfig.load(args.config)
    train_cfg = _load_train_parms(Path(args.train_parms))

    predict_residual = args.predict_residual or bool(train_cfg.get("predict_residual", False))
    loaders = build_dataloaders(exp_cfg.data, predict_residual=predict_residual)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResidualFFIModel(
        input_dim=exp_cfg.model.input_dim,
        hidden_dim=exp_cfg.model.hidden_dim,
        hidden_layers=exp_cfg.model.hidden_layers,
        activation=exp_cfg.model.activation,
        dropout=exp_cfg.model.dropout,
        use_batch_norm=exp_cfg.model.use_batch_norm,
        output_dim=exp_cfg.model.output_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_cfg.optimizer.lr,
        weight_decay=exp_cfg.optimizer.weight_decay,
        betas=exp_cfg.optimizer.betas,
        eps=exp_cfg.optimizer.eps,
    )
    pcgrad = PCGrad(optimizer) if train_cfg.get("pcgrad", False) else None

    constraint_cfg = train_cfg.get("constraints", {}) or {}
    constraint_mgr = None
    if constraint_cfg.get("enabled", False) and constraint_cfg.get("num_constraints", 0) > 0:
        constraint_mgr = EqualityConstraintManager(
            num_constraints=int(constraint_cfg["num_constraints"]),
            lr_lambda=float(constraint_cfg.get("lr_lambda", 1e-2)),
            rho=float(constraint_cfg.get("rho", 0.0)),
            device=device,
        )

    loss_cfg = exp_cfg.loss
    task_names = _active_tasks(loss_cfg)
    weighter = _build_weighter(train_cfg.get("weighting", {}), task_names, device)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")

    best_val = float("inf")
    save_every = int(train_cfg.get("logging", {}).get("save_every", 0) or 0)
    checkpoint_root = Path(exp_cfg.training.checkpoint_dir)
    checkpoint_dir = checkpoint_root / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    early_cfg = train_cfg.get("early_stop", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience", 5))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    no_improve = 0
    best_epoch = 0

    # LR scheduler (optional)
    sched_cfg = train_cfg.get("lr_scheduler", {}) or {}
    scheduler = None
    sched_type = sched_cfg.get("type", "none")
    if sched_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
            threshold=float(sched_cfg.get("threshold", 1e-3)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )

    print(f"Run ID: {run_id}")

    for epoch in range(1, exp_cfg.training.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            loss_cfg,
            weighter,
            pcgrad,
            optimizer,
            constraint_mgr,
            device,
            exp_cfg.training.grad_clip,
        )
        val_loss = evaluate(model, loaders["val"], loss_cfg, device)
        print(
            f"Run {run_id} Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(val_loss)

        improved = val_loss < (best_val - min_delta)
        if improved or best_val == float("inf"):
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                checkpoint_dir / f"best-{run_id}.pt",
            )
        else:
            no_improve += 1

        if save_every > 0 and epoch % save_every == 0:
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                checkpoint_dir / f"epoch_{epoch}-{run_id}.pt",
            )

        if early_enabled and no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch} (patience={patience}, best_epoch={best_epoch}, best_val={best_val:.4f})"
            )
            break

    test_loss = evaluate(model, loaders["test"], loss_cfg, device)
    zero_baseline = evaluate_zero_residual(loaders["test"], loss_cfg, device)
    print(f"Run {run_id} Test loss: {test_loss:.4f} | Box3D baseline (zero residual) loss: {zero_baseline:.4f}")


if __name__ == "__main__":
    main()
