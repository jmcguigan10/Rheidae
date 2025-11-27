from __future__ import annotations

from typing import Dict

import torch

from rheidae.config import LossConfig


def _reshape_flux(flat: torch.Tensor) -> torch.Tensor:
    """
    Reshape (B, 24) -> (B, 6, 4).
    Assumes species-major ordering with components (x, y, z, t).
    """
    if flat.ndim != 2 or flat.shape[1] != 24:
        raise ValueError(f"Expected flat flux of shape (B, 24); got {tuple(flat.shape)}")
    return flat.view(flat.shape[0], 6, 4)


def compute_flux_losses(
    preds_flat: torch.Tensor, target_flat: torch.Tensor, cfg: LossConfig
) -> Dict[str, torch.Tensor]:
    """
    Build per-task loss components. Returns a dict keyed by loss name.
    Each value already includes the static weight from the config.
    """
    preds = _reshape_flux(preds_flat)
    target = _reshape_flux(target_flat)

    pred_vec = preds[:, :, 0:3]
    true_vec = target[:, :, 0:3]
    pred_den = preds[:, :, 3]
    true_den = target[:, :, 3]

    eps = cfg.eps

    losses: Dict[str, torch.Tensor] = {}

    if cfg.w_density > 0:
        density_mse = torch.mean((pred_den - true_den) ** 2)
        losses["density"] = cfg.w_density * density_mse

    if cfg.w_magnitude > 0:
        mag_pred = torch.linalg.norm(pred_vec, dim=-1)
        mag_true = torch.linalg.norm(true_vec, dim=-1)
        magnitude_mse = torch.mean((mag_pred - mag_true) ** 2)
        losses["magnitude"] = cfg.w_magnitude * magnitude_mse

    if cfg.w_direction > 0:
        mag_pred = torch.linalg.norm(pred_vec, dim=-1).clamp(min=eps)
        mag_true = torch.linalg.norm(true_vec, dim=-1).clamp(min=eps)
        cos_sim = torch.sum(pred_vec * true_vec, dim=-1) / (mag_pred * mag_true)
        direction_loss = torch.mean(1.0 - cos_sim)
        losses["direction"] = cfg.w_direction * direction_loss

    if cfg.w_residual_l1 > 0:
        residual_l1 = torch.mean(torch.abs(preds - target))
        losses["residual_l1"] = cfg.w_residual_l1 * residual_l1

    if cfg.w_unphysical > 0:
        mag_pred = torch.linalg.norm(pred_vec, dim=-1)
        density_clip = pred_den.abs().clamp(min=cfg.density_floor)
        speed_ratio = mag_pred / density_clip
        excess = torch.relu(speed_ratio - cfg.speed_of_light)
        unphys_loss = torch.mean(excess**2)
        losses["unphysical"] = cfg.w_unphysical * unphys_loss

    if not losses:
        raise ValueError("No loss components enabled; check the loss weights in config.")

    return losses
