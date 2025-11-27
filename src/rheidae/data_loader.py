from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from rheidae.config import DataConfig


def _maybe_transpose(arr: np.ndarray, expected_feat_dim: Optional[int]) -> np.ndarray:
    """
    Julia wrote the HDF5 files in column-major order, so some datasets load as (feat, N).
    If the first axis matches the expected feature dim and the second is larger, transpose.
    """
    if arr.ndim != 2:
        return arr

    if expected_feat_dim is not None:
        if arr.shape[0] == expected_feat_dim and arr.shape[1] != expected_feat_dim:
            return arr.T
        if arr.shape[1] == expected_feat_dim:
            return arr

    # Fallback: if the first axis is much smaller than the second, assume it's (feat, N).
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128:
        return arr.T
    return arr


def _load_arrays(
    path: str, expected_input_dim: int, expected_target_dim: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    with h5py.File(path, "r") as h5:
        inputs = _maybe_transpose(h5["input"][()], expected_input_dim)
        targets = _maybe_transpose(h5["target"][()], expected_target_dim)
        f_box = None
        if "F_box_flat" in h5:
            f_box = _maybe_transpose(h5["F_box_flat"][()], expected_target_dim)

    return inputs.astype(np.float32), targets.astype(np.float32), (
        None if f_box is None else f_box.astype(np.float32)
    )


class FluxDataset(Dataset):
    """
    Simple in-memory dataset for the preprocessed flux files.
    """

    def __init__(
        self,
        path: str,
        expected_input_dim: int,
        expected_target_dim: int,
        predict_residual: bool = False,
        target_scale: Optional[float] = None,
    ):
        inputs_np, targets_np, f_box_np = _load_arrays(
            path, expected_input_dim, expected_target_dim
        )

        scale = None if target_scale is None else float(target_scale)
        if scale is not None:
            inputs_np = inputs_np * scale
            targets_np = targets_np * scale
            if f_box_np is not None:
                f_box_np = f_box_np * scale

        if predict_residual:
            if f_box_np is None:
                raise ValueError(
                    "predict_residual=True requires F_box_flat in the H5 file"
                )
            targets_np = targets_np - f_box_np

        if inputs_np.shape[1] != expected_input_dim:
            raise ValueError(
                f"Unexpected input shape {inputs_np.shape}; expected feature dim {expected_input_dim}"
            )
        if targets_np.shape[1] != expected_target_dim:
            raise ValueError(
                f"Unexpected target shape {targets_np.shape}; expected feature dim {expected_target_dim}"
            )

        if target_scale is not None:
            targets_np = targets_np * float(target_scale)

        self.inputs = torch.from_numpy(inputs_np)
        self.targets = torch.from_numpy(targets_np)
        self.f_box = None if f_box_np is None else torch.from_numpy(f_box_np)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"input": self.inputs[idx], "target": self.targets[idx]}
        if self.f_box is not None:
            sample["f_box"] = self.f_box[idx]
        return sample


def build_dataloaders(
    data_cfg: DataConfig,
    predict_residual: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from the H5 file defined in the data config.
    """
    dataset = FluxDataset(
        data_cfg.processed_path,
        expected_input_dim=48,
        expected_target_dim=24,
        predict_residual=predict_residual,
        target_scale=data_cfg.target_scale,
    )

    n_total = len(dataset)
    n_test = int(round(data_cfg.test_split * n_total))
    n_val = int(round(data_cfg.val_split * n_total))
    n_train = max(n_total - n_val - n_test, 1)
    # Adjust to ensure sums exactly match n_total
    excess = (n_train + n_val + n_test) - n_total
    if excess > 0:
        n_train = max(n_train - excess, 1)
    elif excess < 0:
        n_train = n_train - excess  # add missing samples to train

    generator = torch.Generator().manual_seed(data_cfg.seed)
    splits = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )
    train_ds, val_ds, test_ds = splits

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            drop_last=data_cfg.drop_last,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=data_cfg.batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=data_cfg.batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            drop_last=False,
        ),
    }
    return loaders
