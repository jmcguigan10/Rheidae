from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _section(
    raw: Dict[str, Any], name: str, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    section = raw.get(name, {}) or {}
    return {k: section.get(k, v) for k, v in defaults.items()}


@dataclass
class DataConfig:
    processed_path: str
    upload_dir: str
    batch_size: int
    num_workers: int
    val_split: float
    test_split: float
    seed: int
    pin_memory: bool
    drop_last: bool
    target_scale: Optional[float]


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int
    hidden_layers: int
    activation: str
    dropout: float
    use_batch_norm: bool
    output_dim: int


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float


@dataclass
class SchedulerConfig:
    warmup_steps: int
    min_lr: float
    max_lr: Optional[float]


@dataclass
class TrainingConfig:
    epochs: int
    grad_clip: float
    mixed_precision: bool
    log_interval: int
    checkpoint_dir: str
    results_dir: str
    resume_from: Optional[str]


@dataclass
class LossConfig:
    w_density: float
    w_magnitude: float
    w_direction: float
    w_unphysical: float
    w_residual_l1: float
    density_floor: float
    speed_of_light: float
    eps: float


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    loss: LossConfig

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        cfg_path = Path(path)
        raw = yaml.safe_load(cfg_path.read_text()) or {}

        data_defaults = {
            "processed_path": ".pdata/preprocessed_all.h5",
            "upload_dir": "data/uploads",
            "batch_size": 256,
            "num_workers": 4,
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,
            "pin_memory": True,
            "drop_last": False,
            "target_scale": None,
        }
        model_defaults = {
            "input_dim": 48,
            "hidden_dim": 256,
            "hidden_layers": 4,
            "activation": "gelu",
            "dropout": 0.05,
            "use_batch_norm": True,
            "output_dim": 24,
        }
        optim_defaults = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }
        sched_defaults = {
            "warmup_steps": 500,
            "min_lr": 1e-6,
            "max_lr": None,
        }
        training_defaults = {
            "epochs": 50,
            "grad_clip": 1.0,
            "mixed_precision": True,
            "log_interval": 10,
            "checkpoint_dir": "checkpoints",
            "results_dir": "results",
            "resume_from": None,
        }
        loss_defaults = {
            "w_density": 1.0,
            "w_magnitude": 1.0,
            "w_direction": 1.0,
            "w_unphysical": 25.0,
            "w_residual_l1": 0.2,
            "density_floor": 0.1,
            "speed_of_light": 1.0,
            "eps": 1e-8,
        }

        data_cfg = DataConfig(**_section(raw, "data", data_defaults))
        model_cfg = ModelConfig(**_section(raw, "model", model_defaults))
        optim_cfg = _section(raw, "optimizer", optim_defaults)
        optimizer_cfg = OptimizerConfig(
            lr=optim_cfg["lr"],
            weight_decay=optim_cfg["weight_decay"],
            betas=tuple(optim_cfg["betas"]),
            eps=optim_cfg["eps"],
        )
        scheduler_cfg = SchedulerConfig(**_section(raw, "scheduler", sched_defaults))
        training_cfg = TrainingConfig(**_section(raw, "training", training_defaults))
        loss_cfg = LossConfig(**_section(raw, "loss", loss_defaults))

        return cls(
            data=data_cfg,
            model=model_cfg,
            optimizer=optimizer_cfg,
            scheduler=scheduler_cfg,
            training=training_cfg,
            loss=loss_cfg,
        )

    def ensure_directories(self) -> None:
        Path(self.data.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.results_dir).mkdir(parents=True, exist_ok=True)
