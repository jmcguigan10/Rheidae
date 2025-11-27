from typing import Callable

import torch
import torch.nn as nn


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "relu":
        return lambda: nn.ReLU(inplace=True)
    if name == "elu":
        return lambda: nn.ELU(inplace=True)
    if name == "silu":
        return lambda: nn.SiLU(inplace=True)
    if name == "gelu":
        return lambda: nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable[[], nn.Module],
        dropout: float,
        use_batch_norm: bool,
    ):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out) if use_batch_norm else None
        self.activation = activation()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.use_residual = dim_in == dim_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if self.bn is not None:
            h = self.bn(h)
        h = self.activation(h)
        h = self.dropout(h)
        if self.use_residual:
            h = h + x
        return h


class ResidualFFIModel(nn.Module):
    """
    Predicts ΔF (residual on top of Box3D) from invariant features.
    """

    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 256,
        hidden_layers: int = 4,
        activation: str = "gelu",
        dropout: float = 0.05,
        use_batch_norm: bool = True,
        output_dim: int = 24,
    ):
        super().__init__()

        act_factory = _activation_factory(activation)
        blocks = []
        dims = [input_dim] + [hidden_dim] * hidden_layers
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            blocks.append(
                ResidualBlock(dim_in, dim_out, act_factory, dropout, use_batch_norm)
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        return self.head(h)
