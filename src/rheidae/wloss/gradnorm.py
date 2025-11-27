from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


class GradNormWeighter(nn.Module):
    """
    GradNorm-based multi-task loss balancing.

    Reference: "GradNorm: Gradient Normalization for Adaptive Loss Balancing
    in Deep Multitask Networks" (Chen et al., 2018).

    You pass in:
      - task_losses: dict of {name: scalar loss}
      - shared_params: iterable of parameters shared across all tasks
    """

    def __init__(
        self,
        task_names: List[str],
        alpha: float = 1.5,
        lambda_gradnorm: float = 0.1,
        device=None,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.alpha = alpha
        self.lambda_gradnorm = lambda_gradnorm
        self.device = device

        # Learnable positive weights; initialize to 1
        self.weights = nn.Parameter(torch.ones(len(self.task_names)))

        # Store initial task losses for relative training rate computation
        self.register_buffer(
            "initial_losses",
            torch.zeros(len(self.task_names), dtype=torch.float32),
        )
        self._initialized = False

    def forward(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_params: Iterable[nn.Parameter],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            total_loss: scalar (includes GradNorm term)
            per_task_weighted_losses: list of w_i * L_i, to send to PCGrad if you want
        """
        assert set(task_losses.keys()) == set(self.task_names)
        device = next(iter(task_losses.values())).device

        # stack current task losses in fixed order
        L_vec = torch.stack([task_losses[name] for name in self.task_names])  # (T,)

        if not self._initialized:
            with torch.no_grad():
                self.initial_losses.copy_(L_vec.detach())
            self._initialized = True

        # Normalize weights to keep them in a reasonable range
        w = torch.relu(self.weights)  # ensure non-negative
        w_norm = len(self.task_names) * w / (w.sum() + 1e-8)

        # Weighted per-task losses
        weighted_losses = [w_norm[i] * L_vec[i] for i in range(len(self.task_names))]
        total_weighted = sum(weighted_losses)

        # Compute gradient norms g_i = || d(w_i L_i)/d theta ||
        grad_norms = []
        shared_params = list(shared_params)
        for i in range(len(self.task_names)):
            self.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(
                weighted_losses[i],
                shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            # L2 norm over all shared params
            squared = [(g**2).sum() for g in grads if g is not None]
            if len(squared) == 0:
                g_norm = torch.tensor(0.0, device=device)
            else:
                g_norm = torch.sqrt(sum(squared) + 1e-16)
            grad_norms.append(g_norm)

        grad_norms = torch.stack(grad_norms)  # (T,)
        mean_grad_norm = grad_norms.mean()

        # Relative inverse training rates
        # r_i = (L_i / L_i0) / mean_j (L_j / L_j0)
        loss_ratio = L_vec / (self.initial_losses + 1e-8)
        mean_ratio = loss_ratio.mean()
        target_grad_norm = mean_grad_norm * (loss_ratio / mean_ratio) ** self.alpha

        # GradNorm objective: sum |g_i - g*_i|
        gradnorm_loss = torch.abs(grad_norms - target_grad_norm.detach()).sum()

        total_loss = total_weighted + self.lambda_gradnorm * gradnorm_loss

        return total_loss, weighted_losses
