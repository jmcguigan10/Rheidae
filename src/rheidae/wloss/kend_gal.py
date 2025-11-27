from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class KendallGalWeighter(nn.Module):
    """
    Homoscedastic uncertainty weighting (Kendall & Gal).
    Learns log_sigma_i per task.

    L_total = sum_i [ 0.5 * exp(-2 log_sigma_i) * L_i + log_sigma_i ]
    """

    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = list(task_names)
        self.log_sigmas = nn.Parameter(torch.zeros(len(self.task_names)))

    def forward(
        self, task_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            total_loss: scalar (for logging / simple training)
            per_task_weighted: list of weighted losses (one per task) to feed into PCGrad
        """
        assert set(task_losses.keys()) == set(self.task_names)
        total_loss = torch.tensor(0.0, device=self.log_sigmas.device)
        weighted_losses: List[torch.Tensor] = []

        for i, name in enumerate(self.task_names):
            loss_i = task_losses[name]
            log_sigma_i = self.log_sigmas[i]
            inv_var_i = torch.exp(-2.0 * log_sigma_i)  # 1 / sigma^2

            weighted_i = 0.5 * inv_var_i * loss_i
            weighted_losses.append(weighted_i)

            total_loss = total_loss + weighted_i + log_sigma_i

        return total_loss, weighted_losses
