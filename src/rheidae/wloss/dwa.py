from typing import Dict, List, Tuple

import torch


class DWALossWeighter:
    """
    Dynamic Weight Average (DWA) for multi-task loss balancing.

    Uses the ratio of previous losses L(t-1)/L(t-2) to compute softmax weights.
    """

    def __init__(self, task_names: List[str], T: float = 2.0, device=None):
        self.task_names = list(task_names)
        self.T = T
        self.device = device
        self.history = {name: [] for name in self.task_names}

    def forward(
        self, task_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Record current scalar values (detach to avoid weird graphs)
        current_vals = {
            name: task_losses[name].detach().item() for name in self.task_names
        }

        ratios = []
        for name in self.task_names:
            hist = self.history[name]
            if len(hist) < 2:
                ratios.append(1.0)
            else:
                r = hist[-1] / (hist[-2] + 1e-8)
                ratios.append(r)

        ratios_tensor = torch.tensor(ratios, dtype=torch.float32, device=self.device)
        weights = torch.softmax(ratios_tensor / self.T, dim=0)

        # Update history
        for name in self.task_names:
            self.history[name].append(current_vals[name])

        total_loss = torch.tensor(0.0, device=task_losses[self.task_names[0]].device)
        weighted_losses: List[torch.Tensor] = []

        for w, name in zip(weights, self.task_names):
            w = w.to(task_losses[name].device)
            w_loss = w * task_losses[name]
            weighted_losses.append(w_loss)
            total_loss = total_loss + w_loss

        return total_loss, weighted_losses
