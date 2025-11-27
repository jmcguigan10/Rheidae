from typing import List

import torch


class PCGrad:
    """
    Wrapper around a base optimizer that applies PCGrad to a list of task losses.

    Usage:
        pc_optim = PCGrad(torch.optim.AdamW(model.parameters(), lr=1e-3))

        task_losses = [loss_A, loss_B, loss_C]  # already appropriately weighted
        pc_optim.zero_grad()
        pc_optim.pc_backward(task_losses)
        pc_optim.step()
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optim = optimizer

    @property
    def param_groups(self):
        return self._optim.param_groups

    def zero_grad(self):
        self._optim.zero_grad()

    @torch.no_grad()
    def _project_grads(
        self, grads: List[List[torch.Tensor]]
    ) -> List[List[torch.Tensor]]:
        """
        grads: list over tasks, each element is list over params (same shapes as params.grad)
        returns projected grads with same structure.
        """
        T = len(grads)
        projected = [[g.clone() for g in task_grads] for task_grads in grads]

        for i in range(T):
            for j in range(T):
                if i == j:
                    continue

                # Compute dot products g_i · g_j and g_j · g_j
                dot_ij = torch.tensor(0.0, device=projected[i][0].device)
                dot_jj = torch.tensor(0.0, device=projected[i][0].device)
                for g_i_p, g_j_p in zip(projected[i], projected[j]):
                    dot_ij += (g_i_p * g_j_p).sum()
                    dot_jj += (g_j_p * g_j_p).sum()

                if dot_ij.item() < 0:  # conflicting directions
                    coeff = dot_ij / (dot_jj + 1e-12)
                    for k in range(len(projected[i])):
                        projected[i][k] = projected[i][k] - coeff * grads[j][k]

        return projected

    def pc_backward(self, task_losses: List[torch.Tensor]):
        """
        task_losses: list of scalar losses (one per task / category)
        """
        shared_params: List[torch.nn.Parameter] = [
            p
            for group in self._optim.param_groups
            for p in group["params"]
            if p.requires_grad
        ]

        # collect grads per task
        raw_grads: List[List[torch.Tensor]] = []
        for loss in task_losses:
            self._optim.zero_grad()
            loss.backward(retain_graph=True)
            grads_for_task = []
            for p in shared_params:
                if p.grad is None:
                    grads_for_task.append(torch.zeros_like(p))
                else:
                    grads_for_task.append(p.grad.detach().clone())
            raw_grads.append(grads_for_task)

        # project
        proj_grads = self._project_grads(raw_grads)

        # accumulate projected gradients into .grad and let optimizer step
        self._optim.zero_grad()
        for param_idx, p in enumerate(shared_params):
            g_sum = torch.zeros_like(p)
            for task_idx in range(len(task_losses)):
                g_sum = g_sum + proj_grads[task_idx][param_idx]
            p.grad = g_sum

    def step(self):
        self._optim.step()
