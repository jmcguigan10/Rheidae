import torch


class EqualityConstraintManager:
    """
    Simple equality-constraint manager using Lagrange multipliers + optional quadratic penalty.

    Constraints: c(theta) ~ 0  (vector of size M)

    L_aug(theta, lambda) = base_loss(theta)
                         + lambda^T c(theta)
                         + 0.5 * rho * ||c(theta)||^2

    lambda updated by dual ascent: lambda <- lambda + lr_lambda * c(theta).
    """

    def __init__(
        self,
        num_constraints: int,
        lr_lambda: float = 1e-2,
        rho: float = 0.0,
        device=None,
    ):
        self.num_constraints = num_constraints
        self.lr_lambda = lr_lambda
        self.rho = rho
        self.lambdas = torch.zeros(num_constraints, device=device)

    def augment_loss(
        self, base_loss: torch.Tensor, c_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        base_loss: scalar torch.Tensor
        c_vec: tensor of shape (num_constraints,), constraint values

        Returns augmented scalar loss (Lagrangian).
        """
        assert c_vec.shape[-1] == self.num_constraints
        lag_term = (self.lambdas.to(c_vec.device) * c_vec).sum()
        penalty_term = 0.5 * self.rho * (c_vec**2).sum()
        return base_loss + lag_term + penalty_term

    def update_multipliers(self, c_vec: torch.Tensor):
        """
        Dual ascent step on lambda: lambda <- lambda + lr_lambda * c(theta)

        Call this *after* optimizer.step(), with c(theta) computed at the same step.
        """
        with torch.no_grad():
            self.lambdas = self.lambdas.to(c_vec.device)
            self.lambdas += self.lr_lambda * c_vec.detach()
