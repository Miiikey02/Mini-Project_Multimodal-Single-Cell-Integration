import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_c   = pred   - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)
        num      = (pred_c * target_c).sum(dim=1)
        denom    = pred_c.norm(dim=1) * target_c.norm(dim=1) + self.eps
        return -(num / denom).mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        pearson_weight: float = 1.0,
        mse_weight: float     = 1.0,
        burnin_epochs: int    = 10,
        eps: float            = 1e-8,
    ):
        super().__init__()
        self.pearson_weight = pearson_weight
        self.mse_weight     = mse_weight
        self.burnin_epochs  = burnin_epochs
        self.pearson        = PearsonLoss(eps=eps)
        self.mse            = nn.MSELoss()
        self._epoch         = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_val = self.mse(pred, target)
        if self._epoch <= self.burnin_epochs:
            return mse_val
        pearson_val = self.pearson(pred, target)
        return self.pearson_weight * pearson_val + self.mse_weight * mse_val
