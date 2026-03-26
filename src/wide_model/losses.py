"""
Loss functions for the wide model.

PearsonLoss
    Differentiable row-wise mean negative Pearson correlation.
    Directly optimises MRPC — the competition metric.

CombinedLoss
    Weighted sum of Pearson + MSE losses with optional MSE-only burn-in.
    Burn-in stabilises early training before the Pearson gradient is engaged.

Why combine both losses
-----------------------
- MSE loss provides dense, stable gradients on absolute scale
- Pearson loss aligns the optimisation objective with the evaluation metric
  (rank/shape of predictions per cell, not absolute values)
- Using both avoids degenerate solutions that maximise Pearson by predicting
  a constant offset (MSE prevents this)
"""

import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    """
    Mean negative row-wise Pearson correlation (differentiable).

    For each row i: computes Pearson r(pred[i,:], target[i,:])
    Returns -mean(r) so that minimising this loss maximises MRPC.

    A small epsilon prevents division by zero for near-constant rows
    during early training.
    """

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
    """
    Weighted Pearson + MSE loss with MSE-only burn-in period.

    During burn-in (epoch <= burnin_epochs): MSE only.
    After burn-in: pearson_weight * PearsonLoss + mse_weight * MSELoss.

    Parameters
    ----------
    pearson_weight : Weight on Pearson loss term (default 1.0).
    mse_weight     : Weight on MSE loss term (default 1.0).
    burnin_epochs  : Epochs to use MSE-only before enabling Pearson (default 10).
    eps            : Epsilon for Pearson denominator stability.
    """

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
        """Call at the start of each epoch to control burn-in behaviour."""
        self._epoch = epoch

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_val = self.mse(pred, target)
        if self._epoch <= self.burnin_epochs:
            return mse_val
        pearson_val = self.pearson(pred, target)
        return self.pearson_weight * pearson_val + self.mse_weight * mse_val
