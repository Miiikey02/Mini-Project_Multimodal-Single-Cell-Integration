import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluate import mean_rowwise_pearson, pearson_per_row

class MultiomeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-5, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = float("-inf") if mode == "max" else float("inf")
        self.counter    = 0
        self.best_epoch = 0
        self.stop       = False

    def step(self, value: float, epoch: int) -> bool:
        improved = (
            value > self.best + self.min_delta if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best       = value
            self.best_epoch = epoch
            self.counter    = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False


class PearsonLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_c   = pred   - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)
        num   = (pred_c * target_c).sum(dim=1)
        denom = pred_c.norm(dim=1) * target_c.norm(dim=1) + self.eps
        return -(num / denom).mean()


def predict_batched(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    parts = []
    X_f32 = X.astype(np.float32)
    with torch.no_grad():
        for start in range(0, len(X_f32), batch_size):
            xb = torch.from_numpy(X_f32[start: start + batch_size]).to(device)
            parts.append(model(xb).cpu().numpy())
    return np.vstack(parts)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float              = 3e-4,
        weight_decay: float    = 1e-4,
        batch_size: int        = 512,
        max_epochs: int        = 150,
        patience: int          = 15,
        scheduler_type: str    = "plateau",
        lr_patience: int       = 5,
        lr_factor: float       = 0.5,
        cosine_t0: int         = 25,
        min_lr: float          = 1e-6,
        eval_batch: int        = 512,
        loss_type: str         = "mse",
        burnin_epochs: int     = 10,
        pearson_weight: float  = 1.0,
    ):
        self.model          = model.to(device)
        self.device         = device
        self.batch_size     = batch_size
        self.max_epochs     = max_epochs
        self.eval_batch     = eval_batch
        self.scheduler_type = scheduler_type
        self.loss_type      = loss_type
        self.burnin_epochs  = burnin_epochs
        self.pearson_weight = pearson_weight
        self._current_epoch = 0

        self.mse_loss     = nn.MSELoss()
        self.pearson_loss = PearsonLoss()
        self.criterion    = self.mse_loss 
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if scheduler_type == "cosine_warm":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=cosine_t0, T_mult=1, eta_min=min_lr,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=lr_factor,
                patience=lr_patience, min_lr=min_lr,
            )
        self.early_stop = EarlyStopping(patience=patience, mode="max")

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(pred, target)
        if self.loss_type != "pearson_mse" or self._current_epoch <= self.burnin_epochs:
            return mse
        return self.pearson_weight * self.pearson_loss(pred, target) + mse

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(xb)
            loss = self._compute_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(xb)
        return total_loss / len(loader.dataset)

    def _eval_mrpc(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        y_pred = predict_batched(self.model, X_val, self.device, self.eval_batch)
        return mean_rowwise_pearson(y_val, y_pred)

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        dataset = MultiomeDataset(X_tr, y_tr)
        loader  = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )

        history = {"epoch": [], "train_loss": [], "val_mrpc": [], "lr": []}
        best_weights = copy.deepcopy(self.model.state_dict())
        t_start = time.time()

        for epoch in range(1, self.max_epochs + 1):
            self._current_epoch = epoch
            train_loss = self._train_epoch(loader)
            val_mrpc   = self._eval_mrpc(X_val, y_val)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.scheduler_type == "cosine_warm":
                self.scheduler.step(epoch)
            else:
                self.scheduler.step(val_mrpc)

            improved = self.early_stop.step(val_mrpc, epoch)
            if improved:
                best_weights = copy.deepcopy(self.model.state_dict())

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_mrpc"].append(val_mrpc)
            history["lr"].append(current_lr)

        self.model.load_state_dict(best_weights)
        history["best_epoch"] = self.early_stop.best_epoch
        history["best_val_mrpc"] = self.early_stop.best
        return history
