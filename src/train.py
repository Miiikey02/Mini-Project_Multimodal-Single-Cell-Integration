"""
Training infrastructure for the MLP model.

Components
----------
MultiomeDataset  — torch Dataset wrapping numpy X/y arrays
EarlyStopping    — monitors a metric (higher=better) and signals when to stop
PearsonLoss      — row-wise mean negative Pearson correlation (differentiable)
Trainer          — manages the full training loop, LR scheduling, and checkpointing
predict_batched  — memory-safe batched inference that returns numpy predictions
"""

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


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiomeDataset(Dataset):
    """
    Wraps numpy feature matrix X and target matrix y as a torch Dataset.

    Keeps data as numpy arrays on CPU; __getitem__ returns float32 tensors.
    This avoids materialising an entire float32 copy of y (9.4 GB) as a
    torch tensor up-front — torch tensors backed by numpy share memory when
    the dtype matches (float32), but we convert explicitly to be safe.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when a monitored metric stops improving.

    Parameters
    ----------
    patience  : Epochs to wait after last improvement before stopping.
    min_delta : Minimum change to qualify as an improvement.
    mode      : 'max' (higher is better, e.g. MRPC) or 'min' (lower is better).
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-5, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = float("-inf") if mode == "max" else float("inf")
        self.counter    = 0
        self.best_epoch = 0
        self.stop       = False

    def step(self, value: float, epoch: int) -> bool:
        """
        Call after each validation evaluation.
        Returns True if the metric improved (new best), False otherwise.
        Sets self.stop = True when patience is exhausted.
        """
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


# ── Pearson Loss ──────────────────────────────────────────────────────────────

class PearsonLoss(nn.Module):
    """
    Differentiable mean negative row-wise Pearson correlation.

    Computes Pearson r between pred[i, :] and target[i, :] for each row i,
    then returns the negative mean — minimising this loss maximises MRPC,
    the competition metric, directly.

    A small epsilon (1e-8) is added to the denominator to prevent division
    by zero for near-constant rows during early training.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_c   = pred   - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)
        num   = (pred_c * target_c).sum(dim=1)
        denom = pred_c.norm(dim=1) * target_c.norm(dim=1) + self.eps
        return -(num / denom).mean()


# ── Batched Inference ─────────────────────────────────────────────────────────

def predict_batched(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Run batched inference; returns predictions as a numpy float32 array.

    Running all 55K test cells through the model at once would require
    55K × 23K × 4 bytes ≈ 5 GB on the device. Batching keeps device memory
    at batch_size × 23K × 4 bytes ≈ 48 MB per batch.
    """
    model.eval()
    parts = []
    X_f32 = X.astype(np.float32)
    with torch.no_grad():
        for start in range(0, len(X_f32), batch_size):
            xb = torch.from_numpy(X_f32[start: start + batch_size]).to(device)
            parts.append(model(xb).cpu().numpy())
    return np.vstack(parts)


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training loop for MLPModel.

    Training objective : MSE loss, or combined Pearson+MSE (pearson_mse mode).
    Validation metric  : Mean Rowwise Pearson Correlation (MRPC) — the competition metric.
    Optimiser          : AdamW with weight decay.
    LR schedule        : ReduceLROnPlateau monitoring validation MRPC.
    Early stopping     : Restores best weights when MRPC stops improving.

    Parameters
    ----------
    model          : MLPModel instance (on CPU; moved to device internally).
    device         : torch.device to train on (MPS, CUDA, or CPU).
    lr             : Initial learning rate.
    weight_decay   : L2 regularisation in AdamW.
    batch_size     : Training mini-batch size.
    max_epochs     : Hard cap on training epochs.
    patience       : Early stopping patience (epochs without MRPC improvement).
    scheduler_type : 'plateau' (ReduceLROnPlateau) or 'cosine_warm'
                     (CosineAnnealingWarmRestarts).
    lr_patience    : Plateau scheduler patience (only for 'plateau').
    lr_factor      : Plateau LR reduction factor (only for 'plateau').
    cosine_t0      : Restart period in epochs (only for 'cosine_warm').
    min_lr         : Minimum LR.
    eval_batch     : Batch size for validation inference.
    loss_type      : 'mse' (standard) or 'pearson_mse' (Pearson + MSE combined).
    burnin_epochs  : Epochs to train on MSE only before enabling Pearson loss.
    pearson_weight : Weight of Pearson loss in combined loss (MSE weight = 1.0).
    """

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
        self.criterion    = self.mse_loss   # kept for backward compat
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

    # ── single epoch ──────────────────────────────────────────────────────────

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined Pearson+MSE loss with burn-in, or plain MSE."""
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

    # ── full fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Train the model with early stopping on validation MRPC.

        Parameters
        ----------
        X_tr, y_tr : Training features and targets (numpy float32).
        X_val, y_val : Validation features and targets (numpy float32).
        verbose    : Print one line per epoch.

        Returns
        -------
        history : dict with lists 'train_loss', 'val_mrpc', 'lr', 'epoch'.
        """
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

            if verbose:
                marker = " ★" if improved else ""
                elapsed = time.time() - t_start
                print(
                    f"  Epoch {epoch:3d}/{self.max_epochs}  "
                    f"loss={train_loss:.5f}  val_MRPC={val_mrpc:.5f}  "
                    f"lr={current_lr:.2e}  ({elapsed:.0f}s){marker}"
                )

            if self.early_stop.stop:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best epoch {self.early_stop.best_epoch}, "
                      f"best MRPC={self.early_stop.best:.5f})")
                break

        # Restore best weights
        self.model.load_state_dict(best_weights)
        history["best_epoch"] = self.early_stop.best_epoch
        history["best_val_mrpc"] = self.early_stop.best
        return history
