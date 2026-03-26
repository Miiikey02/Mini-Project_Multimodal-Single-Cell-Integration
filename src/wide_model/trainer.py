"""
WideTrainer — training loop for WideEncoderDecoder with SVD target space.

Key differences from the base Trainer in src/train.py
------------------------------------------------------
1. OneCycleLR scheduler (steps per batch, not per epoch)
   - Ramps LR up then anneals in one smooth cycle
   - Better suited to large models than CosineAnnealingWarmRestarts
   - Requires knowing steps_per_epoch at construction → built lazily in fit()

2. SVD reconstruction for MRPC evaluation
   - Model predicts in SVD latent space (512 dims)
   - MRPC is computed after reconstructing to gene space: z @ V^T
   - Validation targets passed to fit() are raw genes; SVD targets are derived

3. CombinedLoss (Pearson + MSE with burn-in)
   - Same burn-in strategy as v3/v4 but encapsulated in CombinedLoss class

Run on local Mac  : Day-7 fold only (~30-60 min) or quick smoke test
Run on server     : Full LODO CV (all 4 folds) + Optuna HPO
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluate import mean_rowwise_pearson
from src.wide_model.losses import CombinedLoss


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiomeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = float("-inf")
        self.counter    = 0
        self.best_epoch = 0
        self.stop       = False

    def step(self, value: float, epoch: int) -> bool:
        if value > self.best + self.min_delta:
            self.best       = value
            self.best_epoch = epoch
            self.counter    = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return False


# ── Batched Inference ─────────────────────────────────────────────────────────

def predict_batched(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Batched inference; returns float32 numpy array."""
    model.eval()
    parts = []
    X_f32 = X.astype(np.float32)
    with torch.no_grad():
        for start in range(0, len(X_f32), batch_size):
            xb = torch.from_numpy(X_f32[start: start + batch_size]).to(device)
            parts.append(model(xb).cpu().numpy())
    return np.vstack(parts)


def svd_predict_mrpc(
    model: nn.Module,
    X_val: np.ndarray,
    y_val_genes: np.ndarray,
    svd_obj,
    device: torch.device,
    batch_size: int = 512,
):
    """Predict in SVD space, reconstruct to gene space, compute MRPC."""
    z_pred  = predict_batched(model, X_val, device, batch_size)
    y_pred  = np.clip(z_pred @ svd_obj.components_, 0, None)
    return mean_rowwise_pearson(y_val_genes, y_pred), y_pred


# ── Wide Trainer ──────────────────────────────────────────────────────────────

class WideTrainer:
    """
    Training loop for WideEncoderDecoder with SVD target space.

    Parameters
    ----------
    model           : WideEncoderDecoder instance (moved to device internally).
    device          : torch.device (MPS / CUDA / CPU).
    svd_obj         : Fitted TruncatedSVD object for target reconstruction.
    y_val_genes     : Validation targets in gene space (for MRPC evaluation).
    lr              : Maximum learning rate for OneCycleLR.
    weight_decay    : AdamW weight decay.
    batch_size      : Training batch size.
    max_epochs      : Hard epoch cap.
    patience        : Early stopping patience.
    pearson_weight  : Weight on Pearson loss (MSE weight fixed at 1.0).
    burnin_epochs   : Epochs to use MSE-only before engaging Pearson loss.
    eval_batch      : Batch size for validation inference.
    pct_start       : OneCycleLR warm-up fraction (default 0.3 = 30% of epochs).
    """

    def __init__(
        self,
        model,
        device: torch.device,
        svd_obj,
        y_val_genes: np.ndarray,
        lr: float             = 1e-3,
        weight_decay: float   = 1e-4,
        batch_size: int       = 512,
        max_epochs: int       = 100,
        patience: int         = 20,
        pearson_weight: float = 1.0,
        burnin_epochs: int    = 10,
        eval_batch: int       = 512,
        pct_start: float      = 0.3,
    ):
        self.model        = model.to(device)
        self.device       = device
        self.svd_obj      = svd_obj
        self.y_val_genes  = y_val_genes
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.eval_batch   = eval_batch
        self.pct_start    = pct_start

        self.loss_fn  = CombinedLoss(
            pearson_weight=pearson_weight,
            mse_weight=1.0,
            burnin_epochs=burnin_epochs,
        )
        self.optimizer  = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # OneCycleLR is created lazily in fit() once we know steps_per_epoch
        self.scheduler  = None
        self._lr        = lr
        self.early_stop = EarlyStopping(patience=patience)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()          # OneCycleLR: step per batch
            total_loss += loss.item() * len(xb)
        return total_loss / len(loader.dataset)

    def _eval_mrpc(self, X_val: np.ndarray, _y_val_svd) -> float:
        """Reconstruct SVD predictions to gene space, compute MRPC."""
        mrpc, _ = svd_predict_mrpc(
            self.model, X_val, self.y_val_genes, self.svd_obj,
            self.device, self.eval_batch,
        )
        return mrpc

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr_svd: np.ndarray,
        X_val: np.ndarray,
        y_val_svd: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Train with OneCycleLR and early stopping on validation MRPC.

        Parameters
        ----------
        X_tr, y_tr_svd   : Training features and SVD-compressed targets.
        X_val, y_val_svd : Validation features and SVD-compressed targets.
                           (val MRPC is computed in gene space via SVD reconstruction)
        verbose          : Print one line per epoch.

        Returns
        -------
        history : dict with 'epoch', 'train_loss', 'val_mrpc', 'lr',
                  'best_epoch', 'best_val_mrpc'.
        """
        dataset = MultiomeDataset(X_tr, y_tr_svd)
        loader  = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )

        # Build OneCycleLR now that we know steps_per_epoch
        steps_per_epoch = len(loader)
        self.scheduler  = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr          = self._lr,
            steps_per_epoch = steps_per_epoch,
            epochs          = self.max_epochs,
            pct_start       = self.pct_start,
        )

        history      = {"epoch": [], "train_loss": [], "val_mrpc": [], "lr": []}
        best_weights = copy.deepcopy(self.model.state_dict())
        t_start      = time.time()

        for epoch in range(1, self.max_epochs + 1):
            self.loss_fn.set_epoch(epoch)
            train_loss = self._train_epoch(loader)
            val_mrpc   = self._eval_mrpc(X_val, y_val_svd)
            current_lr = self.optimizer.param_groups[0]["lr"]

            improved = self.early_stop.step(val_mrpc, epoch)
            if improved:
                best_weights = copy.deepcopy(self.model.state_dict())

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_mrpc"].append(val_mrpc)
            history["lr"].append(current_lr)

            if verbose:
                marker  = " ★" if improved else ""
                elapsed = time.time() - t_start
                print(
                    f"  Epoch {epoch:3d}/{self.max_epochs}  "
                    f"loss={train_loss:.5f}  val_MRPC={val_mrpc:.5f}  "
                    f"lr={current_lr:.2e}  ({elapsed:.0f}s){marker}"
                )

            if self.early_stop.stop:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(best epoch {self.early_stop.best_epoch}, "
                    f"best MRPC={self.early_stop.best:.5f})"
                )
                break

        self.model.load_state_dict(best_weights)
        history["best_epoch"]    = self.early_stop.best_epoch
        history["best_val_mrpc"] = self.early_stop.best
        return history
