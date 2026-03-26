"""
Models for the Multiome task.

RidgeMultiOutput
    Wraps sklearn Ridge for multi-output regression (105K cells × 23K genes).
    Because the input is only 129-dimensional (post-LSI), the normal equations
    are solved on a 129×129 gram matrix — extremely cheap regardless of gene count.

MLPModel
    Three-hidden-layer MLP: Linear → BatchNorm → GELU → Dropout, × 3, then output.
    Designed for 129-dim LSI+day input → 23418-gene RNA output.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class RidgeMultiOutput:
    """
    Ridge regression for multi-output prediction (ATAC-LSI → RNA).

    sklearn's Ridge natively supports multi-output regression: it fits a
    separate regularised linear model per gene but shares the same alpha.
    With 129 input features, the 129×129 gram matrix (X^T X) is trivially
    small regardless of the number of cells or genes.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength. Larger values → more shrinkage.
    fit_intercept : bool
        Whether to fit a per-gene bias term (default True).
    scale_features : bool
        StandardScale X before fitting. Recommended when LSI components
        have different variances (default True).
    """

    def __init__(
        self,
        alpha: float = 100.0,
        fit_intercept: bool = True,
        scale_features: bool = True,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.scale_features = scale_features
        self._ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        self._scaler = StandardScaler() if scale_features else None
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeMultiOutput":
        """
        Parameters
        ----------
        X : (n_cells, n_features)  — LSI + day features
        y : (n_cells, n_genes)     — log1p RNA targets
        """
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        self._ridge.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n_cells, n_features)

        Returns
        -------
        np.ndarray, (n_cells, n_genes)  — predicted log1p RNA
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._ridge.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        """Weight matrix, shape (n_genes, n_features)."""
        return self._ridge.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Bias vector, shape (n_genes,)."""
        return self._ridge.intercept_

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        print(f"  Saved RidgeMultiOutput → {path}")

    @staticmethod
    def load(path: Path) -> "RidgeMultiOutput":
        obj = joblib.load(path)
        print(f"  Loaded RidgeMultiOutput ← {path}")
        return obj


# ── MLP Model ─────────────────────────────────────────────────────────────────

class _Block(nn.Module):
    """Linear → BatchNorm1d → GELU → Dropout."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    """Residual block: Linear → BatchNorm1d → GELU → Dropout + skip connection.
    Only valid when in_dim == out_dim."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class MLPModel(nn.Module):
    """
    Multi-layer perceptron for ATAC-seq → RNA expression prediction.

    Architecture
    ------------
    Input (129) → Block(512) → [Res]Block(512) → Block(256) → Linear(23418)

    Each Block is: Linear → BatchNorm1d → GELU → Dropout
    ResBlock adds a skip connection (only on same-dimension 512→512 layer).

    Parameters
    ----------
    in_features   : Input dimension (default 129: 128 LSI + 1 day).
    hidden_dims   : Sequence of hidden layer widths.
    out_features  : Number of output genes (default 23418).
    dropout       : Dropout probability applied after each hidden block.
    use_residual  : Whether to use residual connection on equal-dimension blocks.
    """

    def __init__(
        self,
        in_features: int = 129,
        hidden_dims: Tuple[int, ...] = (512, 512, 256),
        out_features: int = config.N_RNA_GENES,
        dropout: float = 0.1,
        use_residual: bool = False,
    ):
        super().__init__()
        self.in_features  = in_features
        self.hidden_dims  = hidden_dims
        self.out_features = out_features
        self.dropout      = dropout
        self.use_residual = use_residual

        dims = [in_features] + list(hidden_dims)
        blocks = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            if use_residual and in_d == out_d:
                blocks.append(_ResBlock(in_d, dropout))
            else:
                blocks.append(_Block(in_d, out_d, dropout))
        self.hidden = nn.Sequential(*blocks)
        self.output = nn.Linear(hidden_dims[-1], out_features)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict":   self.state_dict(),
            "in_features":  self.in_features,
            "hidden_dims":  self.hidden_dims,
            "out_features": self.out_features,
            "dropout":      self.dropout,
            "use_residual": self.use_residual,
        }, path)
        print(f"  Saved MLPModel → {path}  ({self.n_params():,} params)")

    @staticmethod
    def load(path: Path) -> "MLPModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = MLPModel(
            in_features=ckpt["in_features"],
            hidden_dims=ckpt["hidden_dims"],
            out_features=ckpt["out_features"],
            dropout=ckpt["dropout"],
            use_residual=ckpt.get("use_residual", False),
        )
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded MLPModel ← {path}")
        return model
