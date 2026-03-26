import numpy as np
import pandas as pd


def pearson_per_row(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pearson r for each cell (row). Returns array of shape (n_cells,)."""
    y_true_c = y_true - y_true.mean(axis=1, keepdims=True)
    y_pred_c = y_pred - y_pred.mean(axis=1, keepdims=True)
    num = (y_true_c * y_pred_c).sum(axis=1)
    denom = np.sqrt(
        (y_true_c ** 2).sum(axis=1) * (y_pred_c ** 2).sum(axis=1)
    )
    denom = np.where(denom == 0, np.nan, denom)
    return num / denom


def mean_rowwise_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Official competition metric: mean Pearson correlation across cells.
    Constant predictions for a cell yield NaN → treated as -1 per competition rules.
    """
    r = pearson_per_row(y_true, y_pred)
    r = np.where(np.isnan(r), -1.0, r)
    return float(r.mean())


def evaluate_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    group_col: str,
) -> pd.Series:
    """Compute mean rowwise Pearson per group (e.g., cell_type or day)."""
    results = {}
    groups = meta[group_col].values
    for g in np.unique(groups):
        idx = groups == g
        results[g] = mean_rowwise_pearson(y_true[idx], y_pred[idx])
    return pd.Series(results).sort_index()
