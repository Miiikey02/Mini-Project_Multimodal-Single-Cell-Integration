"""
Data loading utilities for the Multiome task.

All h5 files from the competition use the format:
    /<filename>/axis0         — feature/gene names (columns)
    /<filename>/axis1         — cell IDs (rows)
    /<filename>/block0_values — dense float32 matrix

The datasets are nested under a root group named after the file stem
(e.g., train_multi_inputs/axis1). _root() navigates into that group.
"""

import h5py
import hdf5plugin  # registers Blosc and other filters used by competition h5 files
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Low-level H5 helpers ───────────────────────────────────────────────────────

def _root(f: h5py.File) -> h5py.Group:
    """Return the single top-level group inside an h5 file."""
    keys = list(f.keys())
    if len(keys) == 1:
        return f[keys[0]]
    raise ValueError(f"Expected 1 top-level group, found: {keys}")


def inspect_h5(path: Path) -> dict:
    """Print structure and return basic stats for an h5 file."""
    info = {}
    with h5py.File(path, "r") as f:
        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                info[name] = {"shape": obj.shape, "dtype": obj.dtype}
                print(f"  /{name}: shape={obj.shape}, dtype={obj.dtype}")
        print(f"\n=== {path.name} ===")
        f.visititems(_visitor)
    return info


def load_h5_as_dataframe(
    path: Path,
    rows: Optional[np.ndarray] = None,
    cols: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Load an h5 file into a dense DataFrame.

    Parameters
    ----------
    path : Path to the h5 file.
    rows : Optional integer row indices to load (subset of cells).
    cols : Optional integer column indices to load (subset of features/genes).

    Returns
    -------
    pd.DataFrame with cell_ids as index and feature names as columns.
    """
    with h5py.File(path, "r") as f:
        g = _root(f)
        cell_ids = g["axis1"][:].astype(str)
        feature_names = g["axis0"][:].astype(str)
        matrix = g["block0_values"][:]  # shape: (n_cells, n_features)

    if rows is not None:
        cell_ids = cell_ids[rows]
        matrix = matrix[rows, :]
    if cols is not None:
        feature_names = feature_names[cols]
        matrix = matrix[:, cols]

    return pd.DataFrame(matrix, index=cell_ids, columns=feature_names)


def load_h5_sparse(
    path: Path,
    rows: Optional[np.ndarray] = None,
    cols: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Load an h5 file as a sparse CSR matrix.
    Note: the h5 file stores a dense block, so the full matrix is loaded into
    memory before conversion. Useful when downstream code needs a sparse format,
    but does not reduce peak RAM during loading.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix, shape (n_cells, n_features)
    cell_ids : np.ndarray of cell ID strings
    feature_names : np.ndarray of feature name strings
    """
    with h5py.File(path, "r") as f:
        g = _root(f)
        cell_ids = g["axis1"][:].astype(str)
        feature_names = g["axis0"][:].astype(str)
        matrix = g["block0_values"][:]

    if rows is not None:
        cell_ids = cell_ids[rows]
        matrix = matrix[rows, :]
    if cols is not None:
        feature_names = feature_names[cols]
        matrix = matrix[:, cols]

    return sp.csr_matrix(matrix), cell_ids, feature_names


def get_h5_cell_ids(path: Path) -> np.ndarray:
    """Read only cell IDs from an h5 file (fast, no matrix load)."""
    with h5py.File(path, "r") as f:
        return _root(f)["axis1"][:].astype(str)


def get_h5_feature_names(path: Path) -> np.ndarray:
    """Read only feature/gene names from an h5 file."""
    with h5py.File(path, "r") as f:
        return _root(f)["axis0"][:].astype(str)


# ── Metadata ───────────────────────────────────────────────────────────────────

def load_metadata(path: Path = config.METADATA) -> pd.DataFrame:
    """
    Load metadata.csv. Ensures correct dtypes.

    Columns: cell_id, donor, day, technology, cell_type
    """
    meta = pd.read_csv(path)
    meta["donor"] = meta["donor"].astype(int)
    meta["day"] = meta["day"].astype(int)
    meta = meta.set_index("cell_id")
    return meta


def load_evaluation_ids(path: Path = config.EVALUATION_IDS) -> pd.DataFrame:
    """
    Load evaluation_ids.csv.
    Contains the (cell_id, gene) pairs that are actually scored.
    """
    return pd.read_csv(path)


# ── Convenience loaders ────────────────────────────────────────────────────────

def load_train_data(
    inputs_path: Path = config.TRAIN_INPUTS,
    targets_path: Path = config.TRAIN_TARGETS,
    metadata_path: Path = config.METADATA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load training inputs (ATAC), targets (RNA), feature names, and metadata.

    Returns
    -------
    X : np.ndarray, shape (n_train_cells, n_peaks) — TF-IDF ATAC values
    y : np.ndarray, shape (n_train_cells, n_genes) — log1p RNA values
    atac_features : np.ndarray of peak names
    rna_features : np.ndarray of gene names
    meta : pd.DataFrame indexed by cell_id (donor, day, cell_type)
    """
    print("Loading training inputs (ATAC)...")
    with h5py.File(inputs_path, "r") as f:
        g = _root(f)
        cell_ids_x = g["axis1"][:].astype(str)
        atac_features = g["axis0"][:].astype(str)
        X = g["block0_values"][:]
    print(f"  ATAC: {X.shape}")

    print("Loading training targets (RNA)...")
    with h5py.File(targets_path, "r") as f:
        g = _root(f)
        cell_ids_y = g["axis1"][:].astype(str)
        rna_features = g["axis0"][:].astype(str)
        y = g["block0_values"][:]
    print(f"  RNA:  {y.shape}")

    assert np.all(cell_ids_x == cell_ids_y), "Cell ID mismatch between inputs and targets!"

    meta = load_metadata(metadata_path)
    meta = meta.loc[cell_ids_x]  # align to matrix order

    return X, y, atac_features, rna_features, meta


def load_test_data(
    inputs_path: Path = config.TEST_INPUTS,
    metadata_path: Path = config.METADATA,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load test inputs (ATAC) and metadata.

    Returns
    -------
    X_test : np.ndarray, shape (n_test_cells, n_peaks)
    atac_features : np.ndarray of peak names
    meta : pd.DataFrame indexed by cell_id
    """
    print("Loading test inputs (ATAC)...")
    with h5py.File(inputs_path, "r") as f:
        g = _root(f)
        cell_ids = g["axis1"][:].astype(str)
        atac_features = g["axis0"][:].astype(str)
        X_test = g["block0_values"][:]
    print(f"  ATAC: {X_test.shape}")

    meta = load_metadata(metadata_path)
    meta = meta.loc[cell_ids]

    return X_test, atac_features, meta


# ── Sparsity and summary stats ─────────────────────────────────────────────────

def sparsity_stats(matrix: np.ndarray, name: str = "matrix") -> dict:
    """Compute and print sparsity statistics."""
    n_total = matrix.size
    n_zero = np.sum(matrix == 0)
    n_nonzero = n_total - n_zero
    sparsity = n_zero / n_total

    stats = {
        "name": name,
        "shape": matrix.shape,
        "n_total": n_total,
        "n_nonzero": int(n_nonzero),
        "sparsity": float(sparsity),
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "mean": float(matrix.mean()),
        "median": float(np.median(matrix)),
    }

    print(f"\n--- {name} ---")
    print(f"  Shape:    {stats['shape']}")
    print(f"  Sparsity: {stats['sparsity']:.2%}  ({n_nonzero:,} non-zeros)")
    print(f"  Range:    [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Mean:     {stats['mean']:.4f}  |  Median: {stats['median']:.4f}")

    return stats


def summarize_metadata(meta: pd.DataFrame) -> None:
    """Print cell counts broken down by donor, day, and cell type."""
    print("\n=== Metadata Summary ===")
    print(f"Total cells: {len(meta):,}\n")

    for col in ["donor", "day", "cell_type"]:
        if col in meta.columns:
            counts = meta[col].value_counts().sort_index()
            print(f"By {col}:")
            for k, v in counts.items():
                print(f"  {k}: {v:,}")
            print()

    # Cross-tab: day × cell_type
    if "day" in meta.columns and "cell_type" in meta.columns:
        print("Cell counts by day × cell_type:")
        print(pd.crosstab(meta["day"], meta["cell_type"]))
