import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _root(f: h5py.File) -> h5py.Group:
    keys = list(f.keys())
    if len(keys) == 1:
        return f[keys[0]]
    raise ValueError(f"Expected 1 top-level group, found: {keys}")


def inspect_h5(path: Path) -> dict:
    info = {}
    with h5py.File(path, "r") as f:
        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                info[name] = {"shape": obj.shape, "dtype": obj.dtype}
        f.visititems(_visitor)
    return info


def load_h5_as_dataframe(
    path: Path,
    rows: Optional[np.ndarray] = None,
    cols: Optional[np.ndarray] = None,
) -> pd.DataFrame:
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

    return pd.DataFrame(matrix, index=cell_ids, columns=feature_names)


def load_h5_sparse(
    path: Path,
    rows: Optional[np.ndarray] = None,
    cols: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
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
    with h5py.File(path, "r") as f:
        return _root(f)["axis1"][:].astype(str)


def get_h5_feature_names(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return _root(f)["axis0"][:].astype(str)


def load_metadata(path: Path = config.METADATA) -> pd.DataFrame:
    meta = pd.read_csv(path)
    meta["donor"] = meta["donor"].astype(int)
    meta["day"] = meta["day"].astype(int)
    meta = meta.set_index("cell_id")
    return meta


def load_evaluation_ids(path: Path = config.EVALUATION_IDS) -> pd.DataFrame:
    return pd.read_csv(path)


def load_train_data(
    inputs_path: Path = config.TRAIN_INPUTS,
    targets_path: Path = config.TRAIN_TARGETS,
    metadata_path: Path = config.METADATA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    with h5py.File(inputs_path, "r") as f:
        g = _root(f)
        cell_ids_x = g["axis1"][:].astype(str)
        atac_features = g["axis0"][:].astype(str)
        X = g["block0_values"][:]

    with h5py.File(targets_path, "r") as f:
        g = _root(f)
        cell_ids_y = g["axis1"][:].astype(str)
        rna_features = g["axis0"][:].astype(str)
        y = g["block0_values"][:]

    assert np.all(cell_ids_x == cell_ids_y), "Cell ID mismatch between inputs and targets!"

    meta = load_metadata(metadata_path)
    meta = meta.loc[cell_ids_x] 

    return X, y, atac_features, rna_features, meta


def load_test_data(
    inputs_path: Path = config.TEST_INPUTS,
    metadata_path: Path = config.METADATA,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    with h5py.File(inputs_path, "r") as f:
        g = _root(f)
        cell_ids = g["axis1"][:].astype(str)
        atac_features = g["axis0"][:].astype(str)
        X_test = g["block0_values"][:]

    meta = load_metadata(metadata_path)
    meta = meta.loc[cell_ids]

    return X_test, atac_features, meta


def sparsity_stats(matrix: np.ndarray, name: str = "matrix") -> dict:
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
    return stats


def summarize_metadata(meta: pd.DataFrame) -> None:
    for col in ["donor", "day", "cell_type"]:
        if col in meta.columns:
            counts = meta[col].value_counts().sort_index()

