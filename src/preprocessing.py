import hdf5plugin 
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from pathlib import Path
from typing import Optional, Tuple
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def load_h5_sparse_chunked(
    path: Path,
    chunk_size: int = 5_000,
    zero_threshold: float = 1e-9,
    verbose: bool = True,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        g = list(f.values())[0]
        cell_ids   = g["axis1"][:].astype(str)
        feat_names = g["axis0"][:].astype(str)
        n_cells, n_feats = g["block0_values"].shape

        chunks = []
        n_batches = (n_cells + chunk_size - 1) // chunk_size
        for i in range(n_batches):
            start = i * chunk_size
            end   = min(start + chunk_size, n_cells)
            batch = g["block0_values"][start:end, :]
            batch[batch < zero_threshold] = 0.0
            chunks.append(sp.csr_matrix(batch, dtype=np.float32))

    matrix = sp.vstack(chunks, format="csr")
    actual_density = matrix.nnz / (n_cells * n_feats)
    return matrix, cell_ids, feat_names

    def __init__(
        self,
        n_components: int = config.LSI_COMPONENTS,
        random_state: int = config.RANDOM_SEED,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self._svd = TruncatedSVD(n_components=n_components + 1, random_state=random_state)
        self.fitted_ = False

    def fit(self, X: sp.csr_matrix) -> "ATACPreprocessor":
        self._svd.fit(X)
        ev = self._svd.explained_variance_ratio_
        self.fitted_ = True
        return self

    def transform(self, X: sp.csr_matrix, batch_size: int = 10_000) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        n = X.shape[0]
        parts = []
        n_batches = (n + batch_size - 1) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end   = min(start + batch_size, n)
            Z_batch = self._svd.transform(X[start:end])
            Z_batch = Z_batch[:, 1:]           
            Z_batch = normalize(Z_batch, norm="l2")
            parts.append(Z_batch)
        return np.vstack(parts)

    def fit_transform(self, X: sp.csr_matrix) -> np.ndarray:
        return self.fit(X).transform(X)

    def explained_variance_ratio(self) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Not yet fitted.")
        return self._svd.explained_variance_ratio_[1:]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    def load(path: Path) -> "ATACPreprocessor":
        obj = joblib.load(path)
        return obj


class ScoredGeneFilter:
    def __init__(
        self,
        eval_ids_path: Path = config.EVALUATION_IDS,
        rna_gene_names: Optional[np.ndarray] = None,
        test_cell_ids: Optional[np.ndarray] = None,
    ):
        if rna_gene_names is None:
            with h5py.File(config.TRAIN_TARGETS, "r") as f:
                rna_gene_names = list(f.values())[0]["axis0"][:].astype(str)

        if test_cell_ids is None:
            with h5py.File(config.TEST_INPUTS, "r") as f:
                test_cell_ids = list(f.values())[0]["axis1"][:].astype(str)

        self.rna_gene_names = rna_gene_names
        self.test_cell_ids  = test_cell_ids
        self._build(eval_ids_path)

    def _build(self, eval_ids_path: Path) -> None:
        eval_ids = pd.read_csv(eval_ids_path)

        test_cell_set = set(self.test_cell_ids)
        multi_eval = eval_ids[eval_ids["cell_id"].isin(test_cell_set)].copy()

        scored_cell_set = set(multi_eval["cell_id"].unique())
        self.scored_cell_ids = np.array(
            [c for c in self.test_cell_ids if c in scored_cell_set]
        )
        cell_to_idx = {c: i for i, c in enumerate(self.test_cell_ids)}
        self.scored_cell_indices = np.array(
            [cell_to_idx[c] for c in self.scored_cell_ids]
        )

        scored_gene_set = set(multi_eval["gene_id"].unique())
        gene_to_idx = {g: i for i, g in enumerate(self.rna_gene_names)}
        self.scored_gene_ids = np.array(
            [g for g in self.rna_gene_names if g in scored_gene_set]
        )
        self.scored_gene_indices = np.array(
            [gene_to_idx[g] for g in self.scored_gene_ids]
        )

        self._eval_ids = multi_eval

    def filter_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        return y_pred[np.ix_(self.scored_cell_indices, self.scored_gene_indices)]

    def build_submission(self, y_pred_full: np.ndarray) -> pd.DataFrame:
        y_scored = self.filter_predictions(y_pred_full)
        cell_to_row = {c: i for i, c in enumerate(self.scored_cell_ids)}
        gene_to_col = {g: i for i, g in enumerate(self.scored_gene_ids)}

        predictions = self._eval_ids.copy()
        cell_rows = predictions["cell_id"].map(cell_to_row).values
        gene_cols = predictions["gene_id"].map(gene_to_col).values
        predictions["target"] = y_scored[cell_rows, gene_cols]
        return predictions[["row_id", "target"]]

    def __init__(self, day_min: float = 2.0, day_max: float = 7.0):
        self.day_min = day_min
        self.day_max = day_max

    def transform(self, days: np.ndarray) -> np.ndarray:
        normed = (days.astype(float) - self.day_min) / (self.day_max - self.day_min)
        return normed.reshape(-1, 1)


def build_features(
    X_sparse: sp.csr_matrix,
    days: np.ndarray,
    preprocessor: ATACPreprocessor,
    day_encoder: DayEncoder,
) -> np.ndarray:
    Z = preprocessor.transform(X_sparse)
    d = day_encoder.transform(days)
    return np.hstack([Z, d])


if __name__ == "__main__":
    import time
    from src.data_utils import load_metadata

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X_train_sp, train_cell_ids, atac_features = load_h5_sparse_chunked(
        config.TRAIN_INPUTS, chunk_size=5_000
    )

    t1 = time.time()
    with h5py.File(config.TRAIN_TARGETS, "r") as f:
        g = list(f.values())[0]
        rna_features = g["axis0"][:].astype(str)
        y_train = g["block0_values"][:]

    meta = load_metadata()
    train_meta = meta.loc[train_cell_ids]
    days_train = train_meta["day"].values

    t1 = time.time()
    preprocessor = ATACPreprocessor(n_components=config.LSI_COMPONENTS)
    preprocessor.fit(X_train_sp)

    day_encoder = DayEncoder()
    X_train_feat = build_features(X_train_sp, days_train, preprocessor, day_encoder)

    gene_filter = ScoredGeneFilter(rna_gene_names=rna_features)

    t1 = time.time()
    X_test_sp, test_cell_ids, _ = load_h5_sparse_chunked(
        config.TEST_INPUTS, chunk_size=5_000
    )

    test_meta = meta.loc[test_cell_ids]
    days_test  = test_meta["day"].values
    X_test_feat = build_features(X_test_sp, days_test, preprocessor, day_encoder)

    PRED = config.PREDICTIONS_DIR
    PRED.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor.save(config.MODELS_DIR / "atac_preprocessor.joblib")
    joblib.dump(gene_filter, config.MODELS_DIR / "gene_filter.joblib")

    np.save(PRED / "X_train_feat.npy", X_train_feat)
    np.save(PRED / "X_test_feat.npy",  X_test_feat)

    np.save(PRED / "train_cell_ids.npy", train_cell_ids)
    np.save(PRED / "test_cell_ids.npy",  test_cell_ids)
    np.save(PRED / "rna_features.npy",   rna_features)

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    ev = preprocessor.explained_variance_ratio 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(1, len(ev) + 1), ev, color="steelblue", width=1.0)
    axes[0].set_xlabel("LSI Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot — LSI Components (component 0 dropped)")

    axes[1].plot(range(1, len(ev) + 1), np.cumsum(ev), color="steelblue", marker=".")
    axes[1].axhline(np.cumsum(ev)[-1], color="grey", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Explained Variance")

    plt.tight_layout()
    scree_path = config.FIGURES_DIR / "02_lsi_scree_plot.png"
    plt.savefig(scree_path, dpi=150)
    plt.close()

    Z_train = X_train_feat[:, :2]  

    rng = np.random.RandomState(42)
    n_plot = min(20_000, len(Z_train))
    idx = rng.choice(len(Z_train), n_plot, replace=False)
    Z_plot = Z_train[idx]
    meta_plot = train_meta.iloc[idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scatter_kw = dict(s=2, alpha=0.5, rasterized=True)

    for ax, col, title in zip(
        axes,
        ["donor", "day", "cell_type"],
        ["Donor", "Day", "Cell Type"],
    ):
        cats = meta_plot[col].astype(str)
        unique = sorted(cats.unique())
        palette = cm.tab10.colors
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
        colors = [color_map[c] for c in cats]
        ax.scatter(Z_plot[:, 0], Z_plot[:, 1], c=colors, **scatter_kw)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map[c], markersize=6, label=c)
            for c in unique
        ]
        ax.legend(handles=handles, fontsize=7, markerscale=1.5,
                  loc="upper right", framealpha=0.7)
        ax.set_xlabel("LSI-1")
        ax.set_ylabel("LSI-2")
        ax.set_title(f"LSI Embedding — {title}")

    plt.tight_layout()
    embed_path = config.FIGURES_DIR / "02_lsi_embedding.png"
    plt.savefig(embed_path, dpi=150)
    plt.close()