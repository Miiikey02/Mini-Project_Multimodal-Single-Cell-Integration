"""
Preprocessing pipeline for the Multiome task.

Three main components:
  1. ATACPreprocessor  — LSI (TruncatedSVD) on already-TF-IDF-transformed ATAC peaks
  2. ScoredGeneFilter  — identifies which cells/genes are in the evaluation set
  3. DayEncoder        — normalises the day metadata column as an auxiliary feature

Memory strategy
---------------
The training ATAC matrix (105K × 228K float32) is ~97 GB when dense, which
exceeds typical RAM. However, it is 97.5% sparse: only ~2.5% of peak × cell
entries are non-zero. Loading it as a scipy CSR sparse matrix costs only ~4–5 GB.

All h5 reads therefore go through load_h5_sparse_chunked(), which reads the
file in row batches, thresholds near-zero values, converts each batch to CSR,
and stacks them. sklearn's TruncatedSVD accepts sparse input natively.

Transform is also done in batches so even test-set inference is memory-safe.
"""

import hdf5plugin  # must be imported before h5py to register Blosc codec
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


# ── Sparse H5 loader ──────────────────────────────────────────────────────────

def load_h5_sparse_chunked(
    path: Path,
    chunk_size: int = 5_000,
    zero_threshold: float = 1e-9,
    verbose: bool = True,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Load an h5 matrix as a sparse CSR matrix by reading in row batches.

    Rather than materialising the full dense matrix (~97 GB for training ATAC),
    each chunk of `chunk_size` rows is read, values below `zero_threshold` are
    zeroed out, and the chunk is immediately converted to CSR before the next
    chunk is read. This keeps peak RAM at roughly:
        chunk_size × n_features × 4 bytes  (one dense chunk)
      + cumulative sparse data              (grows as chunks are appended)

    Parameters
    ----------
    path          : Path to the h5 file.
    chunk_size    : Rows per batch. 5000 rows × 228K cols × 4 bytes ≈ 4.3 GB.
    zero_threshold: Values below this are treated as zero before sparsifying.
    verbose       : Print progress.

    Returns
    -------
    matrix     : scipy.sparse.csr_matrix, shape (n_cells, n_features)
    cell_ids   : np.ndarray of cell ID strings
    feat_names : np.ndarray of feature name strings
    """
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
            # threshold near-zeros before converting to sparse
            batch[batch < zero_threshold] = 0.0
            chunks.append(sp.csr_matrix(batch, dtype=np.float32))
            if verbose:
                print(f"\r    chunk {i+1}/{n_batches} ({end:,}/{n_cells:,} cells)", end="", flush=True)

    if verbose:
        print()

    matrix = sp.vstack(chunks, format="csr")
    actual_density = matrix.nnz / (n_cells * n_feats)
    if verbose:
        print(f"    Sparse matrix: {matrix.shape}, density={actual_density:.3%}, "
              f"nnz={matrix.nnz:,}, size≈{matrix.data.nbytes / 1e9:.2f} GB")
    return matrix, cell_ids, feat_names


# ── 1. ATAC Preprocessor (LSI) ────────────────────────────────────────────────

class ATACPreprocessor:
    """
    Reduces ATAC-seq peaks to a dense LSI embedding via TruncatedSVD.

    The input matrix is already TF-IDF transformed by the competition.
    Applying TruncatedSVD directly yields LSI (Latent Semantic Indexing),
    the standard dimensionality reduction for chromatin accessibility data.

    Component 0 is always dropped: it captures global library size
    (sequencing depth) rather than cell-type-specific chromatin structure.
    This is standard practice in single-cell ATAC analysis.

    After SVD, each cell's embedding is L2-normalised so that cosine
    similarity equals dot product, which benefits downstream linear models.

    Parameters
    ----------
    n_components : int
        Number of LSI dimensions to retain (after dropping component 0).
        Default: config.LSI_COMPONENTS (128). Sweep [64, 128, 256] in tuning.
    random_state : int
        Seed for reproducibility of the randomised SVD solver.
    """

    def __init__(
        self,
        n_components: int = config.LSI_COMPONENTS,
        random_state: int = config.RANDOM_SEED,
    ):
        self.n_components = n_components
        self.random_state = random_state
        # +1 because component 0 will be dropped after fitting
        self._svd = TruncatedSVD(n_components=n_components + 1, random_state=random_state)
        self.fitted_ = False

    def fit(self, X: sp.csr_matrix) -> "ATACPreprocessor":
        """
        Fit SVD on the training ATAC sparse matrix.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape (n_cells, n_peaks)
        """
        print(f"  Fitting TruncatedSVD on sparse matrix {X.shape} "
              f"→ {self.n_components} components ...")
        self._svd.fit(X)
        ev = self._svd.explained_variance_ratio_
        print(f"  Component 0 variance (dropped): {ev[0]:.4f}")
        print(f"  Retained components 1–{self.n_components}: "
              f"cumulative variance = {ev[1:].sum():.4f}")
        self.fitted_ = True
        return self

    def transform(self, X: sp.csr_matrix, batch_size: int = 10_000) -> np.ndarray:
        """
        Project ATAC matrix into LSI space in batches (memory-safe).

        Parameters
        ----------
        X          : scipy.sparse.csr_matrix, shape (n_cells, n_peaks)
        batch_size : Rows per transform batch.

        Returns
        -------
        np.ndarray, shape (n_cells, n_components), L2-normalised
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        n = X.shape[0]
        parts = []
        n_batches = (n + batch_size - 1) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end   = min(start + batch_size, n)
            Z_batch = self._svd.transform(X[start:end])
            Z_batch = Z_batch[:, 1:]             # drop component 0
            Z_batch = normalize(Z_batch, norm="l2")
            parts.append(Z_batch)
            print(f"\r    transform batch {i+1}/{n_batches}", end="", flush=True)
        print()
        return np.vstack(parts)

    def fit_transform(self, X: sp.csr_matrix) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Variance ratio for each retained component (component 0 excluded)."""
        if not self.fitted_:
            raise RuntimeError("Not yet fitted.")
        return self._svd.explained_variance_ratio_[1:]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        print(f"  Saved ATACPreprocessor → {path}")

    @staticmethod
    def load(path: Path) -> "ATACPreprocessor":
        obj = joblib.load(path)
        print(f"  Loaded ATACPreprocessor ← {path}")
        return obj


# ── 2. Scored Gene / Cell Filter ──────────────────────────────────────────────

class ScoredGeneFilter:
    """
    Identifies which test cells and RNA genes are in the evaluation set.

    The competition only scores a subset of test cells (~30%) across all
    RNA genes. This class pre-computes:
      - scored_cell_ids     : ordered test cell IDs that will be evaluated
      - scored_cell_indices : their integer positions in the test matrix
      - scored_gene_ids     : gene IDs present in evaluation_ids (all 23K)
      - scored_gene_indices : their positions in the RNA feature array

    Parameters
    ----------
    eval_ids_path  : Path to evaluation_ids.csv
    rna_gene_names : np.ndarray of gene ID strings (RNA axis0)
    test_cell_ids  : np.ndarray of test cell ID strings (test axis1)
    """

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
        print("  Loading evaluation_ids.csv ...")
        eval_ids = pd.read_csv(eval_ids_path)

        # Keep only Multiome rows (cell_ids in test ATAC file)
        test_cell_set = set(self.test_cell_ids)
        multi_eval = eval_ids[eval_ids["cell_id"].isin(test_cell_set)].copy()
        print(f"  Multiome eval pairs: {len(multi_eval):,}  "
              f"({multi_eval['cell_id'].nunique():,} cells × "
              f"{multi_eval['gene_id'].nunique():,} genes)")

        # Scored cells — preserve order from test file
        scored_cell_set = set(multi_eval["cell_id"].unique())
        self.scored_cell_ids = np.array(
            [c for c in self.test_cell_ids if c in scored_cell_set]
        )
        cell_to_idx = {c: i for i, c in enumerate(self.test_cell_ids)}
        self.scored_cell_indices = np.array(
            [cell_to_idx[c] for c in self.scored_cell_ids]
        )

        # Scored genes — preserve RNA matrix column order
        scored_gene_set = set(multi_eval["gene_id"].unique())
        gene_to_idx = {g: i for i, g in enumerate(self.rna_gene_names)}
        self.scored_gene_ids = np.array(
            [g for g in self.rna_gene_names if g in scored_gene_set]
        )
        self.scored_gene_indices = np.array(
            [gene_to_idx[g] for g in self.scored_gene_ids]
        )

        self._eval_ids = multi_eval
        print(f"  Scored cells: {len(self.scored_cell_indices):,} / {len(self.test_cell_ids):,} test cells")
        print(f"  Scored genes: {len(self.scored_gene_indices):,} / {len(self.rna_gene_names):,} genes")

    def filter_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Extract scored (cell, gene) subset from a full prediction matrix.

        Parameters
        ----------
        y_pred : np.ndarray, shape (n_test_cells, n_genes)

        Returns
        -------
        np.ndarray, shape (n_scored_cells, n_scored_genes)
        """
        return y_pred[np.ix_(self.scored_cell_indices, self.scored_gene_indices)]

    def build_submission(self, y_pred_full: np.ndarray) -> pd.DataFrame:
        """
        Build a submission DataFrame in competition format.

        Parameters
        ----------
        y_pred_full : np.ndarray, shape (n_test_cells, n_genes)
            Aligned to all test cells and all RNA genes in h5 order.

        Returns
        -------
        pd.DataFrame with columns [row_id, target]
        """
        y_scored = self.filter_predictions(y_pred_full)
        cell_to_row = {c: i for i, c in enumerate(self.scored_cell_ids)}
        gene_to_col = {g: i for i, g in enumerate(self.scored_gene_ids)}

        predictions = self._eval_ids.copy()
        cell_rows = predictions["cell_id"].map(cell_to_row).values
        gene_cols = predictions["gene_id"].map(gene_to_col).values
        predictions["target"] = y_scored[cell_rows, gene_cols]
        return predictions[["row_id", "target"]]


# ── 3. Day Feature Encoder ─────────────────────────────────────────────────────

class DayEncoder:
    """
    Encodes the 'day' metadata column as a normalised scalar feature.

    Normalised to [0, 1] over the training range (days 2–7).
    Day 10 maps to 1.6, which is deliberately out-of-range — it signals
    to the model that day 10 is beyond the training distribution,
    encouraging extrapolation rather than interpolation.
    """

    def __init__(self, day_min: float = 2.0, day_max: float = 7.0):
        self.day_min = day_min
        self.day_max = day_max

    def transform(self, days: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        days : np.ndarray of int/float, shape (n_cells,)

        Returns
        -------
        np.ndarray, shape (n_cells, 1)
        """
        normed = (days.astype(float) - self.day_min) / (self.day_max - self.day_min)
        return normed.reshape(-1, 1)


# ── 4. Combined feature builder ───────────────────────────────────────────────

def build_features(
    X_sparse: sp.csr_matrix,
    days: np.ndarray,
    preprocessor: ATACPreprocessor,
    day_encoder: DayEncoder,
) -> np.ndarray:
    """
    Project ATAC → LSI, append normalised day, return combined feature matrix.

    Parameters
    ----------
    X_sparse    : sparse ATAC matrix, shape (n_cells, n_peaks)
    days        : integer day values, shape (n_cells,)
    preprocessor: fitted ATACPreprocessor
    day_encoder : DayEncoder

    Returns
    -------
    np.ndarray, shape (n_cells, n_components + 1)
    """
    Z = preprocessor.transform(X_sparse)
    d = day_encoder.transform(days)
    return np.hstack([Z, d])


# ── 5. Run as script ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    from src.data_utils import load_metadata

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Load training ATAC as sparse ──────────────────────────────────────────
    print("\n=== Loading training ATAC (sparse) ===")
    X_train_sp, train_cell_ids, atac_features = load_h5_sparse_chunked(
        config.TRAIN_INPUTS, chunk_size=5_000
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Load training RNA (dense, but only 23K cols × 105K rows ≈ 9 GB) ──────
    print("\n=== Loading training RNA ===")
    t1 = time.time()
    with h5py.File(config.TRAIN_TARGETS, "r") as f:
        g = list(f.values())[0]
        rna_features = g["axis0"][:].astype(str)
        y_train = g["block0_values"][:]
    print(f"  RNA: {y_train.shape}  ({time.time()-t1:.1f}s)")

    # ── Load metadata ─────────────────────────────────────────────────────────
    meta = load_metadata()
    train_meta = meta.loc[train_cell_ids]
    days_train = train_meta["day"].values

    # ── Fit LSI ───────────────────────────────────────────────────────────────
    print("\n=== Fitting LSI ===")
    t1 = time.time()
    preprocessor = ATACPreprocessor(n_components=config.LSI_COMPONENTS)
    preprocessor.fit(X_train_sp)
    print(f"  Fit done in {time.time()-t1:.1f}s")

    # ── Build training features ───────────────────────────────────────────────
    print("\n=== Building training features ===")
    day_encoder = DayEncoder()
    X_train_feat = build_features(X_train_sp, days_train, preprocessor, day_encoder)
    print(f"  Train features: {X_train_feat.shape}")

    # ── Scored gene filter ────────────────────────────────────────────────────
    print("\n=== Building scored gene filter ===")
    gene_filter = ScoredGeneFilter(rna_gene_names=rna_features)

    # ── Load and transform test data ──────────────────────────────────────────
    print("\n=== Loading test ATAC (sparse) ===")
    t1 = time.time()
    X_test_sp, test_cell_ids, _ = load_h5_sparse_chunked(
        config.TEST_INPUTS, chunk_size=5_000
    )
    print(f"  Done in {time.time()-t1:.1f}s")

    test_meta = meta.loc[test_cell_ids]
    days_test  = test_meta["day"].values
    X_test_feat = build_features(X_test_sp, days_test, preprocessor, day_encoder)
    print(f"  Test features: {X_test_feat.shape}")

    # ── Save everything ────────────────────────────────────────────────────────
    print("\n=== Saving ===")
    PRED = config.PREDICTIONS_DIR
    PRED.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # fitted preprocessor
    preprocessor.save(config.MODELS_DIR / "atac_preprocessor.joblib")

    # scored gene filter
    joblib.dump(gene_filter, config.MODELS_DIR / "gene_filter.joblib")
    print(f"  Saved ScoredGeneFilter → {config.MODELS_DIR / 'gene_filter.joblib'}")

    # feature matrices
    np.save(PRED / "X_train_feat.npy", X_train_feat)
    print(f"  Saved X_train_feat {X_train_feat.shape} → {PRED / 'X_train_feat.npy'}")
    np.save(PRED / "X_test_feat.npy",  X_test_feat)
    print(f"  Saved X_test_feat  {X_test_feat.shape} → {PRED / 'X_test_feat.npy'}")

    # cell IDs and feature names (needed for metadata alignment downstream)
    np.save(PRED / "train_cell_ids.npy", train_cell_ids)
    np.save(PRED / "test_cell_ids.npy",  test_cell_ids)
    np.save(PRED / "rna_features.npy",   rna_features)
    print(f"  Saved train_cell_ids ({len(train_cell_ids):,}), "
          f"test_cell_ids ({len(test_cell_ids):,}), "
          f"rna_features ({len(rna_features):,})")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\n=== Generating preprocessing figures ===")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Figure 1: Scree plot — variance explained by each LSI component
    ev = preprocessor.explained_variance_ratio  # shape: (n_components,)
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
    print(f"  Saved: {scree_path}")

    # Figure 2: LSI embedding scatter (components 1 vs 2) coloured by donor / day / cell_type
    # Use only training cells (have known cell_type labels)
    Z_train = X_train_feat[:, :2]   # first 2 LSI components

    # Subsample for plotting speed (max 20K points)
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
    print(f"  Saved: {embed_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"  Train features:  {X_train_feat.shape}  (LSI-{config.LSI_COMPONENTS} + day)")
    print(f"  Test features:   {X_test_feat.shape}")
    print(f"  RNA targets:     {y_train.shape}")
    print(f"  Scored cells:    {len(gene_filter.scored_cell_ids):,}")
    print(f"  Scored genes:    {len(gene_filter.scored_gene_ids):,}")
    print(f"  Total time:      {time.time()-t0:.1f}s")
    print("Done.")
