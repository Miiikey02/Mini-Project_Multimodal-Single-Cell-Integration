import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hdf5plugin
import h5py
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config
from src.preprocessing import (
    ATACPreprocessor, ScoredGeneFilter, DayEncoder,
    build_features, load_h5_sparse_chunked,
)
from src.data_utils import load_metadata

t0 = time.time()
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

preprocessor = ATACPreprocessor.load(config.MODELS_DIR / "atac_preprocessor.joblib")
day_encoder   = DayEncoder()
meta          = load_metadata()

with h5py.File(config.TRAIN_TARGETS, "r") as f:
    rna_features = list(f.values())[0]["axis0"][:].astype(str)

X_train_sp, train_cell_ids, _ = load_h5_sparse_chunked(config.TRAIN_INPUTS, chunk_size=5_000)
days_train   = meta.loc[train_cell_ids, "day"].values
train_meta   = meta.loc[train_cell_ids]
X_train_feat = build_features(X_train_sp, days_train, preprocessor, day_encoder)

X_test_sp, test_cell_ids, _ = load_h5_sparse_chunked(config.TEST_INPUTS, chunk_size=5_000)
days_test   = meta.loc[test_cell_ids, "day"].values
X_test_feat = build_features(X_test_sp, days_test, preprocessor, day_encoder)
del X_test_sp

gene_filter = ScoredGeneFilter(rna_gene_names=rna_features, test_cell_ids=test_cell_ids)

PRED = config.PREDICTIONS_DIR
np.save(PRED / "X_train_feat.npy",  X_train_feat)
np.save(PRED / "X_test_feat.npy",   X_test_feat)
np.save(PRED / "train_cell_ids.npy", train_cell_ids)
np.save(PRED / "test_cell_ids.npy",  test_cell_ids)
np.save(PRED / "rna_features.npy",   rna_features)

joblib.dump(gene_filter, config.MODELS_DIR / "gene_filter.joblib")

ev = preprocessor.explained_variance_ratio

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1, len(ev) + 1), ev, color="steelblue", width=1.0)
axes[0].set_xlabel("LSI Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot — LSI Components (component 0 dropped)")

axes[1].plot(range(1, len(ev) + 1), np.cumsum(ev), color="steelblue", marker=".", markersize=3)
axes[1].axhline(np.cumsum(ev)[-1], color="grey", linestyle="--", linewidth=0.8)
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title(f"Cumulative Variance ({np.cumsum(ev)[-1]:.4f} total)")

plt.tight_layout()
scree_path = config.FIGURES_DIR / "02_lsi_scree_plot.png"
plt.savefig(scree_path, dpi=150)
plt.close()

Z_train = X_train_feat[:, :2]

rng    = np.random.RandomState(42)
n_plot = min(20_000, len(Z_train))
idx    = rng.choice(len(Z_train), n_plot, replace=False)
Z_plot = Z_train[idx]
meta_plot = train_meta.iloc[idx]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
scatter_kw = dict(s=2, alpha=0.4, rasterized=True)

for ax, col, title in zip(
    axes,
    ["donor", "day", "cell_type"],
    ["Donor", "Day", "Cell Type"],
):
    cats    = meta_plot[col].astype(str)
    unique  = sorted(cats.unique())
    palette = list(cm.tab10.colors) + list(cm.tab20.colors)
    cmap    = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
    colors  = [cmap[c] for c in cats]
    ax.scatter(Z_plot[:, 0], Z_plot[:, 1], c=colors, **scatter_kw)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap[c], markersize=6, label=c)
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


