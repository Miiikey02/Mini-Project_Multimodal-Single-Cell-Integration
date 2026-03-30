import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hdf5plugin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
from src.data_utils import (
    inspect_h5,
    load_metadata,
    load_evaluation_ids,
    sparsity_stats,
    summarize_metadata,
    get_h5_cell_ids,
    get_h5_feature_names,
)

FIGURES = config.FIGURES_DIR
FIGURES.mkdir(parents=True, exist_ok=True)

inspect_h5(path)
if config.TRAIN_INPUTS.exists():
    train_cell_ids = get_h5_cell_ids(config.TRAIN_INPUTS)
    atac_features = get_h5_feature_names(config.TRAIN_INPUTS)
    rna_features = get_h5_feature_names(config.TRAIN_TARGETS)
    test_cell_ids = get_h5_cell_ids(config.TEST_INPUTS)

if config.METADATA.exists():
    meta = load_metadata()
    summarize_metadata(meta)

    train_meta = meta[meta["donor"].isin(config.TRAIN_DONORS)]
    test_meta = meta[meta["donor"] == config.TEST_DONOR]

if config.EVALUATION_IDS.exists():
    eval_ids = load_evaluation_ids()

    scored_cells = eval_ids["cell_id"].nunique() if "cell_id" in eval_ids.columns else "N/A"
    scored_genes = eval_ids["gene"].nunique() if "gene" in eval_ids.columns else "N/A"

if config.TRAIN_INPUTS.exists():
    import h5py

    SAMPLE = 5_000
    with h5py.File(config.TRAIN_INPUTS, "r") as f:
        g = list(f.values())[0]
        n_cells = g["block0_values"].shape[0]
        idx = np.random.RandomState(42).choice(n_cells, size=min(SAMPLE, n_cells), replace=False)
        idx.sort()
        X_sample = g["block0_values"][idx, :]

    with h5py.File(config.TRAIN_TARGETS, "r") as f:
        g = list(f.values())[0]
        y_sample = g["block0_values"][idx, :]

    sparsity_stats(X_sample, "ATAC (sample)")
    sparsity_stats(y_sample, "RNA (sample)")

if config.METADATA.exists():
    meta = load_metadata()
    train_meta = meta[meta["donor"].isin(config.TRAIN_DONORS)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    donor_counts = meta.groupby(["donor", "day"]).size().reset_index(name="count")
    pivot = donor_counts.pivot(index="day", columns="donor", values="count").fillna(0)
    pivot.plot(kind="bar", ax=axes[0], colormap="tab10")
    axes[0].set_title("Cell Counts by Day × Donor")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("# Cells")
    axes[0].tick_params(axis="x", rotation=0)

    ct_day = train_meta.groupby(["day", "cell_type"]).size().unstack(fill_value=0)
    ct_day_pct = ct_day.div(ct_day.sum(axis=1), axis=0)
    ct_day_pct.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10")
    axes[1].set_title("Cell Type Composition by Day (Train)")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Fraction")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(loc="upper right", fontsize=7)

    ct_counts = meta["cell_type"].value_counts()
    axes[2].bar(ct_counts.index, ct_counts.values, color=sns.color_palette("tab10", len(ct_counts)))
    axes[2].set_title("Total Cells per Cell Type")
    axes[2].set_xlabel("Cell Type")
    axes[2].set_ylabel("# Cells")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = FIGURES / "01_metadata_overview.png"
    plt.savefig(out, dpi=150)
    plt.close()

if config.TRAIN_INPUTS.exists() and 'X_sample' in dir():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    atac_nonzero = X_sample[X_sample > 0].flatten()
    axes[0].hist(atac_nonzero, bins=100, color="steelblue", edgecolor="none")
    axes[0].set_title("ATAC Values (non-zero, sample)")
    axes[0].set_xlabel("TF-IDF value")
    axes[0].set_ylabel("Count")

    rna_nonzero = y_sample[y_sample > 0].flatten()
    axes[1].hist(rna_nonzero, bins=100, color="salmon", edgecolor="none")
    axes[1].set_title("RNA Values (non-zero, sample)")
    axes[1].set_xlabel("log1p normalized value")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    out = FIGURES / "01_value_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close()

