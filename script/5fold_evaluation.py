import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import torch
import h5py
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import GroupKFold

import config
from src.wide_model import WideEncoderDecoder, WideTrainer
from src.data_utils import load_metadata

FIG  = config.FIGURES_DIR
FIG.mkdir(exist_ok=True)

WIDE_RED  = "#EF5350"
WIDE_DARK = "#B71C1C"

X_base         = np.load(config.PREDICTIONS_DIR / "X_train_feat.npy").astype(np.float32)
train_cell_ids = np.load(config.PREDICTIONS_DIR / "train_cell_ids.npy", allow_pickle=True)

with h5py.File(config.TRAIN_TARGETS, "r") as f:
    y_train_full = list(f.values())[0]["block0_values"][:].astype(np.float32)

meta       = load_metadata()
train_meta = meta.loc[train_cell_ids].copy()
days       = train_meta["day"].values
donors     = train_meta["donor"].values

DONOR_GENDER = {13176: 0, 27678: 0, 31800: 1, 32606: 0}
gender  = np.array([DONOR_GENDER.get(d, 0) for d in donors], dtype=np.float32)
X_train = np.hstack([X_base, gender.reshape(-1, 1)]) 

group_labels = np.array([f"{don}_{day}" for don, day in zip(donors, days)])
unique_groups = sorted(np.unique(group_labels))

for g in unique_groups:
    n = (group_labels == g).sum()

def mrpc(y_true, y_pred):
    yt = y_true - y_true.mean(axis=1, keepdims=True)
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    num = (yt * yp).sum(axis=1)
    den = np.sqrt((yt**2).sum(axis=1) * (yp**2).sum(axis=1)) + 1e-8
    return float((num / den).mean())

def run_wide_fold(X_tr, y_tr, X_va, y_va, device, label=""):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer

    svd = TruncatedSVD(n_components=512, random_state=42, n_iter=7)
    svd.fit(y_tr)
    y_tr_svd = svd.transform(y_tr).astype(np.float32)
    y_va_svd = svd.transform(y_va).astype(np.float32)

    model = WideEncoderDecoder(
        in_features=130, width=2048, n_decoder_blocks=5,
        out_features=512, dropout=0.1,
    )
    trainer = WideTrainer(
        model=model, device=device, svd_obj=svd, y_val_genes=y_va,
        lr=1e-3, weight_decay=1e-4, batch_size=512,
        max_epochs=100, patience=40, pct_start=0.3,
        pearson_weight=1.0, burnin_epochs=10,
    )
    t0 = time.time()
    hist = trainer.fit(X_tr=X_tr, y_tr_svd=y_tr_svd,
                       X_val=X_va, y_val_svd=y_va_svd)
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_va, dtype=torch.float32).to(device)
        z = [model(X_t[i:i+512]).cpu().numpy() for i in range(0, len(X_t), 512)]
    y_pred = (np.vstack(z) @ svd.components_).clip(0)
    score  = mrpc(y_va, y_pred)
    return score, hist["best_epoch"], elapsed

lodo_df = pd.read_csv(config.PREDICTIONS_DIR / "wide_model_cv_results.csv")
lodo_mean = lodo_df["mrpc"].mean()
lodo_std  = lodo_df["mrpc"].std()


gkf     = GroupKFold(n_splits=5)
records = []

for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train_full, groups=group_labels)):
    val_groups = np.unique(group_labels[va_idx])
    val_days   = np.unique(days[va_idx])
    val_donors = np.unique(donors[va_idx])

    X_tr = X_train[tr_idx];      X_va = X_train[va_idx]
    y_tr = y_train_full[tr_idx]; y_va = y_train_full[va_idx]

    score, best_ep, elapsed = run_wide_fold(
        X_tr, y_tr, X_va, y_va, device,
        label=f"Fold {fold_i+1}"
    )
    records.append({
        "fold": fold_i + 1,
        "val_groups": ", ".join(val_groups),
        "val_days":   ", ".join(map(str, val_days)),
        "n_val":      len(va_idx),
        "mrpc":       score,
        "best_epoch": best_ep,
        "elapsed_s":  round(elapsed),
    })

gkf_df   = pd.DataFrame(records)
gkf_mean = gkf_df["mrpc"].mean()
gkf_std  = gkf_df["mrpc"].std()

gkf_df.to_csv(config.PREDICTIONS_DIR / "wide_model_gkf_cv_results.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Wide Model — Evaluation Scheme Comparison", fontsize=13, fontweight="bold")

ax = axes[0]
x  = np.arange(len(lodo_df))
bars = ax.bar(x, lodo_df["mrpc"], color=WIDE_RED, alpha=0.85, width=0.55, zorder=3)
ax.axhline(lodo_mean, color=WIDE_DARK, linestyle="--", linewidth=1.5,
           label=f"Mean={lodo_mean:.5f}")
for bar, v in zip(bars, lodo_df["mrpc"]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"Day {d}" for d in lodo_df["day"]], fontsize=10)
ax.set_ylabel("MRPC"); ax.set_title("Scheme A — LODO CV\n(4 folds, by day)", fontweight="bold")
ax.set_ylim(lodo_df["mrpc"].min() * 0.992, lodo_df["mrpc"].max() * 1.012)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

ax2 = axes[1]
x2  = np.arange(len(gkf_df))
bars2 = ax2.bar(x2, gkf_df["mrpc"], color="#42A5F5", alpha=0.85, width=0.55, zorder=3)
ax2.axhline(gkf_mean, color="#1565C0", linestyle="--", linewidth=1.5,
            label=f"Mean={gkf_mean:.5f}")
for bar, v, grp in zip(bars2, gkf_df["mrpc"], gkf_df["val_groups"]):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.001,
             f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax2.set_xticks(x2)
ax2.set_xticklabels([f"Fold {i+1}\n({g})" for i, g in enumerate(gkf_df["val_groups"])],
                    fontsize=7, rotation=10)
ax2.set_ylabel("MRPC")
ax2.set_title("Scheme B — 5-fold GroupKFold\n(donor × day groups)", fontweight="bold")
ax2.set_ylim(gkf_df["mrpc"].min() * 0.992, gkf_df["mrpc"].max() * 1.012)
ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

ax3 = axes[2]
scheme_means = [lodo_mean, gkf_mean]
scheme_stds  = [lodo_std,  gkf_std]
scheme_names = ["Scheme A\n(LODO, 4-fold)", "Scheme B\n(GroupKFold, 5-fold)"]
colors = [WIDE_RED, "#42A5F5"]

bars3 = ax3.bar([0, 1], scheme_means, yerr=scheme_stds,
                color=colors, alpha=0.85, width=0.45,
                capsize=6, error_kw={"linewidth": 1.5}, zorder=3)
for i, (v, s) in enumerate(zip(scheme_means, scheme_stds)):
    ax3.text(i, v + s + 0.002, f"{v:.5f}\n±{s:.5f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

for xi, scores in [(0, lodo_df["mrpc"].values), (1, gkf_df["mrpc"].values)]:
    jitter = np.random.uniform(-0.08, 0.08, len(scores))
    ax3.scatter(xi + jitter, scores, color="black", s=30, zorder=5, alpha=0.7)

ax3.set_xticks([0, 1]); ax3.set_xticklabels(scheme_names, fontsize=10)
ax3.set_ylabel("MRPC")
ax3.set_title("Mean MRPC ± Std\n(dots = individual folds)", fontweight="bold")
ax3.set_ylim(min(scheme_means) - max(scheme_stds) * 3,
             max(scheme_means) + max(scheme_stds) * 4)
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out = FIG / "13_evaluation_scheme_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
