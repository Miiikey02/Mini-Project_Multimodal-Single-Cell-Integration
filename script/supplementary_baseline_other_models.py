import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hdf5plugin
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import joblib

import config
from src.data_utils import load_metadata
from src.evaluate import mean_rowwise_pearson, pearson_per_row, evaluate_by_group
from src.models import MLPModel, RidgeMultiOutput
from src.train import Trainer, predict_batched

device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.time()

X_train        = np.load(config.PREDICTIONS_DIR / "X_train_feat.npy")
X_test         = np.load(config.PREDICTIONS_DIR / "X_test_feat.npy")
train_cell_ids = np.load(config.PREDICTIONS_DIR / "train_cell_ids.npy", allow_pickle=True)
test_cell_ids  = np.load(config.PREDICTIONS_DIR / "test_cell_ids.npy",  allow_pickle=True)
rna_features   = np.load(config.PREDICTIONS_DIR / "rna_features.npy",   allow_pickle=True)

with h5py.File(config.TRAIN_TARGETS, "r") as f:
    y_train = list(f.values())[0]["block0_values"][:].astype(np.float32)

meta       = load_metadata()
train_meta = meta.loc[train_cell_ids]
test_meta  = meta.loc[test_cell_ids]
days_train = train_meta["day"].values
TRAIN_DAYS = sorted(train_meta["day"].unique())

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0]

cv_results = {a: {} for a in ALPHAS}

for held_day in TRAIN_DAYS:
    train_mask = days_train != held_day
    val_mask   = days_train == held_day
    X_tr, y_tr   = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_train[val_mask],   y_train[val_mask]
    val_meta = train_meta[val_mask]

    for alpha in ALPHAS:
        model = RidgeMultiOutput(alpha=alpha)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        mrpc  = mean_rowwise_pearson(y_val, y_pred)
        by_ct = evaluate_by_group(y_val, y_pred, val_meta, "cell_type")

        cv_results[alpha][held_day] = {
            "mrpc":         mrpc,
            "by_cell_type": by_ct,
            "n_val":        int(val_mask.sum()),
        }

summary_rows = []
for alpha in ALPHAS:
    scores = [cv_results[alpha][d]["mrpc"] for d in TRAIN_DAYS]
    summary_rows.append({
        "alpha": alpha,
        "day2":  cv_results[alpha][2]["mrpc"],
        "day3":  cv_results[alpha][3]["mrpc"],
        "day4":  cv_results[alpha][4]["mrpc"],
        "day7":  cv_results[alpha][7]["mrpc"],
        "mean":  np.mean(scores),
    })

cv_df = pd.DataFrame(summary_rows).set_index("alpha")

best_alpha_day7 = cv_df["day7"].idxmax()
best_alpha_mean = cv_df["mean"].idxmax()
best_alpha = best_alpha_day7

ct_rows = []
for held_day in TRAIN_DAYS:
    by_ct = cv_results[best_alpha][held_day]["by_cell_type"]
    row = {"day": held_day}
    row.update(by_ct.to_dict())
    ct_rows.append(row)
ct_df = pd.DataFrame(ct_rows).set_index("day")

final_model = RidgeMultiOutput(alpha=best_alpha)
final_model.fit(X_train, y_train)
y_pred_test = final_model.predict(X_test)
y_pred_test = np.clip(y_pred_test, 0, None)

gene_filter = joblib.load(config.MODELS_DIR / "gene_filter.joblib")
submission  = gene_filter.build_submission(y_pred_test)
submission.to_csv(config.PREDICTIONS_DIR / "ridge_submission.csv", index=False)

final_model.save(config.MODELS_DIR / "ridge_final.joblib")
cv_df.to_csv(config.PREDICTIONS_DIR / "ridge_cv_results.csv")
ct_df.to_csv(config.PREDICTIONS_DIR / "ridge_cv_cell_type.csv")

config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

log_alphas = np.log10(ALPHAS)
colors_day = {"day2": "#4e79a7", "day3": "#f28e2b",
              "day4": "#e15759", "day7": "#76b7b2"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for col, label, lw, ls in [
    ("day2", "Day 2",         1.2, "--"),
    ("day3", "Day 3",         1.2, "--"),
    ("day4", "Day 4",         1.2, "--"),
    ("day7", "Day 7 (proxy)", 2.5, "-"),
    ("mean", "Mean LODO",     1.8, ":"),
]:
    axes[0].plot(log_alphas, cv_df[col].values, marker="o", markersize=5,
                 label=label, linewidth=lw, linestyle=ls)

axes[0].axvline(np.log10(best_alpha), color="grey", linestyle=":", linewidth=1,
                label=f"Best α={best_alpha:.0f}")
axes[0].set_xticks(log_alphas)
axes[0].set_xticklabels([f"10^{int(x)}" if x == int(x) else f"{10**x}" for x in log_alphas])
axes[0].set_xlabel("Regularisation α (log scale)")
axes[0].set_ylabel("Mean Rowwise Pearson Correlation")
axes[0].set_title("Ridge CV: MRPC vs α — Leave-One-Day-Out")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

ct_df_plot = ct_df.copy()
ct_df_plot.index = [f"Day {d}" for d in ct_df_plot.index]
sns.heatmap(ct_df_plot, annot=True, fmt=".3f", cmap="RdYlGn",
            ax=axes[1], vmin=0, vmax=1,
            linewidths=0.5, annot_kws={"size": 8})
axes[1].set_title(f"MRPC by Cell Type × Day (α={best_alpha})")
axes[1].set_xlabel("Cell Type")
axes[1].set_ylabel("Held-Out Day")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "03_ridge_cv_results.png", dpi=150)
plt.close()

val_mask_day7 = days_train == 7
X_val7  = X_train[val_mask_day7]
y_val7  = y_train[val_mask_day7]
best_model_day7 = RidgeMultiOutput(alpha=best_alpha)
best_model_day7.fit(X_train[days_train != 7], y_train[days_train != 7])
y_pred_val7 = best_model_day7.predict(X_val7)

rng    = np.random.RandomState(42)
n_s    = min(3000, len(y_val7))
idx_s  = rng.choice(len(y_val7), n_s, replace=False)
gene_s = rng.choice(y_val7.shape[1], 1)[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_val7[idx_s, gene_s], y_pred_val7[idx_s, gene_s],
                s=3, alpha=0.4, color="steelblue")
axes[0].set_xlabel("True RNA (log1p)")
axes[0].set_ylabel("Predicted RNA (log1p)")
axes[0].set_title(f"Predicted vs Actual — Day 7 val\n(gene: {rna_features[gene_s]})")
lim = [min(y_val7[:, gene_s].min(), y_pred_val7[:, gene_s].min()),
       max(y_val7[:, gene_s].max(), y_pred_val7[:, gene_s].max())]
axes[0].plot(lim, lim, "r--", linewidth=1, alpha=0.6)

per_cell_r = pearson_per_row(y_val7, y_pred_val7)
axes[1].hist(per_cell_r, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
axes[1].axvline(per_cell_r.mean(),       color="red",    linewidth=1.5,
                label=f"Mean={per_cell_r.mean():.4f}")
axes[1].axvline(np.median(per_cell_r),   color="orange", linewidth=1.5,
                label=f"Median={np.median(per_cell_r):.4f}")
axes[1].set_xlabel("Per-Cell Pearson r")
axes[1].set_ylabel("# Cells")
axes[1].set_title("Distribution of Per-Cell Pearson r — Day 7 Val")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "03_ridge_day7_analysis.png", dpi=150)
plt.close()

mask_tr7  = days_train != 7
mask_val7 = days_train == 7
X_tr7,  y_tr7  = X_train[mask_tr7],  y_train[mask_tr7]
X_val7, y_val7 = X_train[mask_val7], y_train[mask_val7]
val7_meta = train_meta[mask_val7]

model_cv = MLPModel()
trainer_cv = Trainer(
    model        = model_cv,
    device       = device,
    lr           = 3e-4,
    weight_decay = 1e-4,
    batch_size   = 512,
    max_epochs   = 150,
    patience     = 15,
    lr_patience  = 5,
    lr_factor    = 0.5,
    min_lr       = 1e-6,
)
history_cv     = trainer_cv.fit(X_tr7, y_tr7, X_val7, y_val7)
best_day7_mrpc = history_cv["best_val_mrpc"]
best_epoch     = history_cv["best_epoch"]

y_pred_val7_mlp = predict_batched(model_cv, X_val7, device)
by_ct_day7      = evaluate_by_group(y_val7, y_pred_val7_mlp, val7_meta, "cell_type")

lodo_results = {}
lodo_results[7] = {
    "mrpc":         best_day7_mrpc,
    "by_cell_type": by_ct_day7,
    "n_val":        int(mask_val7.sum()),
}

for held_day in [d for d in TRAIN_DAYS if d != 7]:
    tr_mask  = days_train != held_day
    val_mask = days_train == held_day
    m = MLPModel()
    tr = Trainer(
        model=m, device=device,
        lr=3e-4, weight_decay=1e-4, batch_size=512,
        max_epochs=150, patience=15, lr_patience=5,
        lr_factor=0.5, min_lr=1e-6,
    )
    hist = tr.fit(X_train[tr_mask], y_train[tr_mask],
                  X_train[val_mask], y_train[val_mask])
    y_pred_d = predict_batched(m, X_train[val_mask], device)
    mrpc_d   = mean_rowwise_pearson(y_train[val_mask], y_pred_d)
    by_ct_d  = evaluate_by_group(y_train[val_mask], y_pred_d,
                                  train_meta[val_mask], "cell_type")
    lodo_results[held_day] = {
        "mrpc":         mrpc_d,
        "by_cell_type": by_ct_d,
        "n_val":        int(val_mask.sum()),
    }

lodo_scores = {d: lodo_results[d]["mrpc"] for d in TRAIN_DAYS}
lodo_mean   = np.mean(list(lodo_scores.values()))

cv_df_mlp = pd.DataFrame({
    "day":   TRAIN_DAYS,
    "mrpc":  [lodo_scores[d] for d in TRAIN_DAYS],
    "n_val": [lodo_results[d]["n_val"] for d in TRAIN_DAYS],
})
cv_df_mlp.to_csv(config.PREDICTIONS_DIR / "mlp_cv_results.csv", index=False)

ct_rows = []
for d in TRAIN_DAYS:
    row = {"day": d}
    row.update(lodo_results[d]["by_cell_type"].to_dict())
    ct_rows.append(row)
ct_df_mlp = pd.DataFrame(ct_rows).set_index("day")
ct_df_mlp.to_csv(config.PREDICTIONS_DIR / "mlp_cv_cell_type.csv")

hist_df = pd.DataFrame({
    "epoch":      history_cv["epoch"],
    "train_loss": history_cv["train_loss"],
    "val_mrpc":   history_cv["val_mrpc"],
    "lr":         history_cv["lr"],
})
hist_df.to_csv(config.PREDICTIONS_DIR / "mlp_training_history.csv", index=False)

final_model_mlp = MLPModel()
final_trainer   = Trainer(
    model=final_model_mlp, device=device,
    lr=3e-4, weight_decay=1e-4, batch_size=512,
    max_epochs=150, patience=15, lr_patience=5,
    lr_factor=0.5, min_lr=1e-6,
)
final_trainer.fit(X_train, y_train, X_val7, y_val7)

y_pred_test_mlp = np.clip(predict_batched(final_model_mlp, X_test, device), 0, None)
submission_mlp  = gene_filter.build_submission(y_pred_test_mlp)
submission_mlp.to_csv(config.PREDICTIONS_DIR / "mlp_submission.csv", index=False)
final_model_mlp.save(config.MODELS_DIR / "mlp_final.pt")

ridge_cv      = pd.read_csv(config.PREDICTIONS_DIR / "ridge_cv_results.csv", index_col="alpha")
ridge_best_alpha = ridge_cv["day7"].idxmax()
ridge_scores  = {
    "day2": ridge_cv.loc[ridge_best_alpha, "day2"],
    "day3": ridge_cv.loc[ridge_best_alpha, "day3"],
    "day4": ridge_cv.loc[ridge_best_alpha, "day4"],
    "day7": ridge_cv.loc[ridge_best_alpha, "day7"],
    "mean": ridge_cv.loc[ridge_best_alpha, "mean"],
}

comparison_rows = []
for d in TRAIN_DAYS:
    comparison_rows.append({
        "day":   d,
        "ridge": ridge_scores[f"day{d}"],
        "mlp":   lodo_scores[d],
        "delta": lodo_scores[d] - ridge_scores[f"day{d}"],
    })
comparison_rows.append({
    "day":   "mean",
    "ridge": ridge_scores["mean"],
    "mlp":   lodo_mean,
    "delta": lodo_mean - ridge_scores["mean"],
})
comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(config.PREDICTIONS_DIR / "mlp_vs_ridge.csv", index=False)

for _, row in comp_df.iterrows():
    day_str = f"Day {row['day']}" if row['day'] != 'mean' else 'Mean'
    arrow   = "↑" if row["delta"] > 0 else "↓"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs   = history_cv["epoch"]
tr_loss  = history_cv["train_loss"]
val_mrpc = history_cv["val_mrpc"]
best_ep  = history_cv["best_epoch"]

axes[0].plot(epochs, tr_loss, color="#4e79a7", linewidth=1.5, label="Train MSE loss")
axes[0].axvline(best_ep, color="grey", linestyle=":", linewidth=1,
                label=f"Best epoch={best_ep}")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training Loss — Day-7 LODO Fold")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, val_mrpc, color="#e15759", linewidth=1.5, label="Val MRPC (Day 7)")
axes[1].scatter([best_ep], [best_day7_mrpc], s=80, zorder=5, color="#e15759",
                label=f"Best MRPC={best_day7_mrpc:.5f}")
ridge_day7 = ridge_scores["day7"]
axes[1].axhline(ridge_day7, color="#4e79a7", linestyle="--", linewidth=1.5,
                label=f"Ridge baseline={ridge_day7:.5f}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation MRPC")
axes[1].set_title("Validation MRPC — Day-7 LODO Fold")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "04_mlp_training_curve.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

days_plot  = [f"Day {d}" for d in TRAIN_DAYS] + ["Mean"]
ridge_vals = [ridge_scores[f"day{d}"] for d in TRAIN_DAYS] + [ridge_scores["mean"]]
mlp_vals   = [lodo_scores[d] for d in TRAIN_DAYS] + [lodo_mean]

x = np.arange(len(days_plot))
w = 0.35
axes[0].bar(x - w/2, ridge_vals, w, label=f"Ridge (α={ridge_best_alpha:.0f})",
            color="#4e79a7", alpha=0.85)
axes[0].bar(x + w/2, mlp_vals,   w, label="MLP v1",
            color="#e15759", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(days_plot)
axes[0].set_ylabel("Mean Rowwise Pearson Correlation")
axes[0].set_title("MLP vs Ridge — LODO CV MRPC")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis="y")
axes[0].set_ylim(min(ridge_vals + mlp_vals) - 0.02, max(ridge_vals + mlp_vals) + 0.02)

ct_df_plot = ct_df_mlp.copy()
ct_df_plot.index = [f"Day {d}" for d in ct_df_plot.index]
sns.heatmap(ct_df_plot, annot=True, fmt=".3f", cmap="RdYlGn",
            ax=axes[1], vmin=0, vmax=1,
            linewidths=0.5, annot_kws={"size": 8})
axes[1].set_title("MLP MRPC by Cell Type × Day")
axes[1].set_xlabel("Cell Type")
axes[1].set_ylabel("Held-Out Day")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "04_mlp_comparison.png", dpi=150)
plt.close()

ridge_cv_model = RidgeMultiOutput(alpha=float(ridge_best_alpha))
ridge_cv_model.fit(X_tr7, y_tr7)
y_pred_ridge_val7 = ridge_cv_model.predict(X_val7)

per_cell_mlp_v1 = pearson_per_row(y_val7, y_pred_val7_mlp)
per_cell_ridge  = pearson_per_row(y_val7, y_pred_ridge_val7)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(min(per_cell_ridge.min(), per_cell_mlp_v1.min()),
                   max(per_cell_ridge.max(), per_cell_mlp_v1.max()), 80)
ax.hist(per_cell_ridge,  bins=bins, alpha=0.6, color="#4e79a7",
        label=f"Ridge  mean={per_cell_ridge.mean():.4f}",  edgecolor="none")
ax.hist(per_cell_mlp_v1, bins=bins, alpha=0.6, color="#e15759",
        label=f"MLP v1 mean={per_cell_mlp_v1.mean():.4f}", edgecolor="none")
ax.axvline(per_cell_ridge.mean(),  color="#4e79a7", linewidth=1.5, linestyle="--")
ax.axvline(per_cell_mlp_v1.mean(), color="#e15759", linewidth=1.5, linestyle="--")
ax.set_xlabel("Per-Cell Pearson r")
ax.set_ylabel("# Cells")
ax.set_title("Per-Cell Pearson Distribution — Day-7 Val\nMLP v1 vs Ridge")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "04_mlp_vs_ridge_distribution.png", dpi=150)
plt.close()

COLORS = {"Ridge": "#4e79a7", "MLP v1": "#f28e2b", "MLP v2": "#e15759"}

model_v2 = MLPModel(dropout=0.2, use_residual=True)
trainer_v2 = Trainer(
    model          = model_v2,
    device         = device,
    lr             = 1e-3,
    weight_decay   = 5e-4,
    batch_size     = 512,
    max_epochs     = 150,
    patience       = 20,
    scheduler_type = "cosine_warm",
    cosine_t0      = 25,
    min_lr         = 1e-6,
)
history_v2 = trainer_v2.fit(X_tr7, y_tr7, X_val7, y_val7)
best_v2    = history_v2["best_val_mrpc"]

model_v2.save(config.MODELS_DIR / "mlp_v2_day7_lodo.pt")
hist_v2_df = pd.DataFrame({k: history_v2[k] for k in ["epoch", "train_loss", "val_mrpc", "lr"]})
hist_v2_df.to_csv(config.PREDICTIONS_DIR / "mlp_v2_training_history.csv", index=False)

v1_day7    = lodo_scores[7]
ridge_day7 = ridge_scores["day7"]

lodo_v2    = {}
lodo_v2[7] = {"mrpc": best_v2, "n_val": int(mask_val7.sum())}

for held_day in [d for d in TRAIN_DAYS if d != 7]:
    tr_mask  = days_train != held_day
    val_mask = days_train == held_day
    m = MLPModel(dropout=0.2, use_residual=True)
    tr = Trainer(m, device, lr=1e-3, weight_decay=5e-4, batch_size=512,
                 max_epochs=150, patience=20, scheduler_type="cosine_warm",
                 cosine_t0=25, min_lr=1e-6)
    tr.fit(X_train[tr_mask], y_train[tr_mask],
           X_train[val_mask], y_train[val_mask], verbose=False)
    y_pred_d = predict_batched(m, X_train[val_mask], device)
    mrpc_d   = mean_rowwise_pearson(y_train[val_mask], y_pred_d)
    by_ct    = evaluate_by_group(y_train[val_mask], y_pred_d,
                                  train_meta[val_mask], "cell_type")
    lodo_v2[held_day] = {"mrpc": mrpc_d, "n_val": int(val_mask.sum()),
                         "by_cell_type": by_ct}

y_pred_v2_val7 = predict_batched(model_v2, X_val7, device)
lodo_v2[7]["by_cell_type"] = evaluate_by_group(y_val7, y_pred_v2_val7, val7_meta, "cell_type")

cv_v2_rows = [{"day": d, "mrpc": lodo_v2[d]["mrpc"], "n_val": lodo_v2[d]["n_val"]}
              for d in TRAIN_DAYS]
cv_v2_df = pd.DataFrame(cv_v2_rows)
cv_v2_df.to_csv(config.PREDICTIONS_DIR / "mlp_v2_cv_results.csv", index=False)

ct_v2_rows = [{"day": d, **lodo_v2[d]["by_cell_type"].to_dict()} for d in TRAIN_DAYS]
ct_v2_df   = pd.DataFrame(ct_v2_rows).set_index("day")
ct_v2_df.to_csv(config.PREDICTIONS_DIR / "mlp_v2_cv_cell_type.csv")

final_v2 = MLPModel(dropout=0.2, use_residual=True)
final_trainer_v2 = Trainer(final_v2, device, lr=1e-3, weight_decay=5e-4,
                            batch_size=512, max_epochs=150, patience=20,
                            scheduler_type="cosine_warm", cosine_t0=25, min_lr=1e-6)
final_trainer_v2.fit(X_train, y_train, X_val7, y_val7)
final_v2.save(config.MODELS_DIR / "mlp_v2_final.pt")

y_pred_test_v2 = np.clip(predict_batched(final_v2, X_test, device), 0, None)
submission_v2  = gene_filter.build_submission(y_pred_test_v2)
submission_v2.to_csv(config.PREDICTIONS_DIR / "mlp_v2_submission.csv", index=False)

ridge_ct   = pd.read_csv(config.PREDICTIONS_DIR / "ridge_cv_cell_type.csv", index_col="day")
mlp_v1_cv  = pd.read_csv(config.PREDICTIONS_DIR / "mlp_cv_results.csv").set_index("day")
mlp_v1_ct  = pd.read_csv(config.PREDICTIONS_DIR / "mlp_cv_cell_type.csv", index_col="day")
CELL_TYPES = sorted(val7_meta["cell_type"].unique())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

days_label = [f"Day {d}" for d in TRAIN_DAYS] + ["Mean"]
ridge_vals_all = ([ridge_cv.loc[ridge_best_alpha, f"day{d}"] for d in TRAIN_DAYS]
                  + [ridge_cv.loc[ridge_best_alpha, "mean"]])
v1_vals        = ([mlp_v1_cv.loc[d, "mrpc"] for d in TRAIN_DAYS]
                  + [mlp_v1_cv["mrpc"].mean()])
v2_vals        = ([lodo_v2[d]["mrpc"] for d in TRAIN_DAYS]
                  + [cv_v2_df["mrpc"].mean()])

x = np.arange(len(days_label))
w = 0.25
axes[0].bar(x - w,     ridge_vals_all, w, label="Ridge",   color=COLORS["Ridge"],  alpha=0.85)
axes[0].bar(x,         v1_vals,        w, label="MLP v1",  color=COLORS["MLP v1"], alpha=0.85)
axes[0].bar(x + w,     v2_vals,        w, label="MLP v2",  color=COLORS["MLP v2"], alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(days_label)
axes[0].set_ylabel("MRPC")
axes[0].set_title("LODO CV: Ridge vs MLP v1 vs MLP v2")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis="y")
ymin = min(ridge_vals_all + v1_vals + v2_vals) - 0.015
ymax = max(ridge_vals_all + v1_vals + v2_vals) + 0.015
axes[0].set_ylim(ymin, ymax)

delta_v1 = [v - r for v, r in zip(v1_vals, ridge_vals_all)]
delta_v2 = [v - r for v, r in zip(v2_vals, ridge_vals_all)]
axes[1].bar(x - w/2, delta_v1, w, label="MLP v1 − Ridge", color=COLORS["MLP v1"], alpha=0.85)
axes[1].bar(x + w/2, delta_v2, w, label="MLP v2 − Ridge", color=COLORS["MLP v2"], alpha=0.85)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(days_label)
axes[1].set_ylabel("Δ MRPC vs Ridge")
axes[1].set_title("Improvement over Ridge Baseline")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "05_01_lodo_comparison.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

ridge_ct_plot = ridge_ct.copy()
ridge_ct_plot.index = [f"Day {d}" for d in ridge_ct_plot.index]
ct_v2_plot = ct_v2_df.copy()
ct_v2_plot.index = [f"Day {d}" for d in ct_v2_plot.index]
delta_ct = ct_v2_plot - ridge_ct_plot

vmin, vmax = 0.45, 0.75
sns.heatmap(ridge_ct_plot, annot=True, fmt=".3f", cmap="RdYlGn",
            ax=axes[0], vmin=vmin, vmax=vmax,
            linewidths=0.5, annot_kws={"size": 8}, cbar=False)
axes[0].set_title("Ridge MRPC by Cell Type × Day")
axes[0].tick_params(axis="x", rotation=30)

sns.heatmap(ct_v2_plot, annot=True, fmt=".3f", cmap="RdYlGn",
            ax=axes[1], vmin=vmin, vmax=vmax,
            linewidths=0.5, annot_kws={"size": 8}, cbar=False)
axes[1].set_title("MLP v2 MRPC by Cell Type × Day")
axes[1].tick_params(axis="x", rotation=30)

sns.heatmap(delta_ct, annot=True, fmt="+.3f", cmap="RdYlGn",
            ax=axes[2], center=0, vmin=-0.02, vmax=0.02,
            linewidths=0.5, annot_kws={"size": 8})
axes[2].set_title("Δ MRPC (MLP v2 − Ridge)")
axes[2].tick_params(axis="x", rotation=30)

plt.suptitle("Cell-Type Performance: Ridge vs MLP v2", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "05_02_cell_type_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for scores, label, color in [
    (ridge_vals_all[:4], "Ridge",  COLORS["Ridge"]),
    (v1_vals[:4],        "MLP v1", COLORS["MLP v1"]),
    (v2_vals[:4],        "MLP v2", COLORS["MLP v2"]),
]:
    axes[0].plot(TRAIN_DAYS, scores, marker="o", markersize=7,
                 linewidth=2, label=label, color=color)
axes[0].set_xticks(TRAIN_DAYS)
axes[0].set_xlabel("Held-Out Day")
axes[0].set_ylabel("MRPC")
axes[0].set_title("Overall MRPC by Held-Out Day")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].annotate("← temporal proxy for Day 10",
                 xy=(7, v2_vals[3]),
                 xytext=(5.5, v2_vals[3] - 0.012), fontsize=8, color="grey",
                 arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

ct_palette = sns.color_palette("tab10", len(CELL_TYPES))
for i, ct in enumerate(sorted(ct_v2_df.columns)):
    vals = [ct_v2_df.loc[d, ct] for d in TRAIN_DAYS]
    axes[1].plot(TRAIN_DAYS, vals, marker="o", markersize=5,
                 linewidth=1.5, label=ct, color=ct_palette[i])
axes[1].set_xticks(TRAIN_DAYS)
axes[1].set_xlabel("Held-Out Day")
axes[1].set_ylabel("MRPC")
axes[1].set_title("MLP v2: MRPC by Cell Type × Day")
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "05_03_performance_by_day.png", dpi=150)
plt.close()

rng = np.random.RandomState(42)
per_gene_r  = np.array([
    np.corrcoef(y_val7[:, g], y_pred_v2_val7[:, g])[0, 1]
    for g in rng.choice(y_val7.shape[1], 500, replace=False)
])
sorted_genes    = np.argsort(per_gene_r)[::-1]
high_gene_local = rng.choice(y_val7.shape[1], 500, replace=False)[sorted_genes[0]]
gene_name       = rna_features[high_gene_local]

n_ct = len(CELL_TYPES)
fig, axes = plt.subplots(2, n_ct, figsize=(3.5 * n_ct, 7))

for col, ct in enumerate(CELL_TYPES):
    ct_mask = val7_meta["cell_type"].values == ct
    if ct_mask.sum() < 10:
        axes[0, col].axis("off")
        axes[1, col].axis("off")
        continue
    n_s = min(1000, ct_mask.sum())
    idx = rng.choice(ct_mask.sum(), n_s, replace=False)

    y_ct_true  = y_val7[ct_mask][idx, high_gene_local]
    y_ct_ridge = y_pred_ridge_val7[ct_mask][idx, high_gene_local]
    y_ct_v2    = y_pred_v2_val7[ct_mask][idx, high_gene_local]

    ct_mrpc_ridge = mean_rowwise_pearson(y_val7[ct_mask], y_pred_ridge_val7[ct_mask])
    ct_mrpc_v2    = mean_rowwise_pearson(y_val7[ct_mask], y_pred_v2_val7[ct_mask])

    for row, (y_pred_ct, model_label, color, ct_mrpc) in enumerate([
        (y_ct_ridge, "Ridge",  COLORS["Ridge"],  ct_mrpc_ridge),
        (y_ct_v2,   "MLP v2", COLORS["MLP v2"], ct_mrpc_v2),
    ]):
        ax = axes[row, col]
        ax.scatter(y_ct_true, y_pred_ct, s=4, alpha=0.4, color=color)
        lim = [min(y_ct_true.min(), y_pred_ct.min()),
               max(y_ct_true.max(), y_pred_ct.max())]
        ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
        r = np.corrcoef(y_ct_true, y_pred_ct)[0, 1]
        ax.set_title(f"{ct}\n{model_label}  r={r:.3f}  MRPC={ct_mrpc:.3f}", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{model_label}\nPredicted", fontsize=8)
        if row == 1:
            ax.set_xlabel("True RNA (log1p)", fontsize=8)
        ax.tick_params(labelsize=7)

plt.suptitle(f"Predicted vs Actual — Day-7 Val — Gene: {gene_name}", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "05_04_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()

per_cell_ridge_arr = pearson_per_row(y_val7, y_pred_ridge_val7)
per_cell_v2_arr    = pearson_per_row(y_val7, y_pred_v2_val7)

rows = []
for ct in CELL_TYPES:
    ct_mask = val7_meta["cell_type"].values == ct
    for r_val in per_cell_ridge_arr[ct_mask]:
        rows.append({"cell_type": ct, "Pearson r": r_val, "Model": "Ridge"})
    for r_val in per_cell_v2_arr[ct_mask]:
        rows.append({"cell_type": ct, "Pearson r": r_val, "Model": "MLP v2"})
violin_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(13, 5))
sns.violinplot(data=violin_df, x="cell_type", y="Pearson r", hue="Model",
               split=True, inner="quart",
               palette={"Ridge": COLORS["Ridge"], "MLP v2": COLORS["MLP v2"]},
               ax=ax)
ax.set_title("Per-Cell Pearson Distribution by Cell Type — Day-7 Validation")
ax.set_xlabel("Cell Type")
ax.set_ylabel("Per-Cell Pearson r")
ax.axhline(per_cell_ridge_arr.mean(), color=COLORS["Ridge"],  linestyle="--",
           linewidth=1.2, alpha=0.7, label=f"Ridge mean={per_cell_ridge_arr.mean():.4f}")
ax.axhline(per_cell_v2_arr.mean(),    color=COLORS["MLP v2"], linestyle="--",
           linewidth=1.2, alpha=0.7, label=f"MLP v2 mean={per_cell_v2_arr.mean():.4f}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "05_05_pearson_by_cell_type.png", dpi=150)
plt.close()

y_true_c       = y_val7           - y_val7.mean(axis=0, keepdims=True)
y_pred_c_ridge = y_pred_ridge_val7 - y_pred_ridge_val7.mean(axis=0, keepdims=True)
y_pred_c_v2    = y_pred_v2_val7   - y_pred_v2_val7.mean(axis=0, keepdims=True)

num_ridge   = (y_true_c * y_pred_c_ridge).sum(axis=0)
num_v2      = (y_true_c * y_pred_c_v2).sum(axis=0)
denom_true  = np.sqrt((y_true_c**2).sum(axis=0))
denom_ridge = np.sqrt((y_pred_c_ridge**2).sum(axis=0))
denom_v2    = np.sqrt((y_pred_c_v2**2).sum(axis=0))

per_gene_ridge = np.where(denom_true * denom_ridge == 0, np.nan,
                           num_ridge / (denom_true * denom_ridge))
per_gene_v2    = np.where(denom_true * denom_v2    == 0, np.nan,
                           num_v2    / (denom_true * denom_v2))

pg_ridge = per_gene_ridge[~np.isnan(per_gene_ridge)]
pg_v2    = per_gene_v2[~np.isnan(per_gene_v2)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins = np.linspace(min(pg_ridge.min(), pg_v2.min()),
                   max(pg_ridge.max(), pg_v2.max()), 100)
axes[0].hist(pg_ridge, bins=bins, alpha=0.6, color=COLORS["Ridge"],
             label=f"Ridge  median={np.median(pg_ridge):.3f}", edgecolor="none")
axes[0].hist(pg_v2,    bins=bins, alpha=0.6, color=COLORS["MLP v2"],
             label=f"MLP v2 median={np.median(pg_v2):.3f}",   edgecolor="none")
axes[0].axvline(np.median(pg_ridge), color=COLORS["Ridge"],  linestyle="--", linewidth=1.5)
axes[0].axvline(np.median(pg_v2),    color=COLORS["MLP v2"], linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Per-Gene Pearson r (across Day-7 val cells)")
axes[0].set_ylabel("# Genes")
axes[0].set_title("Gene-Level Predictability Distribution")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

thresholds = np.linspace(0, 0.8, 100)
frac_ridge = [(pg_ridge > t).mean() for t in thresholds]
frac_v2    = [(pg_v2    > t).mean() for t in thresholds]
axes[1].plot(thresholds, frac_ridge, color=COLORS["Ridge"],  linewidth=2, label="Ridge")
axes[1].plot(thresholds, frac_v2,    color=COLORS["MLP v2"], linewidth=2, label="MLP v2")
axes[1].set_xlabel("Pearson r threshold")
axes[1].set_ylabel("Fraction of genes above threshold")
axes[1].set_title("Cumulative Gene Predictability (CDF complement)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "05_06_gene_predictability.png", dpi=150)
plt.close()

v2_mean = cv_v2_df["mrpc"].mean()
v1_mean = mlp_v1_cv["mrpc"].mean()

summary = pd.DataFrame([
    {"model": "Ridge",  "day2": ridge_cv.loc[ridge_best_alpha, "day2"],
     "day3": ridge_cv.loc[ridge_best_alpha, "day3"],
     "day4": ridge_cv.loc[ridge_best_alpha, "day4"],
     "day7": ridge_cv.loc[ridge_best_alpha, "day7"],
     "mean": ridge_cv.loc[ridge_best_alpha, "mean"]},
    {"model": "MLP v1", "day2": mlp_v1_cv.loc[2, "mrpc"],
     "day3": mlp_v1_cv.loc[3, "mrpc"],
     "day4": mlp_v1_cv.loc[4, "mrpc"],
     "day7": mlp_v1_cv.loc[7, "mrpc"],
     "mean": v1_mean},
    {"model": "MLP v2", "day2": lodo_v2[2]["mrpc"],
     "day3": lodo_v2[3]["mrpc"],
     "day4": lodo_v2[4]["mrpc"],
     "day7": lodo_v2[7]["mrpc"],
     "mean": v2_mean},
])
summary.to_csv(config.PREDICTIONS_DIR / "model_comparison_summary.csv", index=False)
