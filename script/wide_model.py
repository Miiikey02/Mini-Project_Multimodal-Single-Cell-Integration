import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hdf5plugin, h5py
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.decomposition import TruncatedSVD

import config
from src.data_utils import load_metadata
from src.evaluate import mean_rowwise_pearson, evaluate_by_group
from src.wide_model import WideEncoderDecoder, WideTrainer, svd_predict_mrpc
from src.wide_model.trainer import predict_batched

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true",
                    help="Run Day-7 fold only (local Mac smoke test)")
args = parser.parse_args()
LOCAL_ONLY = args.local

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

DONOR_GENDER = {13176: 0, 27678: 0, 31800: 1, 32606: 0}

HP = dict(
    width            = 2048,
    n_decoder_blocks = 5,
    svd_components   = 512,
    dropout          = 0.1,
    lr               = 1e-3,
    weight_decay     = 1e-4,
    batch_size       = 512,
    max_epochs       = 100,
    patience         = 20,
    pearson_weight   = 1.0,
    burnin_epochs    = 10,
    pct_start        = 0.3,
)

X_base         = np.load(config.PREDICTIONS_DIR / "X_train_feat.npy").astype(np.float32)
X_test_base    = np.load(config.PREDICTIONS_DIR / "X_test_feat.npy").astype(np.float32)
train_cell_ids = np.load(config.PREDICTIONS_DIR / "train_cell_ids.npy", allow_pickle=True)
test_cell_ids  = np.load(config.PREDICTIONS_DIR / "test_cell_ids.npy",  allow_pickle=True)

with h5py.File(config.TRAIN_TARGETS, "r") as f:
    y_train = list(f.values())[0]["block0_values"][:].astype(np.float32)

meta       = load_metadata()
train_meta = meta.loc[train_cell_ids]
test_meta  = meta.loc[test_cell_ids]
days_train = train_meta["day"].values
TRAIN_DAYS = sorted(train_meta["day"].unique())


def add_gender(X_base, meta_df):
    donors = meta_df["donor"].values
    gender = np.array([DONOR_GENDER.get(d, 0) for d in donors], dtype=np.float32)
    return np.hstack([X_base, gender.reshape(-1, 1)])


X_train = add_gender(X_base,      train_meta)
X_test  = add_gender(X_test_base, test_meta)

IN_DIM = X_train.shape[1]

svd = TruncatedSVD(n_components=HP["svd_components"], random_state=42)
svd.fit(y_train)
evr = svd.explained_variance_ratio_.sum()
joblib.dump(svd, config.MODELS_DIR / "wide_model_svd.joblib")

mask_tr7  = days_train != 7
mask_val7 = days_train == 7
X_tr7,  y_tr7  = X_train[mask_tr7],  y_train[mask_tr7]
X_val7, y_val7 = X_train[mask_val7], y_train[mask_val7]
val7_meta      = train_meta[mask_val7]
y_tr7_svd      = svd.transform(y_tr7).astype(np.float32)
y_val7_svd     = svd.transform(y_val7).astype(np.float32)

model_d7 = WideEncoderDecoder(
    in_features      = IN_DIM,
    width            = HP["width"],
    n_decoder_blocks = HP["n_decoder_blocks"],
    out_features     = HP["svd_components"],
    dropout          = HP["dropout"],
)

trainer_d7 = WideTrainer(
    model          = model_d7,
    device         = device,
    svd_obj        = svd,
    y_val_genes    = y_val7,
    lr             = HP["lr"],
    weight_decay   = HP["weight_decay"],
    batch_size     = HP["batch_size"],
    max_epochs     = HP["max_epochs"],
    patience       = HP["patience"],
    pearson_weight = HP["pearson_weight"],
    burnin_epochs  = HP["burnin_epochs"],
    pct_start      = HP["pct_start"],
)

history_d7 = trainer_d7.fit(X_tr7, y_tr7_svd, X_val7, y_val7_svd)
best_d7    = history_d7["best_val_mrpc"]

model_d7.save(config.MODELS_DIR / "wide_model_day7_lodo.pt")
hist_df = pd.DataFrame({k: history_d7[k] for k in ["epoch", "train_loss", "val_mrpc", "lr"]})
hist_df.to_csv(config.PREDICTIONS_DIR / "wide_model_training_history.csv", index=False)

summary  = pd.read_csv(config.PREDICTIONS_DIR / "model_comparison_summary.csv").set_index("model")
ridge_d7 = float(summary.loc["Ridge",  "day7"])
v2_d7    = float(summary.loc["MLP v2", "day7"])

def make_trainer(svd_obj, y_val_genes):
    m = WideEncoderDecoder(
        in_features=IN_DIM, width=HP["width"],
        n_decoder_blocks=HP["n_decoder_blocks"],
        out_features=HP["svd_components"], dropout=HP["dropout"],
    )
    return WideTrainer(
        model=m, device=device, svd_obj=svd_obj, y_val_genes=y_val_genes,
        lr=HP["lr"], weight_decay=HP["weight_decay"], batch_size=HP["batch_size"],
        max_epochs=HP["max_epochs"], patience=HP["patience"],
        pearson_weight=HP["pearson_weight"], burnin_epochs=HP["burnin_epochs"],
        pct_start=HP["pct_start"],
    ), m

lodo_wide = {7: {"mrpc": best_d7, "n_val": int(mask_val7.sum())}}
_, y_pred_val7 = svd_predict_mrpc(model_d7, X_val7, y_val7, svd, device)
lodo_wide[7]["by_cell_type"] = evaluate_by_group(y_val7, y_pred_val7, val7_meta, "cell_type")

for held_day in [d for d in TRAIN_DAYS if d != 7]:
    tr_mask  = days_train != held_day
    val_mask = days_train == held_day
    X_tr_d   = X_train[tr_mask]
    y_tr_d   = y_train[tr_mask]
    X_val_d  = X_train[val_mask]
    y_val_d  = y_train[val_mask]
    meta_d   = train_meta[val_mask]

    y_tr_d_svd  = svd.transform(y_tr_d).astype(np.float32)
    y_val_d_svd = svd.transform(y_val_d).astype(np.float32)

    tr, m = make_trainer(svd, y_val_d)
    tr.fit(X_tr_d, y_tr_d_svd, X_val_d, y_val_d_svd, verbose=False)
    mrpc_d, y_pred_d = svd_predict_mrpc(m, X_val_d, y_val_d, svd, device)
    by_ct = evaluate_by_group(y_val_d, y_pred_d, meta_d, "cell_type")
    lodo_wide[held_day] = {"mrpc": mrpc_d, "n_val": int(val_mask.sum()), "by_cell_type": by_ct}
    
cv_df = pd.DataFrame([{"day": d, "mrpc": lodo_wide[d]["mrpc"], "n_val": lodo_wide[d]["n_val"]}
                       for d in TRAIN_DAYS])
cv_df.to_csv(config.PREDICTIONS_DIR / "wide_model_cv_results.csv", index=False)
wide_mean = cv_df["mrpc"].mean()

y_train_svd  = svd.transform(y_train).astype(np.float32)
final_trainer, final_model = make_trainer(svd, y_val7)
final_trainer.fit(X_train, y_train_svd, X_val7, y_val7_svd)
final_model.save(config.MODELS_DIR / "wide_model_final.pt")

z_test      = predict_batched(final_model, X_test, device)
y_pred_test = np.clip(z_test @ svd.components_, 0, None)
gene_filter = joblib.load(config.MODELS_DIR / "gene_filter.joblib")
submission  = gene_filter.build_submission(y_pred_test)
sub_path    = config.PREDICTIONS_DIR / "wide_model_submission.csv"
submission.to_csv(sub_path, index=False)

updated = pd.concat([
    summary.reset_index(),
    pd.DataFrame([{
        "model": "Wide model",
        "day2": lodo_wide[2]["mrpc"], "day3": lodo_wide[3]["mrpc"],
        "day4": lodo_wide[4]["mrpc"], "day7": best_d7, "mean": wide_mean,
    }])
], ignore_index=True)
updated.to_csv(config.PREDICTIONS_DIR / "model_comparison_summary.csv", index=False)
