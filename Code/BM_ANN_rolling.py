import os
import json
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pickle import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ===================== ROLLING ORIGIN ======================
# Data/origin1 ... Data/origin4 are written by Data_preprocessing_rolling.py
ORIGINS = [1, 2, 3, 4]
TUNE_ORIGIN = 1                # hyperparameters are selected on the validation set of origin 1 only

# ===================== SEEDS ===============================
TUNE_SEEDS = [10, 11, 12]      # grid search: every config is run with ALL of these seeds
FINAL_SEEDS = list(range(10))  # final model: seeds 0-9 on every origin (different from TUNE_SEEDS)


def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)

# ===================== SETTINGS (same as GAN) ==============
BATCH_SIZE = 4
N_EPOCH = 300
PATIENCE = 30

#LR_GRID = [0.0002, 0.0005, 0.001, 0.0016, 0.003, 0.005, 0.008, 0.01]
#UNITS_GRID = [64, 128, 256, 512, 1024, 2048]

# Selected values. Uncomment these lines only after the grid search has been run once
LR_GRID = [0.005]
UNITS_GRID = [1024]

os.makedirs("Results", exist_ok=True)


# ===================== DATA ================================
def load_origin(k):
    """Arrays and scaler of one origin. The ANN takes a flat vector: (samples, window * features)."""
    d = f"Data/origin{k}"
    data = {}
    for name in ["train", "val", "test"]:
        X = np.load(f"{d}/X_{name}.npy", allow_pickle=True).astype("float32")
        data[f"X_{name}"] = X.reshape(X.shape[0], -1)
        data[f"y_{name}"] = np.load(f"{d}/y_{name}.npy", allow_pickle=True).astype("float32")
        data[f"index_{name}"] = np.load(f"{d}/index_{name}.npy", allow_pickle=True)
    data["y_scaler"] = load(open(f"{d}/y_scaler.pkl", "rb"))   # fitted on this origin's training period only
    return data


# ===================== MODEL ===============================
def build_ann(input_dim, output_dim, units, lr):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(units, activation="relu"))
    model.add(Dense(max(units // 2, 8), activation="relu"))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss="mse")
    return model


def fit_ann(data, units, lr, seed, verbose=0):
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_ann(data["X_train"].shape[1], data["y_train"].shape[1], units, lr)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(data["X_train"], data["y_train"], epochs=N_EPOCH, batch_size=BATCH_SIZE,
                        validation_data=(data["X_val"], data["y_val"]), callbacks=[es],
                        verbose=verbose, shuffle=False)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return model, history, best_epoch


# ===================== METRICS (original scale) ============
def compute_metrics(y_true_scaled, y_pred_scaled, y_scaler):
    y_true = y_scaler.inverse_transform(y_true_scaled).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
    err = y_true - y_pred
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "MAPE": float(np.mean(np.abs(err / y_true)) * 100)}



tune = load_origin(TUNE_ORIGIN)
print("Origin", TUNE_ORIGIN, "| input_dim:", tune["X_train"].shape[1], "| train/val/test:",
      len(tune["y_train"]), len(tune["y_val"]), len(tune["y_test"]))


TUNING_LOG = "Results/ann_tuning_all_runs.csv"
if os.path.exists(TUNING_LOG):
    done = {(round(float(r.lr), 6), int(r.units), int(r.seed)) for r in pd.read_csv(TUNING_LOG).itertuples()}
    print(f"Resuming: {len(done)} runs already in {TUNING_LOG}")
else:
    done = set()

total = len(LR_GRID) * len(UNITS_GRID) * len(TUNE_SEEDS)
n = 0
for lr, units in itertools.product(LR_GRID, UNITS_GRID):
    for s in TUNE_SEEDS:
        n += 1
        if (round(float(lr), 6), units, s) in done:
            continue
        model, history, best_epoch = fit_ann(tune, units, lr, s)
        va = compute_metrics(tune["y_val"], model.predict(tune["X_val"], verbose=0), tune["y_scaler"])
        row = {"lr": lr, "units": units, "seed": s, "best_epoch": best_epoch,
               "val_rmse": va["RMSE"], "val_mae": va["MAE"], "val_mape": va["MAPE"]}
        pd.DataFrame([row]).to_csv(TUNING_LOG, mode="a", index=False, header=not os.path.exists(TUNING_LOG))
        print(f"[{n}/{total}] lr={lr:<7} units={units:<4} seed={s:<3} best_epoch={best_epoch:<4} "
              f"val RMSE={va['RMSE']:.0f}  MAE={va['MAE']:.0f}  MAPE={va['MAPE']:.2f}%")

runs_df = pd.read_csv(TUNING_LOG)
runs_df = runs_df[runs_df["lr"].round(6).isin([round(x, 6) for x in LR_GRID])
                  & runs_df["units"].isin(UNITS_GRID) & runs_df["seed"].isin(TUNE_SEEDS)]
print(f"Selection uses {len(runs_df)} runs (expected {total})")

# Aggregate each (lr, units) config over the tuning seeds
res = (runs_df.groupby(["lr", "units"])
       .agg(val_rmse_mean=("val_rmse", "mean"),
            val_rmse_median=("val_rmse", "median"),
            val_rmse_std=("val_rmse", "std"),
            val_rmse_worst=("val_rmse", "max"),
            val_mape_mean=("val_mape", "mean"),
            best_epoch_mean=("best_epoch", "mean"))
       .reset_index())

# Selection criterion: mean + std of val RMSE (SAME as GAN)
res["score"] = res["val_rmse_mean"] + res["val_rmse_std"].fillna(0)
res = res.sort_values("score")
#res.to_csv("Results/ann_tuning_results.csv", index=False)
print("\n=== Tuning results on origin 1 (multi-seed, sorted by mean + std of val RMSE) ===")
print(res.round(4).to_string(index=False))

# Hyperparameters are now FIXED for all origins and seeds
BEST_LR = float(res.iloc[0]["lr"])
BEST_UNITS = int(res.iloc[0]["units"])
with open("Results/best_hparams_ann.json", "w") as f:
    json.dump({"lr": BEST_LR, "units": BEST_UNITS, "tune_seeds": TUNE_SEEDS,
               "tune_origin": TUNE_ORIGIN, "final_seeds": FINAL_SEEDS}, f)
print(f"\nFixed hyperparameters: lr={BEST_LR}, units={BEST_UNITS} "
      f"(selected on origin {TUNE_ORIGIN} over seeds {TUNE_SEEDS}, mean val RMSE={res.iloc[0]['val_rmse_mean']:.0f})")
for name, val, g in [("LR", BEST_LR, LR_GRID), ("units", BEST_UNITS, UNITS_GRID)]:
    if len(g) > 1 and val in (min(g), max(g)):
        print(f"WARNING: selected {name} = {val} is at the edge of its grid {g}; consider extending it")



records = []           # one row = one (origin, seed)
pred_tables = []       # seed-mean test predictions of every origin (original scale)
histories_origin1 = {}

for k in ORIGINS:
    data = load_origin(k)
    sc = data["y_scaler"]
    test_preds = []
    for seed in FINAL_SEEDS:
        model, history, best_epoch = fit_ann(data, BEST_UNITS, BEST_LR, seed)
        tr = compute_metrics(data["y_train"], model.predict(data["X_train"], verbose=0), sc)
        va = compute_metrics(data["y_val"], model.predict(data["X_val"], verbose=0), sc)
        yhat_test = model.predict(data["X_test"], verbose=0)
        te = compute_metrics(data["y_test"], yhat_test, sc)
        test_preds.append(yhat_test)
        if k == ORIGINS[0]:
            histories_origin1[seed] = (history.history, best_epoch)

        records.append({"origin": k, "seed": seed, "best_epoch": best_epoch,
                        "train_RMSE": tr["RMSE"], "train_MAE": tr["MAE"], "train_MAPE": tr["MAPE"],
                        "val_RMSE": va["RMSE"], "val_MAE": va["MAE"], "val_MAPE": va["MAPE"],
                        "test_RMSE": te["RMSE"], "test_MAE": te["MAE"], "test_MAPE": te["MAPE"]})
        print(f"origin={k} seed={seed} best_epoch={best_epoch:<4} "
              f"TEST RMSE={te['RMSE']:.0f}  MAE={te['MAE']:.0f}  MAPE={te['MAPE']:.2f}%")

    # Seed-mean test prediction of this origin, on the original scale
    mean_pred = sc.inverse_transform(np.mean(np.stack(test_preds, axis=0), axis=0))[:, 0]
    actual = sc.inverse_transform(data["y_test"])[:, 0]
    tbl = pd.DataFrame({"origin": k, "date": data["index_test"], "actual": actual,
                        "seeds_mean_prediction": mean_pred})
    #tbl.to_csv(f"Results/test_predicted_meanseed_ann_origin{k}.csv", index=False)
    pred_tables.append(tbl)

per_seed = pd.DataFrame(records)
#per_seed.to_csv("Results/ann_per_seed_results.csv", index=False)


pd.concat(pred_tables, ignore_index=True).to_csv("Results/test_predicted_ann_pooled.csv", index=False)


# =====================================================================
# STEP 3: SUMMARY (method 1)
#   (a) per origin: mean ± std over the 10 seeds
#   (b) overall: the 4 per-origin means -> mean ± std over the origins
# =====================================================================
cols = ["test_MAE", "test_RMSE", "test_MAPE"]
per_origin = per_seed.groupby("origin")[cols].agg(["mean", "std"])
#per_origin.to_csv("Results/ann_per_origin_results.csv")
print("\n=== Test results per origin (mean ± std over seeds) ===")
for k in ORIGINS:
    r = per_origin.loc[k]
    print(f"origin {k}: " + "  ".join(f"{c[5:]}={r[(c, 'mean')]:.2f} ± {r[(c, 'std')]:.2f}" for c in cols))

origin_means = per_seed.groupby("origin")[cols].mean()
print(f"\n=== ANN test over {len(ORIGINS)} origins (mean ± std of the per-origin seed means) ===")
summary = {}
for c in cols:
    mean, std = origin_means[c].mean(), origin_means[c].std(ddof=1)
    summary[c[5:]] = {"mean": mean, "std": std}
    print(f"{c[5:]:<5}: {mean:.2f} ± {std:.2f}")
pd.DataFrame(summary).T.to_csv("Results/test_summary_ann.csv")


