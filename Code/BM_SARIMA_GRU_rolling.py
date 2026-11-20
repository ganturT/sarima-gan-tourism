import os
import json
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ===================== ROLLING ORIGIN ======================
ORIGINS = [1, 2, 3, 4]
TUNE_ORIGIN = 1                # hyperparameters are selected on the validation set of origin 1 only

# ===================== SEEDS (same as the other scripts) ===
TUNE_SEEDS = [10, 11, 12]
FINAL_SEEDS = list(range(10))  # same as in the other scripts

def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)

# ===================== SETTINGS (same as HY_SARIMA_ANN_rolling.py) =
WINDOW_SIZE = 12      # previous 12 residuals -> next residual
BATCH_SIZE = 4
N_EPOCH = 300
PATIENCE = 30

LR_GRID = [0.0002, 0.0005, 0.001, 0.0016, 0.003, 0.005, 0.008, 0.01]   # same grid as the GRU benchmark
UNITS_GRID = [16, 32, 64, 128, 256, 512, 1024, 2048]   # GRU units u; includes the units of SARIMA-GAN (GRU), u = 128

# Selected values. Uncomment these lines only after the grid search has been run once
LR_GRID = [0.0005]
UNITS_GRID = [1024]

EXCLUDE_COVID = False
COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2022-03-01")

os.makedirs("Results", exist_ok=True)


# ===================== DATA ================================
def load_origin(k):
    """Residual windows of one origin. The scaler is fitted on this origin's training residuals only."""
    df = pd.read_csv(f"Results/sarima_origin{k}.csv", parse_dates=["date"]).set_index("date").sort_index()
    resid = df["residual"].to_numpy(dtype=float)
    dates = df.index
    n = len(df)

    # Training residuals used to fit the scaler: training split, after the SARIMA burn-in
    fit_mask = (df["split"] == "train") & (~df["burn_in"])
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(resid[fit_mask.to_numpy()].reshape(-1, 1))
    r_scaled = scaler.transform(resid.reshape(-1, 1)).ravel()

    X = {s: [] for s in ["train", "val", "test"]}
    y = {s: [] for s in ["train", "val", "test"]}
    idx = {s: [] for s in ["train", "val", "test"]}
    for t in range(WINDOW_SIZE, n):
        window = range(t - WINDOW_SIZE, t)                 # residuals strictly BEFORE month t
        if df["burn_in"].iloc[list(window) + [t]].any():   # no burn-in residual in input or target
            continue
        split = df["split"].iloc[t]
        if EXCLUDE_COVID and split == "train":
            touched = dates[t - WINDOW_SIZE:t + 1]
            if ((touched >= COVID_START) & (touched <= COVID_END)).any():
                continue
        X[split].append(r_scaled[t - WINDOW_SIZE:t])
        y[split].append([r_scaled[t]])
        idx[split].append(t)

    data = {"scaler": scaler, "frame": df}
    for s in ["train", "val", "test"]:
        data[f"X_{s}"] = np.asarray(X[s], dtype="float32")
        data[f"y_{s}"] = np.asarray(y[s], dtype="float32")
        data[f"pos_{s}"] = np.asarray(idx[s])
    return data


# ===================== MODEL (same as the GAN generator) ====
def build_gru(input_dim, units, lr):
    """GRU(u) -> GRU(u/2) -> Dense(u) -> Dense(u/2) -> 1; the 12 residuals are read as a (12, 1) sequence."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Reshape((input_dim, 1)))              # (12,) -> (12 months, 1 feature)
    model.add(GRU(units, return_sequences=True))
    model.add(GRU(max(units // 2, 4)))
    model.add(Dense(units, activation="relu"))
    model.add(Dense(max(units // 2, 4), activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss="mse")
    return model


def fit_gru(data, units, lr, seed, verbose=0):
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_gru(data["X_train"].shape[1], units, lr)
    # val_loss = MSE of the scaled residual. Because the hybrid error equals the residual
    # error (y - SARIMA - e_hat = e - e_hat), this is equivalent to the hybrid validation RMSE.
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(data["X_train"], data["y_train"], epochs=N_EPOCH, batch_size=BATCH_SIZE,
                        validation_data=(data["X_val"], data["y_val"]), callbacks=[es],
                        verbose=verbose, shuffle=False)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return model, history, best_epoch


# ===================== HYBRID FORECAST + METRICS ===========
def hybrid(data, split, e_hat_scaled):
    """Hybrid forecast on the original scale for the samples of one split."""
    f = data["frame"].iloc[data[f"pos_{split}"]]
    e_hat = data["scaler"].inverse_transform(e_hat_scaled.reshape(-1, 1)).ravel()
    return f.index, f["actual"].to_numpy(), f["sarima_pred"].to_numpy() + e_hat, f["sarima_pred"].to_numpy()


def metrics(y_true, y_pred):
    err = y_true - y_pred
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "MAPE": float(np.mean(np.abs(err / y_true)) * 100)}


# =====================================================================
# STEP 1: GRID SEARCH on ORIGIN 1 over MULTIPLE seeds - test set NOT used
# =====================================================================
tune = load_origin(TUNE_ORIGIN)
print(f"Origin {TUNE_ORIGIN} | residual windows train/val/test: "
      f"{len(tune['y_train'])} {len(tune['y_val'])} {len(tune['y_test'])} | EXCLUDE_COVID={EXCLUDE_COVID}")
_, yv, _, sv = hybrid(tune, "val", np.zeros(len(tune["y_val"])))
print(f"SARIMA alone on origin {TUNE_ORIGIN} validation: RMSE={metrics(yv, sv)['RMSE']:.0f}")

# ---------------- RESUME ----------------
# Every tuning run is appended to TUNING_LOG as soon as it finishes. Runs already in the log
# (same lr, units and seed) are skipped, so extending the grid only trains the new configurations.
# The log written by the earlier version of this script has the same columns and is reused.
TUNING_LOG = "Results/sarima_gru_tuning_all_runs.csv"
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
        model, history, best_epoch = fit_gru(tune, units, lr, s)
        _, y_true, y_hyb, _ = hybrid(tune, "val", model.predict(tune["X_val"], verbose=0))
        va = metrics(y_true, y_hyb)
        row = {"lr": lr, "units": units, "seed": s, "best_epoch": best_epoch,
               "val_rmse": va["RMSE"], "val_mae": va["MAE"], "val_mape": va["MAPE"]}
        pd.DataFrame([row]).to_csv(TUNING_LOG, mode="a", index=False, header=not os.path.exists(TUNING_LOG))
        print(f"[{n}/{total}] lr={lr:<7} units={units:<4} seed={s:<3} best_epoch={best_epoch:<4} "
              f"val RMSE={va['RMSE']:.0f}  MAE={va['MAE']:.0f}  MAPE={va['MAPE']:.2f}%")

# Only the runs of the current grid are used for the selection (the log may contain other runs)
runs_df = pd.read_csv(TUNING_LOG)
runs_df = runs_df[runs_df["lr"].round(6).isin([round(x, 6) for x in LR_GRID])
                  & runs_df["units"].isin(UNITS_GRID) & runs_df["seed"].isin(TUNE_SEEDS)]
print(f"Selection uses {len(runs_df)} runs (expected {total})")

res = (runs_df.groupby(["lr", "units"])
       .agg(val_rmse_mean=("val_rmse", "mean"),
            val_rmse_std=("val_rmse", "std"),
            val_rmse_worst=("val_rmse", "max"),
            val_mape_mean=("val_mape", "mean"),
            best_epoch_mean=("best_epoch", "mean"))
       .reset_index())
res["score"] = res["val_rmse_mean"] + res["val_rmse_std"].fillna(0)   # same criterion as the other models
res = res.sort_values("score")
#res.to_csv("Results/sarima_gru_tuning_results.csv", index=False)
print("\n=== Tuning results on origin 1 (sorted by mean + std of hybrid val RMSE) ===")
print(res.round(4).to_string(index=False))

BEST_LR = float(res.iloc[0]["lr"])
BEST_UNITS = int(res.iloc[0]["units"])
with open("Results/best_hparams_sarima_gru.json", "w") as f:
    json.dump({"lr": BEST_LR, "units": BEST_UNITS, "tune_seeds": TUNE_SEEDS, "tune_origin": TUNE_ORIGIN,
               "final_seeds": FINAL_SEEDS, "window": WINDOW_SIZE, "exclude_covid": EXCLUDE_COVID}, f)
print(f"\nFixed hyperparameters: lr={BEST_LR}, units={BEST_UNITS}")
if BEST_LR in (LR_GRID[0], LR_GRID[-1]) or BEST_UNITS in (UNITS_GRID[0], UNITS_GRID[-1]):
    print("NOTE: a selected value lies at the edge of the grid; consider extending the grid.")


# =====================================================================
# STEP 2: FINAL TRAINING - every origin, final seeds
# =====================================================================
records, pred_tables = [], []
for k in ORIGINS:
    data = load_origin(k)
    test_preds = []
    for seed in FINAL_SEEDS:
        model, history, best_epoch = fit_gru(data, BEST_UNITS, BEST_LR, seed)
        _, yv, hv, _ = hybrid(data, "val", model.predict(data["X_val"], verbose=0))
        e_hat_test = model.predict(data["X_test"], verbose=0)
        dates, yt, ht, st = hybrid(data, "test", e_hat_test)
        va, te = metrics(yv, hv), metrics(yt, ht)
        test_preds.append(ht)
        records.append({"origin": k, "seed": seed, "best_epoch": best_epoch,
                        "val_RMSE": va["RMSE"], "val_MAE": va["MAE"], "val_MAPE": va["MAPE"],
                        "test_RMSE": te["RMSE"], "test_MAE": te["MAE"], "test_MAPE": te["MAPE"]})
        print(f"origin={k} seed={seed} best_epoch={best_epoch:<4} "
              f"TEST RMSE={te['RMSE']:.0f}  MAE={te['MAE']:.0f}  MAPE={te['MAPE']:.2f}%")

    s_only = metrics(yt, st)
    print(f"  origin {k}: SARIMA alone on the same test months: "
          f"RMSE={s_only['RMSE']:.0f}  MAE={s_only['MAE']:.0f}  MAPE={s_only['MAPE']:.2f}%")
    tbl = pd.DataFrame({"origin": k, "date": dates, "actual": yt, "sarima_pred": st,
                        "seeds_mean_prediction": np.mean(np.stack(test_preds), axis=0)})
    #tbl.to_csv(f"Results/test_predicted_meanseed_sarima_gru_origin{k}.csv", index=False)
    pred_tables.append(tbl)

per_seed = pd.DataFrame(records)
per_seed.to_csv("Results/sarima_gru_per_seed_results.csv", index=False)
# All test forecasts (4 x 12 = 48 rows) for the DM test; keep one forecast per month (latest origin) there.
pd.concat(pred_tables, ignore_index=True).to_csv("Results/test_predicted_sarima_gru_pooled.csv", index=False)


# =====================================================================
# STEP 3: SUMMARY (method 1): per origin mean over seeds, then mean ± std over origins
# =====================================================================
cols = ["test_MAE", "test_RMSE", "test_MAPE"]
per_origin = per_seed.groupby("origin")[cols].agg(["mean", "std"])
#per_origin.to_csv("Results/sarima_gru_per_origin_results.csv")
print("\n=== SARIMA-GRU test results per origin (mean ± std over seeds) ===")
for k in ORIGINS:
    r = per_origin.loc[k]
    print(f"origin {k}: " + "  ".join(f"{c[5:]}={r[(c, 'mean')]:.2f} ± {r[(c, 'std')]:.2f}" for c in cols))

origin_means = per_seed.groupby("origin")[cols].mean()
print(f"\n=== SARIMA-GRU test over {len(ORIGINS)} origins (mean ± std of the per-origin seed means) ===")
summary = {}
for c in cols:
    mean, std = origin_means[c].mean(), origin_means[c].std(ddof=1)
    summary[c[5:]] = {"mean": mean, "std": std}
    print(f"{c[5:]:<5}: {mean:.2f} ± {std:.2f}")
pd.DataFrame(summary).T.to_csv("Results/test_summary_sarima_gru.csv")


# ===================== PLOT: origin 1, SARIMA vs hybrid on the test months =====
#t1 = pred_tables[0]
#plt.figure(figsize=(10, 4.5))
#plt.plot(t1["date"], t1["actual"], "k-o", label="Actual", markersize=4)
#plt.plot(t1["date"], t1["sarima_pred"], "--", label="SARIMA")
#plt.plot(t1["date"], t1["seeds_mean_prediction"], "-", label="SARIMA-GRU (mean over seeds)")
#plt.ylabel("Tourist arrivals")
#plt.title("Origin 1 test months")
#plt.grid(alpha=0.3)
#plt.legend()
#plt.tight_layout()
#plt.savefig("Results/test_pred_sarima_gru_origin1.pdf")
#plt.show()