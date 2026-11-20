import os
import gc
import json
import random
import time
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, LeakyReLU, GRU, Reshape, Permute

RES_DIR = "Results"     # SARIMA residuals written by BM_SARIMA_rolling.py

# ===================== ROLLING ORIGIN ======================
ORIGINS = [1, 2, 3, 4]
TUNE_ORIGIN = 1                # hyperparameters are selected on the validation set of origin 1 only

# ===================== SEEDS (same as the other scripts) ===
TUNE_SEEDS = [10, 11, 12]
FINAL_SEEDS = list(range(10))


def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)


# ===================== SETTINGS (same as GAN_rolling.py) ===
WINDOW_SIZE = 12      # previous 12 residuals -> next residual
BATCH_SIZE = 4
EPOCHS = 300
PATIENCE = 30

LR_GRID = [0.0001, 0.0002, 0.0005, 0.001, 0.0016, 0.003, 0.005]
UNITS_GRID = [16, 32, 64, 128, 256]                  # GRU generator units u (residuals are close to white noise: small models)
D_UNITS_GRID = [8, 16, 32, 64, 128, 256, 512, 1024]              # discriminator units d (Conv1D d/2, d, d; Dense d, d/2)

# Selected values. Uncomment these lines only after the grid search has been run once
LR_GRID = [0.0002]
UNITS_GRID = [128]
D_UNITS_GRID = [512]

# Every tuning run is saved here immediately; if the script stops, running it again
# skips the runs already in this file.
TUNING_LOG = f"{RES_DIR}/sarima_gan_gru_tuning_all_runs.csv"


# ===================== DATA ================================
def load_origin(k):
    """Residual windows of one origin. The scaler is fitted on this origin's training residuals only.
    X = 12 scaled residuals, y = next scaled residual."""
    df = pd.read_csv(f"{RES_DIR}/sarima_origin{k}.csv", parse_dates=["date"]).set_index("date").sort_index()
    resid = df["residual"].to_numpy(dtype=float)
    n = len(df)

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
        x = r_scaled[t - WINDOW_SIZE:t]                    # the discriminator uses the same 12 residuals
        X[split].append(x)
        y[split].append([r_scaled[t]])
        idx[split].append(t)

    data = {"scaler": scaler, "frame": df}
    for s in ["train", "val", "test"]:
        data[f"X_{s}"] = np.asarray(X[s], dtype="float32")
        data[f"y_{s}"] = np.asarray(y[s], dtype="float32")
        data[f"pos_{s}"] = np.asarray(idx[s])
    return data


# ===================== MODELS ==============================
def build_generator(input_dim, units):
    """GRU generator, same structure as the GRU benchmark: GRU(u) -> GRU(u/2) -> Dense(u) -> Dense(u/2) -> 1.
    The flat input of 12 residuals is reshaped to a (12, 1) sequence."""
    n_features = input_dim // WINDOW_SIZE
    return Sequential([tf.keras.Input(shape=(input_dim,)),
                       Reshape((n_features, WINDOW_SIZE)),
                       Permute((2, 1)),                      # -> (12 months, 1)
                       GRU(units, return_sequences=True),
                       GRU(max(units // 2, 4)),
                       Dense(units, activation="relu"),
                       Dense(max(units // 2, 4), activation="relu"),
                       Dense(1)],
                      name="generator")


def build_discriminator(window_size, d_units=64):
    """Conv1D discriminator (as in GAN_rolling.py): scores a (window_size + 1)-step residual sequence."""
    conv_spec = [(max(d_units // 2, 1), 3), (d_units, 4), (d_units, 5)]
    conv_blocks = [Conv1D(filters, kernel, strides=2, padding="same", activation=LeakyReLU(alpha=0.01))
                   for filters, kernel in conv_spec]
    classifier = [Flatten(),
                  Dense(d_units, use_bias=False), LeakyReLU(),
                  Dense(max(d_units // 2, 1), use_bias=False, activation="relu"),
                  Dense(1, activation="sigmoid")]
    return Sequential([tf.keras.Input(shape=(window_size + 1, 1))] + conv_blocks + classifier,
                      name="discriminator")


# ===================== HYBRID FORECAST + METRICS ===========
def hybrid(data, split, e_hat_scaled):
    """Hybrid forecast on the original scale for the samples of one split."""
    f = data["frame"].iloc[data[f"pos_{split}"]]
    e_hat = data["scaler"].inverse_transform(np.asarray(e_hat_scaled).reshape(-1, 1)).ravel()
    return f.index, f["actual"].to_numpy(), f["sarima_pred"].to_numpy() + e_hat, f["sarima_pred"].to_numpy()


def metrics(y_true, y_pred):
    err = y_true - y_pred
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "MAPE": float(np.mean(np.abs(err / y_true)) * 100)}


# ===================== GAN TRAINER (as in GAN_rolling.py) ===
class GANTrainer:

    def __init__(self, generator, discriminator, lr):
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.opt_d = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @staticmethod
    def with_history(x, next_value):
        """[next residual, previous 12 residuals] as a (13, 1) sequence."""
        next_value = tf.reshape(tf.cast(next_value, tf.float32), [-1, 1, 1])
        history = tf.expand_dims(tf.cast(x[:, :WINDOW_SIZE], tf.float32), -1)
        return tf.concat([next_value, history], axis=1)

    @tf.function
    def step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            forecast = self.generator(x, training=True)
            p_real = self.discriminator(self.with_history(x, y), training=True)
            p_fake = self.discriminator(self.with_history(x, forecast), training=True)
            loss_g = self.bce(tf.ones_like(p_fake), p_fake)
            loss_d = self.bce(tf.ones_like(p_real), p_real) + self.bce(tf.zeros_like(p_fake), p_fake)
        updates = [(tape.gradient(loss, net.trainable_variables), net, opt)
                   for loss, net, opt in ((loss_g, self.generator, self.opt_g),
                                          (loss_d, self.discriminator, self.opt_d))]
        for grads, net, opt in updates:
            opt.apply_gradients(zip(grads, net.trainable_variables))
        del tape
        return loss_d, loss_g


def predict(g_model, X):
    return g_model(X, training=False).numpy()


def train_gan(g_model, d_model, lr, data, patience=PATIENCE):
    """Adversarial training; the generator of the best epoch on the HYBRID validation RMSE is restored."""
    trainer = GANTrainer(g_model, d_model, lr)
    hist = {"d_loss": [], "g_loss": [], "val_rmse": []}
    best_val, best_epoch, best_weights, wait = np.inf, 0, None, 0
    ds = tf.data.Dataset.from_tensor_slices((data["X_train"], data["y_train"])).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        d_losses, g_losses = [], []
        for bx, by in ds:
            d_l, g_l = trainer.step(bx, by)
            d_losses.append(float(d_l))
            g_losses.append(float(g_l))
        _, yv, hv, _ = hybrid(data, "val", predict(g_model, data["X_val"]))
        val_rmse = metrics(yv, hv)["RMSE"]
        hist["d_loss"].append(float(np.mean(d_losses)))
        hist["g_loss"].append(float(np.mean(g_losses)))
        hist["val_rmse"].append(val_rmse)

        if val_rmse < best_val:
            best_val, best_epoch, wait = val_rmse, epoch + 1, 0
            best_weights = g_model.get_weights()
        else:
            wait += 1
        if patience is not None and wait >= patience:
            break

    g_model.set_weights(best_weights)
    return hist, best_epoch


def free_memory():
    tf.keras.backend.clear_session()
    gc.collect()


# ===================== ONE RUN IN A SEPARATE PROCESS =======
# Each run is executed in a fresh Python process, so TensorFlow memory is released after every run.
def _one_run(args):
    origin, units, lr, d_units, seed, final = args
    data = load_origin(origin)
    free_memory()
    set_seed(seed)
    g_model = build_generator(data["X_train"].shape[1], units)
    d_model = build_discriminator(WINDOW_SIZE, d_units)
    hist, best_epoch = train_gan(g_model, d_model, lr, data)

    _, yv, hv, _ = hybrid(data, "val", predict(g_model, data["X_val"]))
    out = {"best_epoch": best_epoch, "val": metrics(yv, hv)}
    if final:   # the test set is evaluated only in the final runs
        dates, yt, ht, st = hybrid(data, "test", predict(g_model, data["X_test"]))
        out.update({"test": metrics(yt, ht), "dates": np.asarray(dates), "actual": yt,
                    "hybrid_test": ht, "sarima_test": st, "hist": hist})
    return out


def run_isolated(origin, units, lr, d_units, seed, final=False):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        return pool.apply(_one_run, ((origin, units, lr, d_units, seed, final),))


# ===================== PLOT ================================
def plot_gan_loss_multi_seed(histories, skip=4,
                             file_dg=f"{RES_DIR}/training_loss_dg_sarima_gan_gru_origin1.pdf",
                             file_val=f"{RES_DIR}/training_val_rmse_sarima_gan_gru_origin1.pdf"):
    """Training of the final seeds on origin 1, without a figure title (the caption is in the manuscript):
    file_dg  : D loss | G loss
    file_val : hybrid validation RMSE, dots = epoch restored by early stopping"""
    # ---- D loss and G loss ----
    _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for seed, (h, best_epoch) in histories.items():
        epochs = np.arange(1, len(h["d_loss"]) + 1)
        axes[0].plot(epochs[skip:], np.asarray(h["d_loss"])[skip:], linewidth=1.2, label=f"seed {seed}")
        axes[1].plot(epochs[skip:], np.asarray(h["g_loss"])[skip:], linewidth=1.2)
    axes[0].axhline(2 * np.log(2), color="grey", linestyle="--", linewidth=1)   # equilibrium of the D loss
    axes[1].axhline(np.log(2), color="grey", linestyle="--", linewidth=1)       # equilibrium of the G loss
    axes[0].set_title("D loss")
    axes[1].set_title("G loss")
    axes[0].legend(ncol=2, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_dg, bbox_inches="tight")
    plt.close()

    # ---- hybrid validation RMSE ----
    _, ax = plt.subplots(figsize=(6.5, 4.5))
    for seed, (h, best_epoch) in histories.items():
        epochs = np.arange(1, len(h["val_rmse"]) + 1)
        line, = ax.plot(epochs, h["val_rmse"], linewidth=1.2, label=f"seed {seed}")
        ax.scatter(best_epoch, h["val_rmse"][best_epoch - 1], color=line.get_color(),
                   edgecolor="black", zorder=5, s=30)
    ax.set_title("Hybrid validation RMSE (\u25cf = best epoch)")
    ax.set_xlabel("Epoch")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_val, bbox_inches="tight")
    plt.close()


# ===================== MAIN ================================
if __name__ == "__main__":
    os.makedirs(RES_DIR, exist_ok=True)
    base = "SARIMA"

    tune = load_origin(TUNE_ORIGIN)
    print(f"Origin {TUNE_ORIGIN} | residual windows train/val/test: "
          f"{len(tune['y_train'])} {len(tune['y_val'])} {len(tune['y_test'])} | input_dim={tune['X_train'].shape[1]}")
    _, yv0, _, sv0 = hybrid(tune, "val", np.zeros(len(tune["y_val"])))
    print(f"{base} alone on origin {TUNE_ORIGIN} validation: RMSE={metrics(yv0, sv0)['RMSE']:.0f}")

    # =================================================================
    # STEP 1: GRID SEARCH on ORIGIN 1 over MULTIPLE seeds - test set NOT used (resume)
    # =================================================================
    if os.path.exists(TUNING_LOG):
        done_df = pd.read_csv(TUNING_LOG)
        done = {(round(float(r.lr), 6), int(r.units), int(r.d_units), int(r.seed)) for r in done_df.itertuples()}
        print(f"Resuming: {len(done)} runs already in {TUNING_LOG}")
    else:
        done = set()

    grid = list(itertools.product(LR_GRID, UNITS_GRID, D_UNITS_GRID))
    total, n = len(grid) * len(TUNE_SEEDS), 0
    print(f"{len(grid)} configurations x {len(TUNE_SEEDS)} seeds = {total} runs")
    for lr, units, d_units in grid:
        for s in TUNE_SEEDS:
            n += 1
            if (round(float(lr), 6), units, d_units, s) in done:
                continue
            t0 = time.time()
            out = run_isolated(TUNE_ORIGIN, units, lr, d_units, s)
            va = out["val"]
            row = {"lr": lr, "units": units, "d_units": d_units, "seed": s, "best_epoch": out["best_epoch"],
                   "val_rmse": va["RMSE"], "val_mae": va["MAE"], "val_mape": va["MAPE"]}
            pd.DataFrame([row]).to_csv(TUNING_LOG, mode="a", index=False, header=not os.path.exists(TUNING_LOG))
            print(f"[{n}/{total}] lr={lr:<7} units={units:<5} d_units={d_units:<4} seed={s:<3} "
                  f"best_epoch={out['best_epoch']:<4} val RMSE={va['RMSE']:.0f}  MAE={va['MAE']:.0f}  "
                  f"MAPE={va['MAPE']:.2f}%  ({time.time() - t0:.0f}s)")

    runs_df = pd.read_csv(TUNING_LOG)
    runs_df = runs_df[runs_df["lr"].round(6).isin([round(x, 6) for x in LR_GRID])
                      & runs_df["units"].isin(UNITS_GRID) & runs_df["d_units"].isin(D_UNITS_GRID)
                      & runs_df["seed"].isin(TUNE_SEEDS)]
    res = (runs_df.groupby(["lr", "units", "d_units"])
           .agg(val_rmse_mean=("val_rmse", "mean"),
                val_rmse_std=("val_rmse", "std"),
                val_rmse_worst=("val_rmse", "max"),
                val_mape_mean=("val_mape", "mean"),
                best_epoch_mean=("best_epoch", "mean"))
           .reset_index())
    res["score"] = res["val_rmse_mean"] + res["val_rmse_std"].fillna(0)   # same criterion as the other models
    res = res.sort_values("score")
    # res.to_csv(f"{RES_DIR}/sarima_gan_gru_tuning_results.csv", index=False)   # not read by the PR_ scripts
    print("\n=== Tuning results on origin 1 (sorted by mean + std of hybrid val RMSE) ===")
    print(res.head(20).round(1).to_string(index=False))

    BEST_LR = float(res.iloc[0]["lr"])
    BEST_UNITS = int(res.iloc[0]["units"])
    BEST_D_UNITS = int(res.iloc[0]["d_units"])
    with open(f"{RES_DIR}/best_hparams_sarima_gan_gru.json", "w") as f:
        json.dump({"lr": BEST_LR, "units": BEST_UNITS, "d_units": BEST_D_UNITS, "loss": "BCE",
                   "tune_origin": TUNE_ORIGIN, "tune_seeds": TUNE_SEEDS, "final_seeds": FINAL_SEEDS,
                   "window": WINDOW_SIZE}, f)
    print(f"\nFixed hyperparameters: lr={BEST_LR}, units={BEST_UNITS}, d_units={BEST_D_UNITS}")
    for name, val, g in [("LR", BEST_LR, LR_GRID), ("units", BEST_UNITS, UNITS_GRID),
                         ("d_units", BEST_D_UNITS, D_UNITS_GRID)]:
        if len(g) > 1 and val in (min(g), max(g)):
            print(f"WARNING: selected {name} = {val} is at the edge of its grid {g}; consider extending it")

    # =================================================================
    # STEP 2: FINAL TRAINING - every origin, final seeds
    # =================================================================
    records, pred_tables, histories_origin1 = [], [], {}
    for k in ORIGINS:
        test_preds = []
        for seed in FINAL_SEEDS:
            out = run_isolated(k, BEST_UNITS, BEST_LR, BEST_D_UNITS, seed, final=True)
            va, te = out["val"], out["test"]
            test_preds.append(out["hybrid_test"])
            if k == ORIGINS[0]:
                histories_origin1[seed] = (out["hist"], out["best_epoch"])
            records.append({"origin": k, "seed": seed, "best_epoch": out["best_epoch"],
                            "val_RMSE": va["RMSE"], "val_MAE": va["MAE"], "val_MAPE": va["MAPE"],
                            "test_RMSE": te["RMSE"], "test_MAE": te["MAE"], "test_MAPE": te["MAPE"]})
            print(f"origin={k} seed={seed} best_epoch={out['best_epoch']:<4} "
                  f"TEST RMSE={te['RMSE']:.0f}  MAE={te['MAE']:.0f}  MAPE={te['MAPE']:.2f}%")

        s_only = metrics(out["actual"], out["sarima_test"])
        print(f"  origin {k}: {base} alone on the same test months: "
              f"RMSE={s_only['RMSE']:.0f}  MAE={s_only['MAE']:.0f}  MAPE={s_only['MAPE']:.2f}%")
        tbl = pd.DataFrame({"origin": k, "date": out["dates"], "actual": out["actual"],
                            "sarima_pred": out["sarima_test"],
                            "seeds_mean_prediction": np.mean(np.stack(test_preds), axis=0)})
        # tbl.to_csv(f"{RES_DIR}/test_predicted_meanseed_sarima_gan_gru_origin{k}.csv", index=False)   # the pooled file below is used instead
        pred_tables.append(tbl)

    per_seed = pd.DataFrame(records)
    per_seed.to_csv(f"{RES_DIR}/sarima_gan_gru_per_seed_results.csv", index=False)   # -> PR_rolling_table.py (MAE per origin)
    # All test forecasts (4 x 12 = 48 rows) -> PR_dm_test_rolling.py and PR_plot_forecasts_origins.py (Fig. 2)
    pd.concat(pred_tables, ignore_index=True).to_csv(f"{RES_DIR}/test_predicted_sarima_gan_gru_pooled.csv", index=False)

    # =================================================================
    # STEP 3: SUMMARY (method 1): per origin mean over seeds, then mean ± std over origins
    # =================================================================
    cols = ["test_MAE", "test_RMSE", "test_MAPE"]
    per_origin = per_seed.groupby("origin")[cols].agg(["mean", "std"])
    # per_origin.to_csv(f"{RES_DIR}/sarima_gan_gru_per_origin_results.csv")   # not read by the PR_ scripts (still printed below)
    print(f"\n=== {base}-GAN (GRU generator) test results per origin (mean ± std over seeds) ===")
    for k in ORIGINS:
        r = per_origin.loc[k]
        print(f"origin {k}: " + "  ".join(f"{c[5:]}={r[(c, 'mean')]:.2f} ± {r[(c, 'std')]:.2f}" for c in cols))

    origin_means = per_seed.groupby("origin")[cols].mean()
    print(f"\n=== {base}-GAN (GRU generator) test over {len(ORIGINS)} origins (mean ± std of the per-origin seed means) ===")
    summary = {}
    for c in cols:
        mean, std = origin_means[c].mean(), origin_means[c].std(ddof=1)
        summary[c[5:]] = {"mean": mean, "std": std}
        print(f"{c[5:]:<5}: {mean:.2f} ± {std:.2f}")
    pd.DataFrame(summary).T.to_csv(f"{RES_DIR}/test_summary_sarima_gan_gru.csv")   # -> PR_rolling_table.py (Table 2)

    # Training curves of origin 1 (Fig. 3): D/G loss and validation RMSE in two files
    plot_gan_loss_multi_seed(histories_origin1)

