import os
# import gc   # only used by free_memory()
import json
import random
import time
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pickle import load
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, LeakyReLU


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


# ===================== SETTINGS ============================

#LR_GRID = [0.0002, 0.0005, 0.001, 0.0016, 0.003, 0.005]   # same as the ANN benchmark
#UNITS_GRID = [16, 32, 64, 128, 256, 512, 1024]             # generator units u
#D_UNITS_GRID = [8, 16, 32, 64, 128, 256]                   # discriminator units d (Conv1D d/2, d, d; Dense d, d/2)

# Selected values (best_hparams_gan.json). With these lines the grid search is skipped:
# the runs are read from TUNING_LOG and only the final training is done
LR_GRID = [0.0016]
UNITS_GRID = [32]
D_UNITS_GRID = [128]

EPOCHS = 300
BATCH_SIZE = 4
PATIENCE = 30
VOLUME_COL = 0   # column of X that holds the (scaled) tourist volume

# Every tuning run is saved here immediately; if the script stops (memory, sleep, Ctrl+C),
# running it again skips the runs already in this file.
TUNING_LOG = "Results/tuning_runs_gan_rolling.csv"


# ===================== DATA ================================
def load_origin(k):
    """Arrays and scaler of one origin. X keeps its 3-D shape (samples, window, features);
    the generator flattens it, and the discriminator uses the arrivals column."""
    d = f"Data/origin{k}"
    data = {}
    for name in ["train", "val", "test"]:
        data[f"X_{name}"] = np.load(f"{d}/X_{name}.npy", allow_pickle=True).astype("float64")
        data[f"y_{name}"] = np.load(f"{d}/y_{name}.npy", allow_pickle=True).astype("float64")
        data[f"index_{name}"] = np.load(f"{d}/index_{name}.npy", allow_pickle=True)
    data["y_scaler"] = load(open(f"{d}/y_scaler.pkl", "rb"))   # fitted on this origin's training period only
    return data


# ===================== MODELS ==============================
def build_generator(window_size, horizon, n_features, units):
    """ANN generator, same structure as the ANN benchmark: Dense(u) -> Dense(u/2) -> output."""
    return Sequential([tf.keras.Input(shape=(window_size, n_features)),
                       Flatten(),
                       Dense(units, activation="relu"),
                       Dense(max(units // 2, 8), activation="relu"),
                       Dense(horizon)],
                      name="generator")


def build_discriminator(window_size, d_units=64):
    """Conv1D discriminator (first study): scores a (window_size + 1)-step arrival sequence as real or generated."""
    conv_spec = [(max(d_units // 2, 1), 3), (d_units, 4), (d_units, 5)]          # (filters, kernel size)
    conv_blocks = [Conv1D(filters, kernel, strides=2, padding="same", activation=LeakyReLU(alpha=0.01))
                   for filters, kernel in conv_spec]
    classifier = [Flatten(),
                  Dense(d_units, use_bias=False), LeakyReLU(),
                  Dense(max(d_units // 2, 1), use_bias=False, activation="relu"),
                  Dense(1, activation="sigmoid")]
    return Sequential([tf.keras.Input(shape=(window_size + 1, 1))] + conv_blocks + classifier,
                      name="discriminator")


# ===================== METRICS (original scale) ============
def compute_metrics(y_true_scaled, y_pred_scaled, y_scaler):
    y_true = y_scaler.inverse_transform(y_true_scaled).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
    err = y_true - y_pred
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "MAPE": float(np.mean(np.abs(err / y_true)) * 100)}


# ===================== GAN TRAINER (first study) ===========
class GANTrainer:

    def __init__(self, generator, discriminator, lr):
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.opt_d = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @staticmethod
    def with_history(x, next_value):
        """[next-month value, past 12 months of arrivals] as a (window + 1, 1) sequence."""
        next_value = tf.cast(tf.reshape(next_value, [-1, tf.shape(next_value)[1], 1]), tf.float64)
        history = tf.cast(x[:, :, VOLUME_COL:VOLUME_COL + 1], tf.float64)
        return tf.concat([next_value, history], axis=1)

    @tf.function
    def step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            forecast = self.generator(x, training=True)
            p_real = self.discriminator(self.with_history(x, y), training=True)
            p_fake = self.discriminator(self.with_history(x, forecast), training=True)

            loss_g = self.bce(tf.ones_like(p_fake), p_fake)
            loss_d = self.bce(tf.ones_like(p_real), p_real) + self.bce(tf.zeros_like(p_fake), p_fake)

        # Compute both gradients from the same forward pass, then update both networks
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
    """Adversarial training; the generator of the best validation-RMSE epoch is restored (never test)."""
    trainer = GANTrainer(g_model, d_model, lr)
    sc = data["y_scaler"]
    hist = {"d_loss": [], "g_loss": [], "val_rmse": []}
    # hist = {"d_loss": [], "g_loss": [], "train_rmse": [], "val_rmse": []}   # train_rmse was never used
    best_val, best_epoch, best_weights, wait = np.inf, 0, None, 0
    ds = tf.data.Dataset.from_tensor_slices((data["X_train"], data["y_train"])).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        d_losses, g_losses = [], []
        for bx, by in ds:
            d_l, g_l = trainer.step(bx, by)
            d_losses.append(float(d_l))
            g_losses.append(float(g_l))

        # tr_m = compute_metrics(data["y_train"], predict(g_model, data["X_train"]), sc)   # not used; slows every epoch
        va_m = compute_metrics(data["y_val"], predict(g_model, data["X_val"]), sc)
        hist["d_loss"].append(float(np.mean(d_losses)))
        hist["g_loss"].append(float(np.mean(g_losses)))
        # hist["train_rmse"].append(tr_m["RMSE"])
        hist["val_rmse"].append(va_m["RMSE"])

        if va_m["RMSE"] < best_val:
            best_val, best_epoch, wait = va_m["RMSE"], epoch + 1, 0
            best_weights = g_model.get_weights()
        else:
            wait += 1
        if patience is not None and wait >= patience:
            break

    g_model.set_weights(best_weights)
    return hist, best_epoch


# Not needed: every run starts in a new process (run_isolated), so no memory is left over
# def free_memory():
#     """Release TensorFlow graphs and Python objects left over from the previous run."""
#     tf.keras.backend.clear_session()
#     gc.collect()


# ===================== ONE RUN IN A SEPARATE PROCESS =======
# Each training run is executed in a fresh Python process that exits afterwards,
# so all TensorFlow memory is returned to the operating system after every run.
def _one_run(args):
    origin, units, lr, d_units, seed, final = args
    data = load_origin(origin)
    sc = data["y_scaler"]
    window_size, n_features, horizon = data["X_train"].shape[1], data["X_train"].shape[2], data["y_train"].shape[1]

    # free_memory()   # not needed: every run starts in a new process
    set_seed(seed)
    g_model = build_generator(window_size, horizon, n_features, units)
    d_model = build_discriminator(window_size, d_units)
    hist, best_epoch = train_gan(g_model, d_model, lr, data)

    out = {"best_epoch": best_epoch, "val": compute_metrics(data["y_val"], predict(g_model, data["X_val"]), sc)}
    if final:   # the test set is evaluated only in the final runs
        yhat_test = predict(g_model, data["X_test"])
        out.update({"train": compute_metrics(data["y_train"], predict(g_model, data["X_train"]), sc),
                    "test": compute_metrics(data["y_test"], yhat_test, sc),
                    "yhat_test": yhat_test,
                    "hist": {"d_loss": hist["d_loss"], "g_loss": hist["g_loss"], "val_rmse": hist["val_rmse"]}})
    return out


def run_isolated(origin, units, lr, d_units, seed, final=False):
    ctx = mp.get_context("spawn")           # works on Windows and Linux
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        return pool.apply(_one_run, ((origin, units, lr, d_units, seed, final),))


# ===================== PLOT ================================
def plot_gan_loss_multi_seed(histories, skip=4, filename="Results/training_loss_all_seeds_gan_origin1.pdf"):
    _, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for seed, (h, best_epoch) in histories.items():
        epochs = np.arange(1, len(h["d_loss"]) + 1)
        axes[0].plot(epochs[skip:], np.asarray(h["d_loss"])[skip:], linewidth=1.2, label=f"seed {seed}")
        axes[1].plot(epochs[skip:], np.asarray(h["g_loss"])[skip:], linewidth=1.2)
        axes[2].plot(epochs, h["val_rmse"], linewidth=1.2)
        axes[2].scatter(best_epoch, h["val_rmse"][best_epoch - 1], edgecolor="black", zorder=5, s=30)
    axes[0].axhline(2 * np.log(2), color="grey", linestyle="--", linewidth=1)   # equilibrium of the D loss
    axes[1].axhline(np.log(2), color="grey", linestyle="--", linewidth=1)       # equilibrium of the G loss
    axes[0].set_title("D loss")
    axes[1].set_title("G loss")
    axes[2].set_title("Validation RMSE of G (● = best epoch)")
    axes[0].legend(ncol=2, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.suptitle("GAN (ANN generator, BCE), origin 1")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# ===================== MAIN ================================
if __name__ == "__main__":
    os.makedirs("Results", exist_ok=True)

    # =================================================================
    # STEP 1: GRID SEARCH on ORIGIN 1 over MULTIPLE seeds - test set NOT used
    # Resume: runs already saved in TUNING_LOG are not repeated.
    # =================================================================
    if os.path.exists(TUNING_LOG):
        done_df = pd.read_csv(TUNING_LOG)
        done = {(round(float(r.lr), 6), int(r.units), int(r.d_units), int(r.seed)) for r in done_df.itertuples()}
        print(f"Resuming: {len(done)} runs already in {TUNING_LOG}")
    else:
        done = set()

    grid = list(itertools.product(LR_GRID, UNITS_GRID, D_UNITS_GRID))
    total, n = len(grid) * len(TUNE_SEEDS), 0
    print(f"Origin {TUNE_ORIGIN}: {len(grid)} configurations x {len(TUNE_SEEDS)} seeds = {total} runs")
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

    # Use only the runs of the current grid (the log may contain other runs)
    runs_df = pd.read_csv(TUNING_LOG)
    runs_df = runs_df[runs_df["lr"].round(6).isin([round(x, 6) for x in LR_GRID])
                      & runs_df["units"].isin(UNITS_GRID) & runs_df["d_units"].isin(D_UNITS_GRID)
                      & runs_df["seed"].isin(TUNE_SEEDS)]

    res = (runs_df.groupby(["lr", "units", "d_units"])
           .agg(val_rmse_mean=("val_rmse", "mean"),
                val_rmse_median=("val_rmse", "median"),
                val_rmse_std=("val_rmse", "std"),
                val_rmse_worst=("val_rmse", "max"),
                val_mape_mean=("val_mape", "mean"),
                best_epoch_mean=("best_epoch", "mean"))
           .reset_index())

    # Selection criterion: mean + std of val RMSE (SAME as the ANN)
    res["score"] = res["val_rmse_mean"] + res["val_rmse_std"].fillna(0)
    res = res.sort_values("score")
    res.to_csv("Results/gan_tuning_results.csv", index=False)
    print("\n=== Tuning results on origin 1 (multi-seed, sorted by mean + std of val RMSE) ===")
    print(res.head(20).round(1).to_string(index=False))

    # Hyperparameters are now FIXED for all origins and seeds
    BEST_LR = float(res.iloc[0]["lr"])
    BEST_UNITS = int(res.iloc[0]["units"])
    BEST_D_UNITS = int(res.iloc[0]["d_units"])
    with open("Results/best_hparams_gan.json", "w") as f:
        json.dump({"lr": BEST_LR, "units": BEST_UNITS, "d_units": BEST_D_UNITS, "loss": "BCE",
                   "tune_origin": TUNE_ORIGIN, "tune_seeds": TUNE_SEEDS, "final_seeds": FINAL_SEEDS}, f)
    print(f"\nFixed hyperparameters: lr={BEST_LR}, units={BEST_UNITS}, d_units={BEST_D_UNITS} "
          f"(selected on origin {TUNE_ORIGIN} over seeds {TUNE_SEEDS}, mean val RMSE={res.iloc[0]['val_rmse_mean']:.0f})")
    for name, val, g in [("LR", BEST_LR, LR_GRID), ("units", BEST_UNITS, UNITS_GRID),
                         ("d_units", BEST_D_UNITS, D_UNITS_GRID)]:
        if len(g) > 1 and val in (min(g), max(g)):
            print(f"WARNING: selected {name} = {val} is at the edge of its grid {g}; consider extending it")

    # =================================================================
    # STEP 2: FINAL TRAINING - same hyperparameters, every origin, seeds 0-9
    # Each origin's test set is evaluated ONCE per seed, after all choices are fixed.
    # =================================================================
    records, pred_tables, histories_origin1 = [], [], {}
    for k in ORIGINS:
        data = load_origin(k)
        sc = data["y_scaler"]
        test_preds = []
        for seed in FINAL_SEEDS:
            out = run_isolated(k, BEST_UNITS, BEST_LR, BEST_D_UNITS, seed, final=True)
            tr, va, te = out["train"], out["val"], out["test"]
            test_preds.append(out["yhat_test"])
            if k == ORIGINS[0]:
                histories_origin1[seed] = (out["hist"], out["best_epoch"])
            records.append({"origin": k, "seed": seed, "best_epoch": out["best_epoch"],
                            "train_RMSE": tr["RMSE"], "train_MAE": tr["MAE"], "train_MAPE": tr["MAPE"],
                            "val_RMSE": va["RMSE"], "val_MAE": va["MAE"], "val_MAPE": va["MAPE"],
                            "test_RMSE": te["RMSE"], "test_MAE": te["MAE"], "test_MAPE": te["MAPE"]})
            print(f"origin={k} seed={seed} best_epoch={out['best_epoch']:<4} "
                  f"TEST RMSE={te['RMSE']:.0f}  MAE={te['MAE']:.0f}  MAPE={te['MAPE']:.2f}%")

        # Seed-mean test prediction of this origin, on the original scale
        mean_pred = sc.inverse_transform(np.mean(np.stack(test_preds, axis=0), axis=0))[:, 0]
        actual = sc.inverse_transform(data["y_test"])[:, 0]
        tbl = pd.DataFrame({"origin": k, "date": data["index_test"], "actual": actual,
                            "seeds_mean_prediction": mean_pred})
        #tbl.to_csv(f"Results/test_predicted_meanseed_gan_origin{k}.csv", index=False)
        pred_tables.append(tbl)

    per_seed = pd.DataFrame(records)
    per_seed.to_csv("Results/gan_per_seed_results.csv", index=False)

    # All test forecasts in one file (4 x 12 = 48 rows); read by PR_dm_test_rolling.py,
    # which averages the loss differential of every month over the origins that contain it.
    pd.concat(pred_tables, ignore_index=True).to_csv("Results/test_predicted_gan_pooled.csv", index=False)

    # =================================================================
    # STEP 3: SUMMARY (method 1)
    #   (a) per origin: mean ± std over the seeds
    #   (b) overall: the 4 per-origin means -> mean ± std over the origins
    # =================================================================
    cols = ["test_MAE", "test_RMSE", "test_MAPE"]
    per_origin = per_seed.groupby("origin")[cols].agg(["mean", "std"])
    per_origin.to_csv("Results/gan_per_origin_results.csv")
    print("\n=== Test results per origin (mean ± std over seeds) ===")
    for k in ORIGINS:
        r = per_origin.loc[k]
        print(f"origin {k}: " + "  ".join(f"{c[5:]}={r[(c, 'mean')]:.2f} ± {r[(c, 'std')]:.2f}" for c in cols))

    origin_means = per_seed.groupby("origin")[cols].mean()
    print(f"\n=== GAN test over {len(ORIGINS)} origins (mean ± std of the per-origin seed means) ===")
    summary = {}
    for c in cols:
        mean, std = origin_means[c].mean(), origin_means[c].std(ddof=1)
        summary[c[5:]] = {"mean": mean, "std": std}
        print(f"{c[5:]:<5}: {mean:.2f} ± {std:.2f}")
    pd.DataFrame(summary).T.to_csv("Results/test_summary_gan.csv")

    # D loss | G loss | validation RMSE over all final seeds, origin 1
    plot_gan_loss_multi_seed(histories_origin1)