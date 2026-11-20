import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

# %% --------------------------------------- Settings -----------------------------------------------------------------
DATA_FILE = "Data/main_dataset.csv"
#FEATURES = ["volume", "search", "sin12", "cos12", "covid"]
FEATURES = ["volume"]
TARGET = "volume"

WINDOW_SIZE = 12
HORIZON = 1
N_VAL = 12            # validation months per origin
N_TEST = 12           # test months per origin

# Rolling origin: first TEST month of each origin (shifted by 3 months)
TEST_STARTS = ["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"]

# %% --------------------------------------- Load data ----------------------------------------------------------------
dataset = pd.read_csv(DATA_FILE, parse_dates=["date"]).set_index("date").sort_index()
#dataset.to_csv("Results/dataset.csv")

features = dataset[FEATURES].to_numpy(dtype=float)
target = dataset[[TARGET]].to_numpy(dtype=float)
n_months, N_FEATURES = features.shape

# %% --------------------------------------- Window indices (same for every origin) -----------------------------------
n_samples = n_months - WINDOW_SIZE - HORIZON + 1                    # 94
target_pos = np.arange(n_samples) + WINDOW_SIZE                     # month index of each sample's target
input_rows = np.arange(n_samples)[:, None] + np.arange(WINDOW_SIZE)[None, :]
target_rows = target_pos[:, None] + np.arange(HORIZON)[None, :]
dates_all = dataset.index[target_pos]                               # target month of each sample

# %% --------------------------------------- One split per origin -----------------------------------------------------
for k, test_start in enumerate(TEST_STARTS, 1):
    ts = pd.Timestamp(test_start)
    vs = ts - pd.DateOffset(months=N_VAL)          # first validation month
    te = ts + pd.DateOffset(months=N_TEST)         # month after the last test month

    # Samples are assigned by the month of their TARGET
    is_train = dates_all < vs
    is_val = (dates_all >= vs) & (dates_all < ts)
    is_test = (dates_all >= ts) & (dates_all < te)
    assert is_val.sum() == N_VAL and is_test.sum() == N_TEST, f"origin {k}: not enough months"

    # Scalers fitted on the training months of THIS origin only
    last_train_month = target_pos[np.where(is_train)[0][-1]] + HORIZON - 1
    fit_months = slice(0, last_train_month + 1)
    X_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(features[fit_months])
    y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(target[fit_months])
    X_all = X_scaler.transform(features)[input_rows]
    y_all = y_scaler.transform(target)[target_rows, 0]

    out_dir = f"Data/origin{k}"
    os.makedirs(out_dir, exist_ok=True)
    dump(y_scaler, open(f"{out_dir}/y_scaler.pkl", "wb"))

    print(f"\nOrigin {k}: scaler fitted on {dataset.index[0]:%Y-%m} -> {dataset.index[last_train_month]:%Y-%m}")
    for name, mask in [("train", is_train), ("val", is_val), ("test", is_test)]:
        np.save(f"{out_dir}/X_{name}.npy", X_all[mask])
        np.save(f"{out_dir}/y_{name}.npy", y_all[mask])
        np.save(f"{out_dir}/index_{name}.npy", dates_all[mask])
        d = dates_all[mask]
        print(f"  {name:<5}: {mask.sum():3d} samples | target months "
              f"{d[0]:%Y-%m} -> {d[-1]:%Y-%m} | X {X_all[mask].shape}")

# %% --------------------------------------- Plot the dataset ---------------------------------------------------------
#plt.figure(figsize=(16, 8))
#plt.bar(dataset.index, dataset[TARGET], color="blue", width=10)
#plt.xlabel("Date(month)", fontsize=20)
#plt.ylabel("Number of tourists", fontsize=20)
##plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#plt.grid(axis="y", linestyle="--", alpha=0.7)
#plt.savefig("Results/dataset_plot.pdf")
#plt.show()