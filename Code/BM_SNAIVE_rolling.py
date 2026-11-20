import os
import warnings
import pandas as pd
import numpy as np
# import json          # only needed for best_hparams_snaive.json (commented out below)
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


S = 12              # seasonal period (months)
N_VAL = 12          # validation months per origin
N_TEST = 12         # test months per origin

# Rolling origin: first TEST month of each origin (same as the other models)
TEST_STARTS = ["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"]

os.makedirs("Results", exist_ok=True)

df = pd.read_csv('Data/main_dataset.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date').asfreq('MS')

# Monthly series (target)
series = df['volume'].astype(float)
assert series.isna().sum() == 0, "Missing months in volume"

# Seasonal naive: the forecast for each month is the actual value of the same month one year earlier.
# It has no parameters, so it is the same for every origin; only the evaluated months change.
seasonal_naive = series.shift(S)


def make_splits(test_start):
    """Train / validation / test months of one origin."""
    ts = pd.Timestamp(test_start)
    vs = ts - pd.DateOffset(months=N_VAL)       # first validation month
    te = ts + pd.DateOffset(months=N_TEST)      # month after the last test month
    train = series[series.index < vs]
    val = series[(series.index >= vs) & (series.index < ts)]
    test = series[(series.index >= ts) & (series.index < te)]
    assert len(val) == N_VAL and len(test) == N_TEST
    return train, val, test


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true.values - y_pred.values) / y_true.values)) * 100
    return mae, rmse, mape


# -------------------------------------
# 1. Forecasts and metrics for every origin
# -------------------------------------
rows, pred_tables = [], []
for k, test_start in enumerate(TEST_STARTS, 1):
    train, val, test = make_splits(test_start)

    val_pred = seasonal_naive.loc[val.index]
    forecast = seasonal_naive.loc[test.index]
    assert forecast.notna().all(), "Not enough history for the seasonal naive forecast"

    v_mae, v_rmse, v_mape = metrics(val, val_pred)      # for reference only; there is nothing to tune
    mae, rmse, mape = metrics(test, forecast)

    # forecast.rename("predicted_arrivals").to_csv(f"Results/test_predicted_snaive_origin{k}.csv")
    pred_tables.append(pd.DataFrame({"origin": k, "date": test.index, "actual": test.values,
                                     "snaive_pred": forecast.values}))
    rows.append({"origin": k, "test_start": test_start,
                 "train_end": train.index.max().strftime("%Y-%m"),
                 "val_MAE": v_mae, "val_RMSE": v_rmse, "val_MAPE": v_mape,
                 "test_MAE": mae, "test_RMSE": rmse, "test_MAPE": mape})
    print(f"Origin {k} | test {test.index.min():%Y-%m} -> {test.index.max():%Y-%m} | "
          f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")


# -------------------------------------
# 2. Summary over the origins (method 1): mean ± std of the per-origin metrics
# -------------------------------------
summary = pd.DataFrame(rows)
summary.to_csv("Results/snaive_rolling_origin_metrics.csv", index=False)
m = summary[["test_MAE", "test_RMSE", "test_MAPE"]]
print(f"\nSeasonal naive test over {len(TEST_STARTS)} origins (mean ± std):")
for c in m.columns:
    print(f"  {c[5:]:<4}: {m[c].mean():,.2f} ± {m[c].std():,.2f}")

# All test forecasts in one file (4 x 12 = 48 rows) for the DM test later.
# The test windows overlap: before the DM test keep one forecast per month (the latest origin).
pd.concat(pred_tables, ignore_index=True).to_csv("Results/test_predicted_snaive_pooled.csv", index=False)

# with open("Results/best_hparams_snaive.json", "w") as f:
#     json.dump({"season": S, "test_starts": TEST_STARTS,
#                "val_RMSE_per_origin": [float(r["val_RMSE"]) for r in rows]}, f, indent=2)