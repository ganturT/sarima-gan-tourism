import os
import warnings
import itertools
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

S = 12              # seasonal period (months)
N_VAL = 12          # validation months per origin
N_TEST = 12         # test months per origin

# Rolling origin: first TEST month of each origin (same as Data_preprocessing_rolling.py)
TEST_STARTS = ["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"]

os.makedirs("Results", exist_ok=True)

df = pd.read_csv('Data/main_dataset.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date').asfreq('MS')

# Monthly series (target)
series = df['volume'].astype(float)
assert series.isna().sum() == 0, "Missing months in volume"


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


# -------------------------------------
# 1. Rolling one-step-ahead forecast
#    Parameters are estimated once on the fit data; after each forecast the
#    actual value is added to the model state (no re-estimation).
# -------------------------------------
def rolling_forecast(res, eval_y):
    preds = []
    for t in range(len(eval_y)):
        preds.append(float(res.forecast(steps=1).iloc[0]))
        res = res.append(eval_y.iloc[t:t + 1], refit=False)
    return pd.Series(preds, index=eval_y.index), res


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true.values - y_pred.values) / y_true.values)) * 100
    return mae, rmse, mape


# -------------------------------------
# 2. Hyperparameter tuning (p, d, q)(P, D, Q, 12) on the VALIDATION set of ORIGIN 1 only.
#    The selected order is then fixed for all origins. No test set is used here.
# -------------------------------------
p_grid = [0, 1, 2]
d_grid = [0, 1]
q_grid = [0, 1, 2]
P_grid = [0, 1]
D_grid = [0, 1]
Q_grid = [0, 1]

train1, val1, _ = make_splits(TEST_STARTS[0])
print("Order selection on origin 1")
print("Train:", train1.index.min().date(), "->", train1.index.max().date(), f"({len(train1)})")
print("Val:  ", val1.index.min().date(), "->", val1.index.max().date(), f"({len(val1)})")

results = []
for p, d, q, P, D, Q in itertools.product(p_grid, d_grid, q_grid, P_grid, D_grid, Q_grid):
    if p == q == P == Q == 0:
        continue                                     # no AR / MA term at all
    order, seasonal_order = (p, d, q), (P, D, Q, S)
    try:
        res = ARIMA(train1, order=order, seasonal_order=seasonal_order).fit()
        pred, _ = rolling_forecast(res, val1)
        rmse = np.sqrt(mean_squared_error(val1, pred))
        results.append((order, seasonal_order, rmse))
        print(f"SARIMA{order}x{seasonal_order}  val RMSE = {rmse:,.0f}")
    except Exception as e:
        print(f"SARIMA{order}x{seasonal_order}  failed: {e}")

best_order, best_seasonal, best_rmse = min(results, key=lambda r: r[2])
print("\nBest order:", best_order, "x", best_seasonal, "| origin-1 validation RMSE:", round(best_rmse, 2))
# pd.DataFrame([(str(o), str(so), r) for o, so, r in sorted(results, key=lambda r: r[2])],
#              columns=["order", "seasonal_order", "val_RMSE"]).to_csv("Results/sarima_tuning_results.csv", index=False)



rows = []
for k, test_start in enumerate(TEST_STARTS, 1):
    train, val, test = make_splits(test_start)

    res = ARIMA(train, order=best_order, seasonal_order=best_seasonal).fit()
    train_pred = res.predict(start=train.index[0], end=train.index[-1])    # in-sample one step
    val_pred, res = rolling_forecast(res, val)       # validation -> forecasts + model state
    test_pred, res = rolling_forecast(res, test)     # test, one month at a time


    burn = best_order[1] + best_seasonal[1] * S
    out = pd.concat([
        pd.DataFrame({"actual": train, "sarima_pred": train_pred, "split": "train"}),
        pd.DataFrame({"actual": val, "sarima_pred": val_pred, "split": "val"}),
        pd.DataFrame({"actual": test, "sarima_pred": test_pred, "split": "test"}),
    ])
    out["residual"] = out["actual"] - out["sarima_pred"]
    out["burn_in"] = False
    out.iloc[:burn, out.columns.get_loc("burn_in")] = True
    out.index.name = "date"
    out.to_csv(f"Results/sarima_origin{k}.csv")

    # Test forecasts in the same file format as before
    # forecast = test_pred.rename("predicted_arrivals")
    # forecast.to_csv(f"Results/test_predicted_sarima_origin{k}.csv")

    mae, rmse, mape = metrics(test, test_pred)
    v_mae, v_rmse, v_mape = metrics(val, val_pred)
    rows.append({"origin": k, "test_start": test_start,
                 "train_end": train.index.max().strftime("%Y-%m"),
                 "val_MAE": v_mae, "val_RMSE": v_rmse, "val_MAPE": v_mape,
                 "test_MAE": mae, "test_RMSE": rmse, "test_MAPE": mape})
    print(f"Origin {k} | train -> {train.index.max():%Y-%m} | test {test.index.min():%Y-%m} -> "
          f"{test.index.max():%Y-%m} | MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")


# -------------------------------------
# 4. Summary over the origins (method 1): metrics per origin, then mean ± std over the 4 origins
# -------------------------------------
summary = pd.DataFrame(rows)
summary.to_csv("Results/sarima_rolling_origin_metrics.csv", index=False)
m = summary[["test_MAE", "test_RMSE", "test_MAPE"]]
print(f"\nSARIMA test over {len(TEST_STARTS)} origins (mean ± std):")
for c in m.columns:
    print(f"  {c[5:]:<4}: {m[c].mean():,.2f} ± {m[c].std():,.2f}")


pooled = pd.concat(
    [pd.read_csv(f"Results/sarima_origin{k}.csv", parse_dates=["date"]).query("split == 'test'").assign(origin=k)
     for k in range(1, len(TEST_STARTS) + 1)],
    ignore_index=True)
pooled[["origin", "date", "actual", "sarima_pred", "residual"]].to_csv(
    "Results/test_predicted_sarima_pooled.csv", index=False)

with open("Results/best_hparams_sarima.json", "w") as f:
    json.dump({"order": str(best_order), "seasonal_order": str(best_seasonal),
               "exogenous": None, "selected_on": "origin 1 validation",
               "origin1_val_RMSE": float(best_rmse), "test_starts": TEST_STARTS}, f, indent=2)