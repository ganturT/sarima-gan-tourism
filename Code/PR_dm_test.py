
import os
import numpy as np
import pandas as pd
from scipy import stats


try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:           # interactive console / notebook: use the current folder
    CODE_DIR = os.getcwd()
PROJECT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR).lower() == "code" else CODE_DIR

RES_DIR = next((os.path.join(PROJECT_DIR, f) for f in ["Results", "Result"]
                if os.path.isdir(os.path.join(PROJECT_DIR, f))), None)
assert RES_DIR is not None, f"No Results folder was found in {PROJECT_DIR}"
print(f"Reading the forecasts from: {RES_DIR}")

H = 1                       # one-step-ahead forecasts

# name -> (pooled file, prediction column). Neural models: seed-mean forecast of each origin.
FILES = {
    "SARIMA-GAN (GRU)": ("test_predicted_sarima_gan_gru_pooled.csv", "seeds_mean_prediction"),
    "SARIMA-GRU":       ("test_predicted_sarima_gru_pooled.csv", "seeds_mean_prediction"),
    "SARIMA-GAN (ANN)": ("test_predicted_sarima_gan_pooled.csv", "seeds_mean_prediction"),
    "SARIMA-ANN":       ("test_predicted_sarima_ann_pooled.csv", "seeds_mean_prediction"),
    "GAN (GRU)":        ("test_predicted_gan_gru_pooled.csv", "seeds_mean_prediction"),
    "GRU":              ("test_predicted_gru_pooled.csv", "seeds_mean_prediction"),
    "GAN (ANN)":        ("test_predicted_gan_pooled.csv", "seeds_mean_prediction"),
    "ANN":              ("test_predicted_ann_pooled.csv", "seeds_mean_prediction"),
    "SARIMA":           ("test_predicted_sarima_pooled.csv", "sarima_pred"),
    "Seasonal naive":   ("test_predicted_snaive_pooled.csv", "snaive_pred"),
}

# (model 1, model 2): a negative DM statistic means that model 1 has the smaller loss
PAIRS = [
    ("SARIMA-GAN (GRU)", "SARIMA"),
    ("SARIMA-GAN (GRU)", "SARIMA-GRU"),          # same network: adversarial vs MSE training
    ("SARIMA-GAN (GRU)", "Seasonal naive"),
    ("SARIMA-GAN (GRU)", "SARIMA-GAN (ANN)"),    # same objective: GRU vs ANN generator
    ("SARIMA-GAN (GRU)", "GAN (GRU)"),           # hybrid vs stand-alone GAN with the same generator
    ("SARIMA-GAN (ANN)", "SARIMA-ANN"),          # same network: adversarial vs MSE training
    ("GAN (GRU)", "GRU"),
    ("GAN (ANN)", "ANN"),
    ("SARIMA-GRU", "SARIMA"),
    ("SARIMA-ANN", "SARIMA"),
]


def dm_from_d(d, h=1):
    """DM statistic with the HLN correction from a series of loss differentials d_t.
    Returns (DM*, two-sided p, one-sided p for 'model 1 is more accurate')."""
    d = np.asarray(d, dtype=float)
    T = len(d)
    d_bar = d.mean()
    gamma = [np.sum((d[k:] - d_bar) * (d[:T - k] - d_bar)) / T for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    dm = d_bar / np.sqrt(var_d)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_star = dm * hln
    p_two = 2 * stats.t.sf(abs(dm_star), df=T - 1)
    p_one = stats.t.cdf(dm_star, df=T - 1)
    return dm_star, p_two, p_one


def loss_fn(err, loss):
    return err ** 2 if loss == "squared" else np.abs(err)


def load_pooled(path, column):
    """All 48 forecasts of one model: columns origin, date, actual, pred."""
    df = pd.read_csv(path, parse_dates=["date"])
    assert {"origin", "date", "actual", column} <= set(df.columns), \
        f"{path}: needs origin, date, actual, {column}; has {list(df.columns)}"
    return df[["origin", "date", "actual", column]].rename(columns={column: "pred"})


# ---------------------------------------------------------------------
# Load the forecasts of every model (missing files are skipped)
# ---------------------------------------------------------------------
pooled = {}
for name, (fname, column) in FILES.items():
    path = os.path.join(RES_DIR, fname)
    if not os.path.exists(path):
        print(f"Not found, skipped: {path}")
        continue
    pooled[name] = load_pooled(path, column)
assert pooled, "no forecast file was found; check RES_DIR"

ref = next(iter(pooled.values()))
for name, df in pooled.items():
    m = ref.merge(df, on=["origin", "date"], suffixes=("_r", ""))
    assert len(m) == len(ref) == len(df), f"{name}: origins/months differ from the other models"
    assert np.allclose(m["actual_r"], m["actual"], rtol=1e-6), f"{name}: actual values differ"
months = np.sort(ref["date"].unique())
print(f"\n{len(ref)} forecasts per model, {len(months)} distinct test months "
      f"({pd.Timestamp(months[0]):%Y-%m} -> {pd.Timestamp(months[-1]):%Y-%m}), h = {H}")
assert len(months) == 21, "expected 21 distinct test months; check the pooled files"


def monthly_d(m1, m2, loss):
    """Loss differential of the 21 months for one pair: d of every forecast, averaged per month over origins."""
    # merge on origin and date only: the "actual" columns of different scripts can differ in the last
    # decimals (inverse scaling), and an exact match on them would silently drop months
    a = pooled[m1].merge(pooled[m2][["origin", "date", "pred"]], on=["origin", "date"], suffixes=("_1", "_2"))
    assert len(a) == len(pooled[m1]), f"{m1} vs {m2}: forecasts could not be matched"
    a["d"] = loss_fn(a["actual"] - a["pred_1"], loss) - loss_fn(a["actual"] - a["pred_2"], loss)
    d = a.groupby("date")["d"].mean().sort_index()
    assert len(d) == 21, f"{m1} vs {m2}: {len(d)} months instead of 21"
    return d.to_numpy()


# Accuracy of all 48 forecasts (the forecasts behind the test)
print("\nAccuracy of the 48 forecasts used in the DM test:")
for name, df in pooled.items():
    e = df["actual"] - df["pred"]
    print(f"  {name:<18} MAE {np.mean(np.abs(e)):>10,.2f}  RMSE {np.sqrt(np.mean(e ** 2)):>10,.2f}  "
          f"MAPE {np.mean(np.abs(e / df['actual'])) * 100:6.2f}%")

# ---------------------------------------------------------------------
# DM tests
# ---------------------------------------------------------------------
rows, raw = [], {}
for m1, m2 in PAIRS:
    if m1 not in pooled or m2 not in pooled:
        print(f"Skipped (forecasts missing): {m1} vs {m2}")
        continue
    for loss in ["squared", "absolute"]:
        dm, p_two, p_one = dm_from_d(monthly_d(m1, m2, loss), h=H)
        raw[(m1, m2, loss)] = (dm, p_two)
        rows.append({"comparison": f"{m1} vs {m2}", "loss": loss, "DM": round(dm, 3),
                     "p_two_sided": round(p_two, 4), "p_one_sided": round(p_one, 4)})

res = pd.DataFrame(rows)
print("\nDM < 0: the first model has the smaller loss. p_one_sided tests H1: the first model is more accurate.\n")
print(res.to_string(index=False))
res.to_csv(os.path.join(RES_DIR, "table_dm_test_rolling.csv"), index=False)


# ---------------------------------------------------------------------
# LaTeX table for the Springer Nature template (sn-jnl)
# ---------------------------------------------------------------------
def fmt_dm(x):
    return f"${x:.3f}$"


def fmt_p(p):
    s = "$<$0.001" if p < 0.001 else f"{p:.3f}"
    return rf"\textbf{{{s}}}" if p < 0.05 else s


lines = [
    r"\begin{table}[t]",
    r"\caption{Diebold--Mariano tests with the Harvey--Leybourne--Newbold correction on the 21 distinct "
    r"test months ($h = " + str(H) + r"$). For every month, the loss differential is averaged over all "
    r"rolling origins whose test window contains it. A negative statistic means that the first model has "
    r"the smaller loss; $p$-values are two-sided and values below 0.05 are shown in bold.}\label{tab:dm}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccc}",
    r"\toprule",
    r" & \multicolumn{2}{c}{Squared error} & \multicolumn{2}{c}{Absolute error} \\",
    r"\cmidrule{2-3}\cmidrule{4-5}",
    r"Comparison & DM & $p$ & DM & $p$ \\",
    r"\midrule",
]
for m1, m2 in PAIRS:
    if (m1, m2, "squared") not in raw:
        continue
    dm_sq, p_sq = raw[(m1, m2, "squared")]
    dm_ab, p_ab = raw[(m1, m2, "absolute")]
    lines.append(f"{m1} vs {m2} & {fmt_dm(dm_sq)} & {fmt_p(p_sq)} & {fmt_dm(dm_ab)} & {fmt_p(p_ab)} \\\\")
lines += [r"\botrule", r"\end{tabular*}", r"\end{table}"]

with open(os.path.join(RES_DIR, "table_dm_test_rolling.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved in {RES_DIR}: table_dm_test_rolling.csv and table_dm_test_rolling.tex")
print("\n" + "\n".join(lines))