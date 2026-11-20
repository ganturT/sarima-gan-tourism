import os
import pandas as pd

try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:           # interactive console / notebook: use the current folder
    CODE_DIR = os.getcwd()
PROJECT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR).lower() == "code" else CODE_DIR
RES_DIR = next((os.path.join(PROJECT_DIR, f) for f in ["Results", "Result"]
                if os.path.isdir(os.path.join(PROJECT_DIR, f))), None)
assert RES_DIR is not None, f"No Results folder was found in {PROJECT_DIR}"
print(f"Reading the results from: {RES_DIR}\n")

METRICS = ["MAE", "RMSE", "MAPE"]
PROPOSED = "SARIMA-GAN (GRU generator, proposed)"

# Neural models: <RES_DIR>/test_summary_<prefix>.csv (rows MAE/RMSE/MAPE, columns mean/std over origins)
NEURAL = {PROPOSED:                     "sarima_gan_gru",
          "SARIMA-GRU (MSE)":           "sarima_gru",
          "SARIMA-GAN (ANN generator)": "sarima_gan",
          "SARIMA-ANN (MSE)":           "sarima_ann",
          "GAN (GRU generator)":        "gan_gru",
          "GRU (MSE)":                  "gru",
          "GAN (ANN generator)":        "gan",
          "ANN (MSE)":                  "ann"}
# Deterministic models: <RES_DIR>/<prefix>_rolling_origin_metrics.csv (one row per origin)
DETERMINISTIC = {"SARIMA":         "sarima",
                 "Seasonal naive": "snaive"}

# Row order of the table, in groups (a \midrule is placed between groups)
GROUPS = [("Hybrid models", [PROPOSED, "SARIMA-GRU (MSE)", "SARIMA-GAN (ANN generator)", "SARIMA-ANN (MSE)"]),
          ("Stand-alone neural models", ["GAN (GRU generator)", "GRU (MSE)", "GAN (ANN generator)", "ANN (MSE)"]),
          ("Benchmarks", ["SARIMA", "Seasonal naive"])]
REFERENCE = "SARIMA"     # changes in the text are reported relative to this model


def neural_summary(prefix):
    path = os.path.join(RES_DIR, f"test_summary_{prefix}.csv")
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path, index_col=0)
    return {m: (float(s.loc[m, "mean"]), float(s.loc[m, "std"])) for m in METRICS}


def deterministic_summary(prefix):
    path = os.path.join(RES_DIR, f"{prefix}_rolling_origin_metrics.csv")
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path)
    return {m: (float(s[f"test_{m}"].mean()), float(s[f"test_{m}"].std(ddof=1))) for m in METRICS}


summ = {}
for name, prefix in NEURAL.items():
    summ[name] = neural_summary(prefix)
for name, prefix in DETERMINISTIC.items():
    summ[name] = deterministic_summary(prefix)

missing = [n for n, v in summ.items() if v is None]
if missing:
    print("Not yet available (skipped):", ", ".join(missing), "\n")

groups = [(t, [n for n in rows if summ[n] is not None]) for t, rows in GROUPS]
groups = [(t, rows) for t, rows in groups if rows]
row_order = [n for _, rows in groups for n in rows]

mean_df = pd.DataFrame({n: {m: summ[n][m][0] for m in METRICS} for n in row_order}).T
best = mean_df.idxmin()          # best (lowest mean) model for every metric


def fmt(name, m, latex=False):
    mu, sd = summ[name][m]
    pm = r" $\pm$ " if latex else " ± "
    if m == "MAPE":
        return f"{mu:.2f}{pm}{sd:.2f}"
    return f"{mu:,.0f}{pm}{sd:,.0f}"        # arrivals: whole numbers, as in the manuscript


# ---------- console / CSV ----------
table = pd.DataFrame({m: [fmt(n, m) for n in row_order] for m in METRICS}, index=row_order)
table.index.name = "Models"
table.columns = ["MAE", "RMSE", "MAPE (%)"]
print(table.to_string())
table.to_csv(os.path.join(RES_DIR, "table_rolling_mean_std.csv"))

# Change of every model relative to the reference model and to the proposed model (for the text)
for ref in [REFERENCE, PROPOSED]:
    if ref not in row_order:
        continue
    print(f"\nChange relative to {ref}:")
    for n in row_order:
        if n == ref:
            continue
        d = {m: (mean_df.loc[n, m] / mean_df.loc[ref, m] - 1) * 100 for m in ["MAE", "RMSE"]}
        dm = mean_df.loc[n, "MAPE"] - mean_df.loc[ref, "MAPE"]
        print(f"  {n:<38} MAE {d['MAE']:+6.1f}%  RMSE {d['RMSE']:+6.1f}%  MAPE {dm:+6.2f} points")

lines = [
    r"\begin{table}[t]",
    r"\caption{Test accuracy on the four rolling origins (mean $\pm$ standard deviation over origins). "
    r"Neural models: per-origin mean over ten seeds. SARIMA and seasonal naive are deterministic. "
    r"The best value of each metric is in bold.}\label{tab:main}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccc}",
    r"\toprule",
    r"Model & MAE & RMSE & MAPE (\%) \\",
    r"\midrule",
]
for i, (title, rows) in enumerate(groups):
    if i > 0:
        lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
    for n in rows:
        cells = []
        for m in METRICS:
            c = fmt(n, m, latex=True)
            cells.append(rf"\textbf{{{c}}}" if best[m] == n else c)
        lines.append(f"{n} & " + " & ".join(cells) + r" \\")
lines += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
with open(os.path.join(RES_DIR, "table_rolling_mean_std.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines))


# ---------- per-origin MAE (shows where the models differ; Extended Data) ----------
per_origin = {}
for name, prefix in NEURAL.items():
    path = os.path.join(RES_DIR, f"{prefix}_per_seed_results.csv")
    if os.path.exists(path):
        per_origin[name] = pd.read_csv(path).groupby("origin")["test_MAE"].mean()
for name, prefix in DETERMINISTIC.items():
    path = os.path.join(RES_DIR, f"{prefix}_rolling_origin_metrics.csv")
    if os.path.exists(path):
        per_origin[name] = pd.read_csv(path).set_index("origin")["test_MAE"]
po = pd.DataFrame(per_origin)[[n for n in row_order if n in per_origin]]
print("\nTest MAE per origin:")
print(po.round(0).T.to_string())
po.round(2).to_csv(os.path.join(RES_DIR, "table_rolling_mae_per_origin.csv"))

# LaTeX: models in rows, origins in columns; the best model of every origin in bold
TEST_WINDOWS = {1: "2024-02", 2: "2024-05", 3: "2024-08", 4: "2024-11"}
origins = list(po.index)
best_o = po.idxmin(axis=1)
lines = [
    r"\begin{table}[h]",
    r"\caption{Test MAE of every model on each rolling origin (neural models: mean over ten seeds). "
    r"The column heads give the first test month of each origin; the lowest MAE of each origin is in bold.}"
    r"\label{tab:mae_origin}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}l" + "c" * len(origins) + "}",
    r"\toprule",
    "Model & " + " & ".join(f"Origin {k} ({TEST_WINDOWS.get(k, '')})" for k in origins) + r" \\",
    r"\midrule",
]
for n in po.columns:
    cells = []
    for k in origins:
        c = f"{po.loc[k, n]:,.0f}"
        cells.append(rf"\textbf{{{c}}}" if best_o[k] == n else c)
    lines.append(f"{n} & " + " & ".join(cells) + r" \\")
lines += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
with open(os.path.join(RES_DIR, "table_rolling_mae_per_origin.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nSaved in {RES_DIR}: table_rolling_mean_std.csv/.tex, table_rolling_mae_per_origin.csv/.tex")