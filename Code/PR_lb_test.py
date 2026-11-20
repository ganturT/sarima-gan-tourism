import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:           # interactive console / notebook: use the current folder
    CODE_DIR = os.getcwd()
PROJECT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR).lower() == "code" else CODE_DIR
RES_DIR = next((os.path.join(PROJECT_DIR, f) for f in ["Results", "Result"]
                if os.path.isdir(os.path.join(PROJECT_DIR, f))), None)
assert RES_DIR is not None, f"No Results folder was found in {PROJECT_DIR}"
print(f"Reading the SARIMA residuals from: {RES_DIR}")

ORIGINS = [1, 2, 3, 4]
LAGS = [12, 24]


def acf(x, nlags):
    """Sample autocorrelations r_1 ... r_nlags (same definition as statsmodels' acf)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.sum(x ** 2)
    return np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


def ljung_box(x, h):
    """Ljung-Box Q statistic and p-value for lags 1..h (chi-square with h degrees of freedom)."""
    n = len(x)
    r = acf(x, h)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, h + 1)))
    return float(q), float(chi2.sf(q, df=h))


def as_bool(s):
    """burn_in may be read as bool or as the strings 'True'/'False'."""
    return s.astype(str).str.strip().str.lower().isin(["true", "1"])


rows, resid_origin1 = [], None
for k in ORIGINS:
    path = os.path.join(RES_DIR, f"sarima_origin{k}.csv")
    assert os.path.exists(path), f"Missing: {path}"
    df = pd.read_csv(path, parse_dates=["date"])
    # Training residuals after the burn-in: exactly the residuals the hybrid models learn from
    train = df[(df["split"] == "train") & (~as_bool(df["burn_in"]))]
    r = train["residual"].to_numpy(dtype=float)
    if k == ORIGINS[0]:
        resid_origin1 = r
    row = {"origin": k, "n": len(r),
           "from": train["date"].min().strftime("%Y-%m"), "to": train["date"].max().strftime("%Y-%m")}
    for h in LAGS:
        q, p = ljung_box(r, h)
        row[f"Q({h})"], row[f"p({h})"] = q, p
    rows.append(row)

lb = pd.DataFrame(rows)
lb.to_csv(os.path.join(RES_DIR, "ljung_box_residuals.csv"), index=False)

print("\n=== Ljung-Box test on the SARIMA training residuals (after burn-in) ===")
print(lb.round(3).to_string(index=False))
for h in LAGS:
    sig = lb[lb[f"p({h})"] < 0.05]["origin"].tolist()
    print(f"lag {h}: {len(sig)} of {len(lb)} origins significant at 5% {sig if sig else ''}")


# ---------- LaTeX table for the Springer Nature template (sn-jnl) ----------
def fmt_p(p):
    s = "$<$0.001" if p < 0.001 else f"{p:.3f}"
    return rf"\textbf{{{s}}}" if p < 0.05 else s


lines = [
    r"\begin{table}[t]",
    r"\caption{Ljung--Box test on the SARIMA training residuals after the burn-in period. $n$: number of "
    r"residuals; $Q(h)$: Ljung--Box statistic up to lag $h$; $p$-values below 0.05 are shown in bold.}"
    r"\label{tab:lb}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccccc}",
    r"\toprule",
    r"Origin & Residuals & $n$ & $Q(12)$ & $p$ & $Q(24)$ & $p$ \\",
    r"\midrule",
]
for _, r in lb.iterrows():
    lines.append(f"{int(r['origin'])} & {r['from']} -- {r['to']} & {int(r['n'])} & {r['Q(12)']:.2f} & "
                 f"{fmt_p(r['p(12)'])} & {r['Q(24)']:.2f} & {fmt_p(r['p(24)'])} \\\\")
lines += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
with open(os.path.join(RES_DIR, "ljung_box_residuals.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines))


# ---------- ACF of the origin-1 training residuals ----------
nl = min(24, len(resid_origin1) - 1)
a = acf(resid_origin1, nl)
band = 1.96 / np.sqrt(len(resid_origin1))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.bar(np.arange(1, nl + 1), a, width=0.4, color="tab:blue")
ax.axhline(band, color="grey", linestyle="--", linewidth=1)
ax.axhline(-band, color="grey", linestyle="--", linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lag (months)")
ax.set_ylabel("ACF of SARIMA residuals")
ax.set_title("Origin 1, training residuals after burn-in")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "acf_residuals_origin1.pdf"), bbox_inches="tight")
plt.close()

print(f"\nSaved in {RES_DIR}: ljung_box_residuals.csv, ljung_box_residuals.tex, acf_residuals_origin1.pdf")