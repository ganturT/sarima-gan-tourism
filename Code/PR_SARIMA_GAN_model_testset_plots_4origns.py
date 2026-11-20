import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:           # interactive console / notebook: use the current folder
    CODE_DIR = os.getcwd()
PROJECT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR).lower() == "code" else CODE_DIR
RES_DIR = next((os.path.join(PROJECT_DIR, f) for f in ["Results", "Result"]
                if os.path.isdir(os.path.join(PROJECT_DIR, f))), None)
assert RES_DIR is not None, f"No Results folder was found in {PROJECT_DIR}"

FILE = os.path.join(RES_DIR, "test_predicted_sarima_gan_gru_pooled.csv")
df = pd.read_csv(FILE, parse_dates=["date"])
origins = sorted(df["origin"].unique())

thousands = FuncFormatter(lambda x, _: f"{x:,.0f}")
ymax = max(df["actual"].max(), df["sarima_pred"].max(), df["seeds_mean_prediction"].max()) * 1.05
ymin = 0


def draw(ax, d, show_legend=True):
    ax.plot(d["date"], d["actual"], "k-o", markersize=4, linewidth=1.6, label="Actual")
    ax.plot(d["date"], d["sarima_pred"], "--", color="tab:blue", linewidth=1.5, label="SARIMA")
    ax.plot(d["date"], d["seeds_mean_prediction"], "-", color="tab:orange", linewidth=1.8,
            label="SARIMA-GAN (mean over 10 seeds)")
    ax.set_ylim(ymin, ymax)                      # same scale on every panel
    ax.yaxis.set_major_formatter(thousands)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Tourist arrivals", fontsize=9)
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8, loc="upper left")


# ---------- one figure with four panels ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for ax, k, letter in zip(axes.ravel(), origins, "abcd"):
    d = df[df["origin"] == k].sort_values("date")
    draw(ax, d, show_legend=(k == origins[0]))
    ax.set_title(f"({letter}) Origin {k}: test {d['date'].min():%Y-%m} to {d['date'].max():%Y-%m}", fontsize=10)
plt.tight_layout()
out = os.path.join(RES_DIR, "fig_forecasts_origins.pdf")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ---------- one figure per origin ----------
for k in origins:
    d = df[df["origin"] == k].sort_values("date")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw(ax, d)
    ax.set_xlabel("Month", fontsize=9)
    plt.tight_layout()
    out = os.path.join(RES_DIR, f"fig_forecast_origin{k}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")