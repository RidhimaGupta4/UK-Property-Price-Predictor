"""
UK Property Price Predictor — EDA & Chart Generation
=====================================================
Generates 8 publication-quality charts.

Run:
    python 04_eda_charts.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings, os

warnings.filterwarnings("ignore")

DATA = "/home/claude/uk-property-predictor/data/processed"
OUT  = "/home/claude/uk-property-predictor/outputs"
os.makedirs(OUT, exist_ok=True)

props   = pd.read_csv(f"{DATA}/properties.csv")
preds   = pd.read_csv(f"{DATA}/test_predictions.csv")
fi      = pd.read_csv(f"{DATA}/feature_importance.csv")
reg_acc = pd.read_csv(f"{DATA}/regional_accuracy.csv")

# ── Palette ───────────────────────────────────────────────────────────────────
C1   = "#1D3461"   # Dark navy
C2   = "#1F7A8C"   # Teal
C3   = "#E84855"   # Red
C4   = "#F4A261"   # Amber
C5   = "#2EC4B6"   # Turquoise
GRID = "#E8EDF2"
BG   = "#F8FAFC"
TEXT = "#1A202C"
MUTED= "#718096"

def fmt_gbp(x, _): return f"£{x/1000:.0f}k"

def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.get_figure().patch.set_facecolor("white")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="--")
    ax.grid(axis="x", visible=False)
    if title:  ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, pad=12)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=MUTED)


# ── Chart 1: Regional Median Prices ──────────────────────────────────────────
def chart_regional_prices():
    reg = props.groupby("region")["price_gbp"].median().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    colors = [C3 if "London" in r else C1 for r in reg.index]
    bars = ax.barh(reg.index, reg.values, color=colors, height=0.65, alpha=0.88)

    for bar, val in zip(bars, reg.values):
        ax.text(val + 3000, bar.get_y() + bar.get_height()/2,
                f"£{val/1000:.0f}k", va="center", fontsize=9,
                color=TEXT, fontweight="bold")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_gbp))
    style(ax, title="Median House Price by UK Region\nLand Registry Price Paid Data 2019–2024",
          xlabel="Median Price (£)")
    ax.set_xlim(0, reg.max() * 1.18)
    legend = [mpatches.Patch(color=C3, label="London"),
              mpatches.Patch(color=C1, label="Rest of England & Wales")]
    ax.legend(handles=legend, fontsize=9, framealpha=0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/01_regional_prices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 01_regional_prices.png")


# ── Chart 2: Price by Property Type ──────────────────────────────────────────
def chart_price_by_type():
    type_order = ["Detached", "Semi-Detached", "Terraced", "Flat"]
    type_data  = props.groupby("property_type")["price_gbp"]
    medians    = type_data.median().reindex(type_order)
    q25        = type_data.quantile(0.25).reindex(type_order)
    q75        = type_data.quantile(0.75).reindex(type_order)
    counts     = type_data.count().reindex(type_order)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    colors = [C1, C2, C4, C5]
    x = np.arange(len(type_order))
    bars = ax.bar(x, medians, color=colors, width=0.55, alpha=0.88)
    ax.errorbar(x, medians, yerr=[medians - q25, q75 - medians],
                fmt="none", color=TEXT, capsize=8, capthick=2, linewidth=2, alpha=0.6)

    for i, (bar, med, n) in enumerate(zip(bars, medians, counts)):
        ax.text(bar.get_x()+bar.get_width()/2, med + 8000,
                f"£{med/1000:.0f}k\n({n:,} sales)", ha="center",
                fontsize=9, fontweight="bold", color=TEXT)

    ax.set_xticks(x); ax.set_xticklabels(type_order, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_gbp))
    style(ax, title="Median Price by Property Type\nBars show median · Error bars show IQR (25th–75th percentile)",
          ylabel="Price (£)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_price_by_type.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 02_price_by_type.png")


# ── Chart 3: Feature Importance ───────────────────────────────────────────────
def chart_feature_importance():
    top15 = fi.head(15).sort_values("importance_pct")
    labels = top15["feature"].str.replace("_", " ").str.title()
    colors = [C3 if v > 15 else C1 if v > 5 else C2
              for v in top15["importance_pct"]]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    bars = ax.barh(labels, top15["importance_pct"], color=colors, height=0.65, alpha=0.88)
    for bar, val in zip(bars, top15["importance_pct"]):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9, color=TEXT, fontweight="bold")

    style(ax, title="Gradient Boosting — Feature Importance\nTop 15 predictors of UK house price",
          xlabel="Importance (%)")
    legend = [mpatches.Patch(color=C3,  label="Critical (>15%)"),
              mpatches.Patch(color=C1,  label="High (5–15%)"),
              mpatches.Patch(color=C2,  label="Medium (<5%)")]
    ax.legend(handles=legend, fontsize=9, framealpha=0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 03_feature_importance.png")


# ── Chart 4: Actual vs Predicted scatter ──────────────────────────────────────
def chart_actual_vs_predicted():
    sample = preds.sample(min(800, len(preds)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")

    scatter = ax.scatter(
        sample["actual_price"] / 1000,
        sample["predicted_price"] / 1000,
        c=sample["error_pct"].abs(),
        cmap="RdYlGn_r", alpha=0.55, s=20,
        vmin=0, vmax=25
    )
    max_val = max(sample["actual_price"].max(), sample["predicted_price"].max()) / 1000
    ax.plot([0, max_val], [0, max_val], color=TEXT, linewidth=1.5,
            linestyle="--", alpha=0.6, label="Perfect prediction")
    ax.plot([0, max_val], [0, max_val*1.1], color=C4, linewidth=1,
            linestyle=":", alpha=0.5, label="+10% band")
    ax.plot([0, max_val], [0, max_val*0.9], color=C4, linewidth=1,
            linestyle=":", alpha=0.5)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Absolute % Error", fontsize=9, color=MUTED)
    cbar.ax.tick_params(labelsize=8)

    r2 = np.corrcoef(sample["actual_price"], sample["predicted_price"])[0,1]**2
    mae = (sample["predicted_price"] - sample["actual_price"]).abs().mean()
    ax.text(0.05, 0.92, f"R² = {r2:.4f}\nMAE = £{mae:,.0f}",
            transform=ax.transAxes, fontsize=11, color=C1,
            fontweight="bold", bbox=dict(facecolor="white", alpha=0.8, edgecolor=GRID))

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}k"))
    style(ax, title="Actual vs Predicted House Price\nGradient Boosting model · Test set (20% holdout)",
          xlabel="Actual Price (£)", ylabel="Predicted Price (£)")
    ax.legend(fontsize=9, framealpha=0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/04_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 04_actual_vs_predicted.png")


# ── Chart 5: Year-on-Year Price Trend ─────────────────────────────────────────
def chart_yoy_trend():
    yearly = props.groupby(["sale_year","property_type"])["price_gbp"].median().reset_index()
    type_order = ["Detached","Semi-Detached","Terraced","Flat"]
    colors_map = {"Detached":C1,"Semi-Detached":C2,"Terraced":C4,"Flat":C5}

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    for pt in type_order:
        sub = yearly[yearly["property_type"]==pt].sort_values("sale_year")
        ax.plot(sub["sale_year"], sub["price_gbp"]/1000,
                color=colors_map[pt], linewidth=2.5, marker="o",
                markersize=7, label=pt)
        ax.fill_between(sub["sale_year"], sub["price_gbp"]/1000,
                        alpha=0.06, color=colors_map[pt])

    ax.axvspan(2020.7, 2022.3, alpha=0.06, color=C4)
    ax.text(2021.5, ax.get_ylim()[1]*0.97, "COVID\nBoom", ha="center",
            fontsize=8, color=C4, fontweight="bold")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}k"))
    ax.set_xticks(props["sale_year"].unique())
    style(ax, title="Median House Price Trend by Property Type 2019–2024\nPost-COVID boom, 2023 correction, and 2024 recovery",
          xlabel="Year", ylabel="Median Price (£)")
    ax.legend(fontsize=10, framealpha=0, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_yoy_price_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 05_yoy_price_trend.png")


# ── Chart 6: EPC & School Rating Price Premiums ────────────────────────────────
def chart_premiums():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    # EPC
    epc_order = ["A","B","C","D","E","F","G"]
    epc = props.groupby("epc_rating")["price_gbp"].median().reindex(epc_order).dropna()
    colors_epc = [C2 if r in ["A","B","C"] else C4 if r == "D" else C3
                  for r in epc.index]
    bars = ax1.bar(epc.index, epc.values/1000, color=colors_epc, width=0.6, alpha=0.88)
    for bar, val in zip(bars, epc.values):
        ax1.text(bar.get_x()+bar.get_width()/2, val/1000+2,
                 f"£{val/1000:.0f}k", ha="center", fontsize=9,
                 fontweight="bold", color=TEXT)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}k"))
    style(ax1, title="EPC Rating vs Median Price\nGreen homes command a premium",
          xlabel="EPC Rating", ylabel="Median Price (£)")
    ax1.set_facecolor(BG)

    # Ofsted
    school_order = ["Outstanding","Good","Requires Improvement","Inadequate"]
    school = props.groupby("nearest_school_ofsted")["price_gbp"].median().reindex(school_order).dropna()
    labels_short = ["Outstanding","Good","Req. Improvement","Inadequate"]
    colors_sch = [C2, C1, C4, C3]
    bars2 = ax2.bar(range(len(school)), school.values/1000,
                    color=colors_sch, width=0.6, alpha=0.88)
    for bar, val in zip(bars2, school.values):
        ax2.text(bar.get_x()+bar.get_width()/2, val/1000+2,
                 f"£{val/1000:.0f}k", ha="center", fontsize=9,
                 fontweight="bold", color=TEXT)
    ax2.set_xticks(range(len(school)))
    ax2.set_xticklabels(labels_short, rotation=15, ha="right", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}k"))
    style(ax2, title="Ofsted School Rating vs Median Price\nOutstanding schools add significant premium",
          ylabel="Median Price (£)")

    plt.suptitle("The Green Premium & School Premium — Feature Price Analysis",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/06_epc_school_premiums.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_epc_school_premiums.png")


# ── Chart 7: Model Comparison ──────────────────────────────────────────────────
def chart_model_comparison():
    models = ["Ridge\nRegression", "Random\nForest", "Gradient\nBoosting\n(XGBoost-style)"]
    r2     = [0.934, 0.965, 0.987]
    rmse   = [74872, 54398, 32738]
    mae    = [42143, 33551, 20726]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), facecolor="white")
    colors = [C4, C2, C1]

    def annotate_bars(ax, bars, vals, fmt):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    fmt(val), ha="center", fontsize=10, fontweight="bold", color=TEXT)

    b1 = ax1.bar(models, r2, color=colors, width=0.5, alpha=0.88)
    annotate_bars(ax1, b1, r2, lambda v: f"{v:.3f}")
    ax1.set_ylim(0.85, 1.01)
    ax1.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    style(ax1, title="R² Score\n(Higher = Better, max 1.0)", ylabel="R²")

    b2 = ax2.bar(models, [r/1000 for r in rmse], color=colors, width=0.5, alpha=0.88)
    annotate_bars(ax2, b2, [r/1000 for r in rmse], lambda v: f"£{v:.0f}k")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}k"))
    style(ax2, title="RMSE\n(Lower = Better)", ylabel="Root Mean Square Error (£)")

    b3 = ax3.bar(models, [m/1000 for m in mae], color=colors, width=0.5, alpha=0.88)
    annotate_bars(ax3, b3, [m/1000 for m in mae], lambda v: f"£{v:.0f}k")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}k"))
    style(ax3, title="MAE\n(Lower = Better)", ylabel="Mean Absolute Error (£)")

    plt.suptitle("Model Comparison — Ridge vs Random Forest vs Gradient Boosting",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/07_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_model_comparison.png")


# ── Chart 8: Commute Premium & Price Distribution ─────────────────────────────
def chart_commute_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    # Commute vs price
    bins = [0, 15, 30, 45, 60, 75, 120]
    labels = ["0–15", "16–30", "31–45", "46–60", "61–75", "75+"]
    props["commute_band"] = pd.cut(props["commute_mins"], bins=bins, labels=labels)
    comm = props.groupby("commute_band", observed=True)["price_gbp"].median()

    ax1.plot(range(len(comm)), comm.values/1000, color=C1, linewidth=2.5,
             marker="o", markersize=9)
    ax1.fill_between(range(len(comm)), comm.values/1000, alpha=0.12, color=C1)
    for i, (lab, val) in enumerate(zip(comm.index, comm.values)):
        ax1.annotate(f"£{val/1000:.0f}k",
                     (i, val/1000), xytext=(0, 12),
                     textcoords="offset points", ha="center",
                     fontsize=9, fontweight="bold", color=C1)
    ax1.set_xticks(range(len(comm)))
    ax1.set_xticklabels([f"{l} min" for l in comm.index], fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}k"))
    style(ax1, title="Commute Time vs Median Price\nEvery 15 minutes = significant price reduction",
          xlabel="Commute Band", ylabel="Median Price (£)")

    # Price distribution histogram
    price_data = props["price_gbp"] / 1000
    ax2.hist(price_data[price_data <= 800], bins=60, color=C1, alpha=0.75,
             edgecolor="white", linewidth=0.3)
    ax2.axvline(price_data.median(), color=C3, linewidth=2, linestyle="--",
                label=f"Median £{price_data.median():.0f}k")
    ax2.axvline(price_data.mean(), color=C4, linewidth=2, linestyle="-.",
                label=f"Mean £{price_data.mean():.0f}k")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}k"))
    style(ax2, title="Price Distribution — England & Wales\n(capped at £800k for readability)",
          xlabel="Price (£)", ylabel="Number of Properties")
    ax2.legend(fontsize=9, framealpha=0)

    plt.tight_layout()
    plt.savefig(f"{OUT}/08_commute_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_commute_distribution.png")


if __name__ == "__main__":
    print("Generating property price analysis charts...\n")
    chart_regional_prices()
    chart_price_by_type()
    chart_feature_importance()
    chart_actual_vs_predicted()
    chart_yoy_trend()
    chart_premiums()
    chart_model_comparison()
    chart_commute_distribution()
    print(f"\nAll 8 charts saved to {OUT}/")
