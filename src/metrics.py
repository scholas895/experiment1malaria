"""
================================================================================
 Experiment 2 — SUPER-FIGURE
 Sensitivity + Specificity + Precision + F1 in ONE combined figure
================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
IN_DIR  = "Experiment2_Outputs/Tables"
OUT_DIR = "Experiment2_Outputs"
FIG_DIR = os.path.join(OUT_DIR, "SuperFigures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 13,
                     "axes.labelsize": 12, "figure.dpi": 300, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})

MODELS = ["SARIMA", "Random Forest", "XGBoost", "LSTM", "TFT",
          "LSTM+TFT", "Ensemble"]

METRICS = ["Sensitivity", "Specificity", "Precision", "F1"]
LABELS  = {"Sensitivity": "Sensitivity (Recall)",
           "Specificity": "Specificity",
           "Precision":   "Precision",
           "F1":          "F1-score"}

PALETTE = {
    "SARIMA":        "#8c564b",
    "Random Forest": "#1f77b4",
    "XGBoost":       "#ff7f0e",
    "LSTM":          "#2ca02c",
    "TFT":           "#d62728",
    "LSTM+TFT":      "#9467bd",
    "Ensemble":      "#111111",
}


# -----------------------------------------------------------------------------
# Load metrics
# -----------------------------------------------------------------------------
def load_all():
    dfs = {}
    for h in (1, 2, 3):
        f = os.path.join(IN_DIR, f"Baseline_Metrics_h{h}.csv")
        if not os.path.exists(f):
            print(f"  [skip] {f} not found");  continue
        df = pd.read_csv(f, index_col=0)
        df = df.loc[[m for m in MODELS if m in df.index]]
        dfs[h] = df
    return dfs


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  saved {name}.png + .pdf")


# =============================================================================
# VARIANT A — 3 rows (h=1,2,3) x 4 columns (metrics)
# =============================================================================
def super_figure_A(dfs):
    horizons = sorted(dfs.keys())
    nrow, ncol = len(horizons), len(METRICS)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(4.2 * ncol, 3.6 * nrow),
                             sharey=True)

    for r, h in enumerate(horizons):
        df = dfs[h]
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            if metric not in df.columns:
                ax.axis("off");  continue
            vals   = df[metric].astype(float)
            colors = [PALETTE.get(m, "gray") for m in vals.index]
            bars = ax.bar(range(len(vals)), vals.values,
                          color=colors, edgecolor="black", linewidth=0.6)

            # Value labels
            for b, v in zip(bars, vals.values):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width()/2, v,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(range(len(vals)))
            if r == nrow - 1:
                ax.set_xticklabels(vals.index, rotation=35,
                                   ha="right", fontsize=9)
            else:
                ax.set_xticklabels([])

            if r == 0:
                ax.set_title(LABELS[metric], fontsize=12)
            if c == 0:
                ax.set_ylabel(f"h = {h}\nValue", fontsize=10)

            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Early-warning detection metrics — Sensitivity, Specificity, "
                 "Precision, F1\n(rows = forecast horizon, columns = metric)",
                 fontsize=14, y=1.005)
    plt.tight_layout()
    save_fig(fig, "SuperFigure_A_Detection_3x4")


# =============================================================================
# VARIANT B — 4 rows (metrics) x 3 columns (horizons)
# =============================================================================
def super_figure_B(dfs):
    horizons = sorted(dfs.keys())
    nrow, ncol = len(METRICS), len(horizons)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(5.5 * ncol, 3.4 * nrow),
                             sharey=True)

    for r, metric in enumerate(METRICS):
        for c, h in enumerate(horizons):
            ax = axes[r, c]
            df = dfs[h]
            if metric not in df.columns:
                ax.axis("off");  continue
            vals   = df[metric].astype(float)
            colors = [PALETTE.get(m, "gray") for m in vals.index]
            bars = ax.bar(range(len(vals)), vals.values,
                          color=colors, edgecolor="black", linewidth=0.6)

            for b, v in zip(bars, vals.values):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width()/2, v,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(range(len(vals)))
            if r == nrow - 1:
                ax.set_xticklabels(vals.index, rotation=35,
                                   ha="right", fontsize=9)
            else:
                ax.set_xticklabels([])

            if r == 0:
                ax.set_title(f"h = {h} month(s)", fontsize=12)
            if c == 0:
                ax.set_ylabel(LABELS[metric], fontsize=11)

            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Detection metrics across horizons — "
                 "Sensitivity, Specificity, Precision, F1",
                 fontsize=14, y=1.005)
    plt.tight_layout()
    save_fig(fig, "SuperFigure_B_Detection_4x3")


# =============================================================================
# VARIANT C — Grouped bars: 1 panel per metric, 3 horizons side-by-side per model
# =============================================================================
def super_figure_C(dfs):
    horizons = sorted(dfs.keys())
    fig, axes = plt.subplots(1, 4, figsize=(5.5 * 4, 5), sharey=True)

    for ax, metric in zip(axes, METRICS):
        data = []
        for h in horizons:
            df = dfs[h]
            if metric not in df.columns:  continue
            for m in df.index:
                try:
                    v = float(df.loc[m, metric])
                except Exception:
                    v = np.nan
                data.append({"Model": m, "Horizon": f"h={h}", "Value": v})
        d = pd.DataFrame(data)

        sns.barplot(data=d, x="Model", y="Value", hue="Horizon",
                    palette="Set2", edgecolor="black", ax=ax)
        ax.set_title(LABELS[metric], fontsize=12)
        ax.set_xlabel("");  ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
        if metric == METRICS[0]:
            ax.legend(title="Horizon", fontsize=8, loc="upper right")
        else:
            ax.legend_.remove()

    fig.suptitle("Detection metrics grouped by horizon — "
                 "Sensitivity, Specificity, Precision, F1",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, "SuperFigure_C_GroupedByHorizon")


# =============================================================================
# VARIANT D — Radar: 1 radar per horizon, spokes = 4 metrics
# =============================================================================
def super_figure_D(dfs):
    horizons = sorted(dfs.keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 6.5),
                             subplot_kw=dict(polar=True))
    if len(horizons) == 1:  axes = [axes]

    angles = np.linspace(0, 2*np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    for ax, h in zip(axes, horizons):
        df = dfs[h]
        for m in df.index:
            try:
                vals = [float(df.loc[m, k]) for k in METRICS]
            except Exception:
                continue
            if any(np.isnan(vals)):
                continue
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=1.8,
                    label=m, color=PALETTE.get(m, None))
            ax.fill(angles, vals, alpha=0.07, color=PALETTE.get(m, None))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([LABELS[k] for k in METRICS], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"h = {h} month(s)", fontsize=12, pad=18)

    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.45, 1.10), fontsize=9)
    fig.suptitle("Detection-metric profile per model (radar)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, "SuperFigure_D_Radar")


# =============================================================================
# VARIANT E — Heatmap: rows=models, cols=metrics x horizon
# =============================================================================
def super_figure_E(dfs):
    horizons = sorted(dfs.keys())
    cols, data = [], []
    for m in MODELS:
        row = {}
        for h in horizons:
            df = dfs[h]
            if m not in df.index:  continue
            for metric in METRICS:
                if metric not in df.columns:  continue
                try:
                    row[f"{metric}\nh={h}"] = float(df.loc[m, metric])
                except Exception:
                    row[f"{metric}\nh={h}"] = np.nan
        data.append(row)
        cols = list(row.keys())
    heat = pd.DataFrame(data, index=MODELS)[cols]

    fig, ax = plt.subplots(figsize=(1.4 * len(cols), 4.8))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={"label": "Value"}, ax=ax,
                linewidths=0.4, linecolor="white",
                vmin=0, vmax=1)
    ax.set_title("Detection metrics × horizon heatmap (Sensitivity / "
                 "Specificity / Precision / F1)")
    ax.set_xlabel("");  ax.set_ylabel("Model")
    plt.tight_layout()
    save_fig(fig, "SuperFigure_E_Heatmap")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  Super-figure — Sensitivity + Specificity + Precision + F1")
    print("=" * 70)

    dfs = load_all()
    if not dfs:
        print("No Baseline_Metrics files found.")
        return
    print(f"  Loaded horizons: {list(dfs.keys())}")

    print("\n[A] 3 rows × 4 columns grid ...")
    super_figure_A(dfs)

    print("\n[B] 4 rows × 3 columns grid ...")
    super_figure_B(dfs)

    print("\n[C] Grouped by horizon ...")
    super_figure_C(dfs)

    print("\n[D] Radar per horizon ...")
    super_figure_D(dfs)

    print("\n[E] Heatmap models × (metrics × horizons) ...")
    super_figure_E(dfs)

    print(f"\nAll super-figures saved to: {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()