"""
================================================================================
 CPST-MalariaNet — MasterGrid_AllMetrics  (v2 filename, legend + safe save)
================================================================================
 Output:
   Experiment2_Outputs/CombinedFigures/MasterGrid_AllMetrics_v2.png
   Experiment2_Outputs/CombinedFigures/MasterGrid_AllMetrics_v2.pdf
================================================================================
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
IN_DIR  = "Experiment2_Outputs/Tables"
OUT_DIR = "Experiment2_Outputs/CombinedFigures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# -----------------------------------------------------------------------------
# Fixed model palette
# -----------------------------------------------------------------------------
MODELS = ["SARIMA", "Random Forest", "XGBoost",
          "LSTM", "TFT", "LSTM+TFT", "Ensemble"]

PALETTE = {
    "SARIMA":        "#8c564b",
    "Random Forest": "#1f77b4",
    "XGBoost":       "#ff7f0e",
    "LSTM":          "#2ca02c",
    "TFT":           "#d62728",
    "LSTM+TFT":      "#9467bd",
    "Ensemble":      "#111111",
}

METRICS = ["Sensitivity", "Specificity", "Precision", "F1",
           "AUC_ROC", "AUPRC",
           "FalseAlarmRate", "MissedAlertRate", "Brier",
           "LeadTime"]

METRIC_TITLES = {
    "Sensitivity":     "Sensitivity",
    "Specificity":     "Specificity",
    "Precision":       "Precision",
    "F1":              "F1-score",
    "AUC_ROC":         "AUC-ROC",
    "AUPRC":           "AUPRC",
    "FalseAlarmRate":  "False-Alarm Rate",
    "MissedAlertRate": "Missed-Alert Rate",
    "Brier":           "Brier score",
    "LeadTime":        "Lead Time (months)",
}


# -----------------------------------------------------------------------------
# Safe save — tries base name, falls back to timestamped name
# -----------------------------------------------------------------------------
def _try_save(fig, path):
    """Return True if saved, False if permission denied."""
    try:
        fig.savefig(path, dpi=300)
        return True
    except PermissionError:
        return False


def save_fig(fig, base_name):
    """
    Save as PNG + PDF. If a file is locked by another program,
    append a timestamped suffix instead.
    """
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{base_name}.{ext}")
        if _try_save(fig, path):
            print(f"  saved {base_name}.{ext}")
            continue
        # locked file → timestamped fallback
        ts = time.strftime("%Y%m%d_%H%M%S")
        alt_name = f"{base_name}_{ts}.{ext}"
        alt_path = os.path.join(OUT_DIR, alt_name)
        if _try_save(fig, alt_path):
            print(f"  ! {base_name}.{ext} locked — saved as {alt_name}")
        else:
            print(f"  ! could not save {alt_name}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Load baseline metrics
# -----------------------------------------------------------------------------
def load_horizons():
    dfs = {}
    for h in (1, 2, 3):
        f = os.path.join(IN_DIR, f"Baseline_Metrics_h{h}.csv")
        if not os.path.exists(f):
            print(f"  [skip] {f} not found")
            continue
        df = pd.read_csv(f, index_col=0)
        df = df.loc[[m for m in MODELS if m in df.index]]
        dfs[h] = df
    return dfs


# -----------------------------------------------------------------------------
# Master grid
# -----------------------------------------------------------------------------
def master_grid_with_legend(dfs):
    horizons = sorted(dfs.keys())
    nrow = len(horizons)
    ncol = len(METRICS)

    model_order = list(dfs[horizons[0]].index)
    colours = {m: PALETTE.get(m, "gray") for m in model_order}

    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=colours[m],
                      edgecolor="black",
                      linewidth=0.6)
        for m in model_order
    ]

    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(2.4 * ncol, 2.9 * nrow),
                             sharey=False)
    if nrow == 1:
        axes = axes.reshape(1, -1)

    for r, h in enumerate(horizons):
        df_h = dfs[h]
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            if metric not in df_h.columns:
                ax.axis("off")
                continue
            vals = df_h[metric].astype(float)
            bar_colors = [colours.get(name, "gray") for name in vals.index]
            ax.bar(range(len(vals)), vals.values,
                   color=bar_colors,
                   edgecolor="black",
                   linewidth=0.4)
            if r == 0:
                ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=10)
            if c == 0:
                ax.set_ylabel(f"h = {h}", fontsize=10)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.3)

    fig.legend(handles, model_order,
               loc="lower center",
               ncol=len(model_order),
               fontsize=10,
               frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Early-warning metrics — master grid "
                 "(rows = horizon, columns = metric)",
                 fontsize=14, y=1.005)

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    # NEW NAME — avoids the locked file
    save_fig(fig, "MasterGrid_AllMetrics_v2")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  MasterGrid_AllMetrics (v2) — with model legend")
    print("=" * 70)

    dfs = load_horizons()
    if not dfs:
        print("No Baseline_Metrics files found.")
        raise SystemExit(1)

    print(f"  Loaded horizons: {list(dfs.keys())}")
    for h, df in dfs.items():
        print(f"    h={h}: {df.shape[0]} models × {df.shape[1]} columns")

    master_grid_with_legend(dfs)
    print(f"\nSaved to: {os.path.abspath(OUT_DIR)}")