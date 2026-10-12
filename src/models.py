"""
================================================================================
 Experiment 2 — COMBINED / GROUPED BAR CHARTS
 Reads  : Experiment2_Outputs/Tables/Baseline_Metrics_h{1,2,3}.csv
 Writes : Combined & grouped bar charts (PNG @300 dpi + PDF)
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
FIG_DIR = os.path.join(OUT_DIR, "CombinedFigures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 13,
                     "axes.labelsize": 12, "figure.dpi": 300, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})

MODELS = ["SARIMA", "Random Forest", "XGBoost", "LSTM", "TFT",
          "LSTM+TFT", "Ensemble"]

# Grouped metric buckets — chosen by "what goes together in the same chart"
METRIC_GROUPS = {
    "Detection":   ["Sensitivity", "Specificity", "Precision", "F1"],
    "Discrimination": ["AUC_ROC", "AUPRC"],
    "Errors":      ["FalseAlarmRate", "MissedAlertRate", "Brier"],
    "Operational": ["LeadTime", "DetectionRate"],
}


# -----------------------------------------------------------------------------
# Load all horizons
# -----------------------------------------------------------------------------
def load_all():
    dfs = {}
    for h in (1, 2, 3):
        f = os.path.join(IN_DIR, f"Baseline_Metrics_h{h}.csv")
        if not os.path.exists(f):
            print(f"  [skip] {f} not found")
            continue
        df = pd.read_csv(f, index_col=0)
        df = df.loc[[m for m in MODELS if m in df.index]]
        df = df.drop(columns=[c for c in ("TP", "FP", "FN", "TN") if c in df.columns])
        dfs[h] = df
    return dfs


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  saved {name}.png + .pdf")


def _bar_labels(ax, fmt="%.2f", fontsize=8):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=fontsize, padding=2)


# -----------------------------------------------------------------------------
# 1. Grouped bar — one metric group per figure, all horizons
# -----------------------------------------------------------------------------
def grouped_bars_one_group(dfs, group_name, metrics):
    n_h = len(dfs)
    fig, axes = plt.subplots(1, n_h, figsize=(5.5 * n_h, 5), sharey=True)
    if n_h == 1: axes = [axes]

    for ax, (h, df) in zip(axes, dfs.items()):
        sub = df[metrics].copy()
        sub.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.6,
                 colormap="tab10", width=0.75)
        ax.set_title(f"h = {h} month(s)")
        ax.set_xlabel(""); ax.set_ylabel(", ".join(metrics))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, title=None)
        _bar_labels(ax, fmt="%.2f", fontsize=7)

    fig.suptitle(f"{group_name} metrics — model comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, f"Grouped_{group_name.replace(' ','')}")


# -----------------------------------------------------------------------------
# 2. Grouped bar — one metric, all horizons & all models
# -----------------------------------------------------------------------------
def grouped_bars_one_metric(dfs, metric, ylim=None):
    n_h = len(dfs)
    fig, axes = plt.subplots(1, n_h, figsize=(5.5 * n_h, 4.8), sharey=True)
    if n_h == 1: axes = [axes]

    for ax, (h, df) in zip(axes, dfs.items()):
        vals = df[metric].astype(float)
        bars = ax.bar(range(len(vals)), vals.values,
                      color=sns.color_palette("viridis", len(vals)),
                      edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=30, ha="right")
        ax.set_title(f"h = {h} month(s)")
        ax.set_ylabel(metric)
        if ylim is not None: ax.set_ylim(*ylim)
        for b, v in zip(bars, vals.values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{metric} — baseline comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, f"Grouped_{metric}_AllHorizons")


# -----------------------------------------------------------------------------
# 3. Multi-panel — every metric, one panel each, h=1 (thesis figure)
# -----------------------------------------------------------------------------
def multi_panel_all_metrics(dfs, h=1):
    if h not in dfs:  return
    df = dfs[h]
    metrics = ["Sensitivity", "Specificity", "Precision", "F1",
               "AUC_ROC", "AUPRC", "FalseAlarmRate", "MissedAlertRate",
               "Brier", "LeadTime", "DetectionRate"]
    metrics = [m for m in metrics if m in df.columns]

    n = len(metrics)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
    axes = axes.flatten()

    for i, m in enumerate(metrics):
        ax = axes[i]
        vals = df[m].astype(float).sort_values(ascending=False)
        colors = sns.color_palette("viridis", len(vals))
        bars = ax.bar(range(len(vals)), vals.values,
                      color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(m, fontsize=11)
        for b, v in zip(bars, vals.values):
            if not np.isnan(v):
                ax.text(b.get_x()+b.get_width()/2, v,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Early-warning metrics — all models (h = {h})",
                 fontsize=15, y=1.00)
    plt.tight_layout()
    save_fig(fig, f"MultiPanel_AllMetrics_h{h}")


# -----------------------------------------------------------------------------
# 4. Model-per-figure multi-panel (thesis figure, alternate layout)
# -----------------------------------------------------------------------------
def multi_panel_by_model(dfs, h=1):
    if h not in dfs:  return
    df = dfs[h]
    metrics = ["Sensitivity", "Specificity", "Precision", "F1",
               "AUC_ROC", "AUPRC", "FalseAlarmRate", "MissedAlertRate",
               "Brier", "LeadTime", "DetectionRate"]
    metrics = [m for m in metrics if m in df.columns]
    models = [m for m in df.index]

    n = len(models)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        vals = df.loc[model, metrics].astype(float)
        bars = ax.bar(range(len(vals)), vals.values,
                      color=sns.color_palette("magma", len(vals)),
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=7)
        ax.set_title(model, fontsize=11)
        ax.set_ylim(0, 1.05)
        for b, v in zip(bars, vals.values):
            if not np.isnan(v):
                ax.text(b.get_x()+b.get_width()/2, v,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Per-model metric profile (h = {h})", fontsize=15, y=1.00)
    plt.tight_layout()
    save_fig(fig, f"MultiPanel_ByModel_h{h}")


# -----------------------------------------------------------------------------
# 5. Grouped bars across horizons — one figure per metric
# -----------------------------------------------------------------------------
def horizons_grouped_bars(dfs, metric, ylim=(0, 1.05)):
    if not dfs:  return
    horizons = sorted(dfs.keys())
    models   = [m for m in MODELS if all(m in dfs[h].index for h in horizons)]

    data = []
    for h in horizons:
        for m in models:
            try:
                v = float(dfs[h].loc[m, metric])
            except Exception:
                v = np.nan
            data.append({"Horizon": f"h={h}", "Model": m, metric: v})
    d = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=d, x="Model", y=metric, hue="Horizon",
                palette="Set2", edgecolor="black", ax=ax)
    ax.set_title(f"{metric} by model & horizon")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)
    if metric not in ("LeadTime",):
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
    plt.tight_layout()
    save_fig(fig, f"AcrossHorizons_{metric}")


# -----------------------------------------------------------------------------
# 6. Radar chart — every model, all metrics normalised
# -----------------------------------------------------------------------------
def radar_chart(dfs, h=1):
    if h not in dfs:  return
    df = dfs[h].copy()
    metrics = ["Sensitivity", "Specificity", "Precision", "F1",
               "AUC_ROC", "AUPRC", "DetectionRate"]
    metrics = [m for m in metrics if m in df.columns]
    # Add inverse of error metrics so bigger is always better
    for inv, src in [("1-FAR", "FalseAlarmRate"),
                     ("1-MAR", "MissedAlertRate"),
                     ("1-Brier", "Brier")]:
        if src in df.columns:
            df[inv] = 1 - df[src].astype(float)
            metrics.append(inv)

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    for m in df.index:
        vals = df.loc[m, metrics].astype(float).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, label=m)
        ax.fill(angles, vals, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Model performance profile (h={h})", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
    plt.tight_layout()
    save_fig(fig, f"Radar_Profile_h{h}")


# -----------------------------------------------------------------------------
# 7. Master figure — all metrics in one grid × all horizons as rows
# -----------------------------------------------------------------------------
def master_grid(dfs):
    metrics = ["Sensitivity", "Specificity", "Precision", "F1",
               "AUC_ROC", "AUPRC", "FalseAlarmRate", "MissedAlertRate",
               "Brier", "LeadTime"]
    horizons = sorted(dfs.keys())
    ncol = len(metrics)
    nrow = len(horizons)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(2.2*ncol, 2.4*nrow),
                             sharey=False)
    if nrow == 1: axes = axes.reshape(1, -1)

    for r, h in enumerate(horizons):
        df = dfs[h]
        for c, m in enumerate(metrics):
            ax = axes[r, c]
            if m not in df.columns:
                ax.axis("off"); continue
            vals = df[m].astype(float)
            ax.bar(range(len(vals)), vals.values,
                   color=sns.color_palette("viridis", len(vals)),
                   edgecolor="black", linewidth=0.4)
            ax.set_xticks([])
            if r == 0:
                ax.set_title(m, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"h={h}", fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Early-warning metrics — master grid (rows = horizon, cols = metric)",
                 fontsize=14, y=1.005)
    plt.tight_layout()
    save_fig(fig, "MasterGrid_AllMetrics")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Combined / grouped bar charts — Experiment 2")
    print("=" * 70)

    dfs = load_all()
    if not dfs:
        print("No Baseline_Metrics files found. Run the baseline script first.")
        return
    print(f"  Loaded horizons: {list(dfs.keys())}")

    print("\n[1] Grouped metric-group bars ...")
    for gname, metrics in METRIC_GROUPS.items():
        metrics = [m for m in metrics if m in dfs[list(dfs.keys())[0]].columns]
        if metrics:
            grouped_bars_one_group(dfs, gname, metrics)

    print("\n[2] Grouped single-metric bars (all horizons) ...")
    for m in ["Sensitivity", "Specificity", "Precision", "F1", "AUC_ROC",
              "AUPRC", "FalseAlarmRate", "MissedAlertRate", "Brier",
              "LeadTime", "DetectionRate"]:
        grouped_bars_one_metric(dfs, m, ylim=(0, 1.05) if m != "LeadTime" else None)

    print("\n[3] Multi-panel per horizon ...")
    for h in dfs.keys():
        multi_panel_all_metrics(dfs, h=h)
        multi_panel_by_model(dfs, h=h)
        radar_chart(dfs, h=h)

    print("\n[4] Across-horizons grouped bars ...")
    for m in ["Sensitivity", "Specificity", "Precision", "F1",
              "AUC_ROC", "AUPRC", "FalseAlarmRate", "MissedAlertRate",
              "Brier", "LeadTime"]:
        horizons_grouped_bars(dfs, m)

    print("\n[5] Master grid ...")
    master_grid(dfs)

    print(f"\nAll combined figures saved to: {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()