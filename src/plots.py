"""
================================================================================
 Experiment 2 — Early-Warning Baseline Comparison
 Produces bar charts (PNG @300 dpi + PDF) for:
   • Sensitivity / Recall
   • AUC-ROC
   • Specificity
   • Precision
   • F1-score
   • False-Alarm Rate
   • Missed-Alert Rate
   • Brier Score
   • Calibration (curve)
   • Operational Lead Time
================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, confusion_matrix, brier_score_loss,
                             roc_curve, precision_recall_curve, average_precision_score)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
IN_DIR  = "Experiment1_Outputs/Tables"
OUT_DIR = "Experiment2_Outputs"
FIG_DIR = os.path.join(OUT_DIR, "Figures")
TAB_DIR = os.path.join(OUT_DIR, "Tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 13,
                     "axes.labelsize": 12, "figure.dpi": 300, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})

MODELS = ["SARIMA", "Random Forest", "XGBoost", "LSTM", "TFT",
          "LSTM+TFT", "Ensemble"]
COLORS = {"SARIMA": "#8c564b", "Random Forest": "#1f77b4",
          "XGBoost": "#ff7f0e", "LSTM": "#2ca02c",
          "TFT": "#d62728", "LSTM+TFT": "#9467bd",
          "Ensemble": "#111111"}


# -----------------------------------------------------------------------------
# 1. Metrics
# -----------------------------------------------------------------------------
def compute_outbreak_threshold(y_true):
    """Outbreak defined as > 75th percentile of the district-level case distribution."""
    return float(np.nanpercentile(y_true, 75))


def predicted_probability(y_pred, threshold, scale):
    """
    Map a continuous prediction to a probability of exceeding the threshold
    via a monotone sigmoid around the threshold.
    """
    z = (y_pred - threshold) / max(scale, 1e-9)
    return 1.0 / (1.0 + np.exp(-z))


def safe_div(a, b):
    return a / b if b > 0 else np.nan


def early_warning_metrics(y_true, y_pred, threshold, scale,
                          y_true_bin=None, y_pred_bin=None):
    """
    Compute every metric required for the baseline comparison.
    y_true, y_pred : arrays of continuous malaria cases
    threshold      : outbreak threshold
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 5:
        return None

    yb_true = (y_true > threshold).astype(int)
    yb_pred = (y_pred > threshold).astype(int)
    prob    = predicted_probability(y_pred, threshold, scale)

    tn, fp, fn, tp = confusion_matrix(yb_true, yb_pred, labels=[0, 1]).ravel()

    sens     = safe_div(tp, tp + fn)                 # = recall
    spec     = safe_div(tn, tn + fp)
    prec     = safe_div(tp, tp + fp)
    f1       = safe_div(2 * prec * sens, prec + sens) if prec + sens > 0 else 0.0
    far      = safe_div(fp, fp + tn)                 # false-alarm rate
    mar      = safe_div(fn, fn + tp)                 # missed-alert rate (= 1 - sens)
    brier    = brier_score_loss(yb_true, prob)

    try:
        auc = roc_auc_score(yb_true, prob) if len(np.unique(yb_true)) > 1 else np.nan
    except ValueError:
        auc = np.nan
    try:
        auprc = average_precision_score(yb_true, prob) if len(np.unique(yb_true)) > 1 else np.nan
    except ValueError:
        auprc = np.nan

    return dict(Sensitivity=sens, Specificity=spec, Precision=prec,
                F1=f1, FalseAlarmRate=far, MissedAlertRate=mar,
                Brier=brier, AUC_ROC=auc, AUPRC=auprc,
                TP=tp, FP=fp, FN=fn, TN=tn,
                y_true=yb_true, y_pred_prob=prob)


# -----------------------------------------------------------------------------
# 2. Operational lead time
# -----------------------------------------------------------------------------
def operational_lead_time(y_true, y_pred, threshold, max_lead=3):
    """
    For every actual outbreak (actual > threshold), find the earliest month
    t - k (k ∈ 1..max_lead) in which the model's prediction also exceeded
    the threshold. Return the mean lead time and the detection rate.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    if n < max_lead + 2:
        return np.nan, np.nan

    events, leads = 0, []
    for t in range(max_lead, n):
        if y_true[t] > threshold:                    # actual outbreak
            events += 1
            found = np.nan
            for k in range(1, max_lead + 1):
                if y_pred[t - k] > threshold:
                    found = k
                    break
            leads.append(found)

    if events == 0:
        return np.nan, np.nan
    detected = sum(1 for l in leads if not np.isnan(l))
    mean_lead = np.nanmean([l for l in leads if not np.isnan(l)]) if detected else np.nan
    return mean_lead, detected / events


# -----------------------------------------------------------------------------
# 3. Load predictions and compute metrics per model/horizon
# -----------------------------------------------------------------------------
def compute_all_metrics(h):
    f = os.path.join(IN_DIR, f"Predictions_h{h}.csv")
    if not os.path.exists(f):
        print(f"  [skip] {f} not found.")
        return None, None
    df = pd.read_csv(f, parse_dates=["Date"])

    # Global threshold across all districts for horizon h
    threshold = compute_outbreak_threshold(df["Actual"].values)
    scale = float(np.nanstd(df["Actual"].values))
    print(f"  h={h}  threshold = {threshold:.1f}  (75th pct)  scale = {scale:.1f}")

    rows = []
    calib_data = {}
    roc_data   = {}
    pr_data    = {}
    for m in MODELS:
        if m not in df.columns:
            continue
        res = early_warning_metrics(df["Actual"].values, df[m].values,
                                    threshold, scale)
        if res is None:
            continue
        lead, det = operational_lead_time(df["Actual"].values, df[m].values,
                                          threshold, max_lead=3)
        res["LeadTime"]      = lead
        res["DetectionRate"] = det
        res["Model"]         = m
        rows.append(res)

        calib_data[m] = (res["y_true"], res["y_pred_prob"])
        try:
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_pred_prob"])
            roc_data[m] = (fpr, tpr)
        except Exception:
            pass
        try:
            p, r, _ = precision_recall_curve(res["y_true"], res["y_pred_prob"])
            pr_data[m] = (r, p)
        except Exception:
            pass

    summary = pd.DataFrame(rows).set_index("Model")
    keep = ["Sensitivity", "Specificity", "Precision", "F1",
            "FalseAlarmRate", "MissedAlertRate", "Brier",
            "AUC_ROC", "AUPRC", "LeadTime", "DetectionRate"]
    summary_out = summary[keep]
    return summary_out, dict(calib=calib_data, roc=roc_data, pr=pr_data,
                             threshold=threshold, scale=scale)


# -----------------------------------------------------------------------------
# 4. Bar-chart per metric
# -----------------------------------------------------------------------------
def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  saved {name}.png + .pdf")


def bar_chart(summary, metric, title, ylabel, h,
              ylim=None, higher_is_better=True, fmt="%.3f"):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    vals = summary[metric].astype(float)
    order = vals.sort_values(ascending=not higher_is_better).index.tolist()
    colors = [COLORS.get(m, "gray") for m in order]
    bars = ax.bar(order, vals[order], color=colors,
                  edgecolor="black", linewidth=0.7)
    for b, v in zip(bars, vals[order]):
        if np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2,
                v if not np.isnan(v) else 0,
                fmt % v, ha="center", va="bottom", fontsize=9)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_title(f"{title} — h = {h} month(s)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = metric.replace(" ", "")
    save_fig(fig, f"Baseline_{fname}_h{h}")


def all_bar_charts(summary, h):
    bar_chart(summary, "Sensitivity",      "Sensitivity / Recall",   "Recall",   h, ylim=(0, 1.05))
    bar_chart(summary, "AUC_ROC",          "AUC-ROC",                "AUC",      h, ylim=(0, 1.05))
    bar_chart(summary, "Specificity",      "Specificity",            "Specificity", h, ylim=(0, 1.05))
    bar_chart(summary, "Precision",        "Precision",              "Precision", h, ylim=(0, 1.05))
    bar_chart(summary, "F1",               "F1-score",               "F1",        h, ylim=(0, 1.05))
    bar_chart(summary, "FalseAlarmRate",   "False-Alarm Rate",       "FAR",       h,
              ylim=(0, 1.05), higher_is_better=False)
    bar_chart(summary, "MissedAlertRate",  "Missed-Alert Rate",      "MAR",       h,
              ylim=(0, 1.05), higher_is_better=False)
    bar_chart(summary, "Brier",            "Brier Score",            "Brier",     h,
              ylim=(0, 1.0), higher_is_better=False)
    if summary["LeadTime"].notna().any():
        bar_chart(summary, "LeadTime",     "Operational Lead Time",  "Months",     h,
                  ylim=(0, 3.5), higher_is_better=True, fmt="%.2f")


# -----------------------------------------------------------------------------
# 5. Calibration curves
# -----------------------------------------------------------------------------
def calibration_plot(calib_dict, h, n_bins=5):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for m, (y_true, y_prob) in calib_dict.items():
        try:
            bins = np.linspace(0, 1, n_bins + 1)
            idx  = np.digitize(y_prob, bins) - 1
            xs, ys = [], []
            for b in range(n_bins):
                msk = idx == b
                if msk.sum() < 3:
                    continue
                xs.append(y_prob[msk].mean())
                ys.append(y_true[msk].mean())
            if len(xs) >= 2:
                ax.plot(xs, ys, "o-", label=m, color=COLORS.get(m, None),
                        linewidth=1.6, markersize=6)
        except Exception:
            continue
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed outbreak frequency")
    ax.set_title(f"Calibration curves — h = {h}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    save_fig(fig, f"Baseline_Calibration_h{h}")


# -----------------------------------------------------------------------------
# 6. ROC and PR curves
# -----------------------------------------------------------------------------
def roc_plot(roc_dict, h):
    if not roc_dict:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    for m, (fpr, tpr) in roc_dict.items():
        ax.plot(fpr, tpr, linewidth=1.6, label=m, color=COLORS.get(m, None))
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"ROC curves — h = {h}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    save_fig(fig, f"Baseline_ROC_h{h}")


def pr_plot(pr_dict, h):
    if not pr_dict:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for m, (r, p) in pr_dict.items():
        ax.plot(r, p, linewidth=1.6, label=m, color=COLORS.get(m, None))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall curves — h = {h}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    save_fig(fig, f"Baseline_PrecisionRecall_h{h}")


# -----------------------------------------------------------------------------
# 7. Combined summary heatmap + summary table
# -----------------------------------------------------------------------------
def summary_heatmap(summary, h):
    heat = summary[["Sensitivity", "Specificity", "Precision", "F1",
                    "AUC_ROC", "FalseAlarmRate", "MissedAlertRate",
                    "Brier", "LeadTime"]].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu",
                cbar_kws={"label": "value"}, ax=ax,
                linewidths=0.4, linecolor="white")
    ax.set_title(f"Early-warning metrics — all models (h={h})")
    plt.tight_layout()
    save_fig(fig, f"Baseline_SummaryHeatmap_h{h}")


# -----------------------------------------------------------------------------
# 8. Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Experiment 2 — Baseline Comparison Bar Charts")
    print("=" * 70)

    all_summaries = {}
    for h in (1, 2, 3):
        print(f"\n--- Horizon h = {h} ---")
        summary, extra = compute_all_metrics(h)
        if summary is None:
            continue
        all_summaries[h] = summary
        summary.to_csv(os.path.join(TAB_DIR, f"Baseline_Metrics_h{h}.csv"))
        try:
            summary.to_latex(os.path.join(TAB_DIR, f"Baseline_Metrics_h{h}.tex"),
                             float_format="%.3f",
                             caption=f"Early-warning metrics at h={h}",
                             label=f"tab:ew_h{h}")
        except Exception:
            pass
        all_bar_charts(summary, h)
        calibration_plot(extra["calib"], h)
        roc_plot(extra["roc"], h)
        pr_plot(extra["pr"], h)
        summary_heatmap(summary, h)

    # Combined table
    if all_summaries:
        comb = []
        for h, s in all_summaries.items():
            s2 = s.copy()
            s2.insert(0, "Horizon", f"h={h}")
            comb.append(s2)
        combined = pd.concat(comb)
        combined.to_csv(os.path.join(TAB_DIR, "Baseline_Metrics_AllHorizons.csv"))
        try:
            combined.to_latex(os.path.join(TAB_DIR, "Baseline_Metrics_AllHorizons.tex"),
                              float_format="%.3f",
                              caption="Early-warning metrics across horizons.",
                              label="tab:ew_all")
        except Exception:
            pass
        print("\n" + "=" * 70 + "\n  SUMMARY\n" + "=" * 70)
        print(combined.round(3).to_string())

    print(f"\nAll figures and tables saved to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()