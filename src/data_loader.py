"""
================================================================================
 CPST-MalariaNet — Experiment 1 : Malaria Prediction in Burundi
 Models   : SARIMA | RF | XGBoost | LSTM | TFT | LSTM+TFT | Full Ensemble
 Metrics  : MAE, MSE, RMSE, MAPE, R²
 Outputs  : PNG (300 dpi) + PDF figures + CSV/LaTeX tables
================================================================================
"""

import os, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                     MultiHeadAttention, Add, LayerNormalization,
                                     GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(42); tf.random.set_seed(42)

FILEPATH = "BdiDATASETS.xlsx"
OUT_DIR  = "Experiment1_Outputs"
FIG_DIR  = os.path.join(OUT_DIR, "Figures"); os.makedirs(FIG_DIR, exist_ok=True)
TAB_DIR  = os.path.join(OUT_DIR, "Tables");  os.makedirs(TAB_DIR, exist_ok=True)

plt.rcParams.update({"font.family":"serif","font.size":11,"axes.titlesize":13,
                     "axes.labelsize":12,"figure.dpi":300,"savefig.dpi":300,
                     "savefig.bbox":"tight"})


# =============================================================================
# 1. DATA LOADING (robust)
# =============================================================================
def _to_float(x):
    try:
        s = str(x).strip().replace(" ", "").replace("\u202f", "").replace(",", ".")
        return float(s)
    except Exception:
        return None


def extract_malaria_from_sheet(filepath, sheet_name, year):
    """
    Extract (District, Year, Month, Malaria_Cases) from a sheet.
    Strategy: locate every row whose first meaningful cell starts with 'DS ',
              then take the next 12 numeric values as months 1–12.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    records = []
    for i in range(len(df)):
        row = df.iloc[i].values
        district, dcol = None, None
        for j, v in enumerate(row):
            if pd.isna(v):      continue
            s = str(v).strip()
            if s.startswith("DS ") and 3 < len(s) < 80 and not s.lower().startswith("ds total"):
                district, dcol = s, j
                break
        if district is None:    continue
        vals = []
        for k in range(dcol + 1, len(row)):
            v = row[k]
            if pd.isna(v):      continue
            f = _to_float(v)
            if f is not None and 0 <= f <= 5_000_000:
                vals.append(f)
            if len(vals) >= 12: break
        if len(vals) < 12:      continue
        for m, val in enumerate(vals[:12], start=1):
            records.append({"District": district, "Year": year,
                            "Month": m, "Malaria_Cases": val})
    return pd.DataFrame(records)


def load_malaria_cases(filepath):
    parts = []
    for sheet, year in [("2022", 2022), ("2023", 2023), ("2024", 2024)]:
        try:
            p = extract_malaria_from_sheet(filepath, sheet, year)
            print(f"  Sheet {sheet}: {len(p)} records, "
                  f"{p['District'].nunique() if not p.empty else 0} districts")
            parts.append(p)
        except Exception as e:
            print(f"  Sheet {sheet}: FAILED -> {e}")
    if not parts:
        raise RuntimeError("No malaria data could be extracted.")
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["District", "Year", "Month"], keep="first")
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01")
    df = df.sort_values(["District", "Date"]).reset_index(drop=True)
    print(f"  → Total malaria: {len(df)} rows | "
          f"{df['District'].nunique()} districts | "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_national_env(filepath):
    """
    Load national-level mean environmental features from 'DataMonthltyYear'.
    Header is on Excel row 2 → pandas index 1.
    """
    df = pd.read_excel(filepath, sheet_name="DataMonthltyYear", header=1)
    df.columns = [str(c).strip() for c in df.columns]

    yyyymm_col = df.columns[0]
    df["_ym"] = df[yyyymm_col].astype(str).str.replace(r"\.0$", "", regex=True)
    df["Date"] = pd.to_datetime(df["_ym"], format="%Y%m", errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    rainfall_cols = ["Buja","Gitega","Cankuzo","Gisozi","Muyinga",
                     "Nyanza_lac","Kirundo","Kinyinya","Rwegura","Mparambo"]
    temp_cols     = ["Bja_Mean","Cank_Mean","Gisozi_Mean","Giteg_Mean","Karuzi_Mean",
                     "Kiny_Mean","Kirun_Mean","Mparamb_mean","Tora_Mean","Musasa_Mean",
                     "Muying_Mean","Nyanza_Mean","Rwegura_mean"]
    hum_cols      = ["Bujumbura (%)","Cankuzo %","Gitega %","Kirundo %",
                     "Gisozi%","Muyinga%","Kinyinya%","Nyanza%"]

    def _mean(cols):
        keep = [c for c in cols if c in df.columns]
        return df[keep].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True) \
               if keep else np.nan

    df["Rainfall_Mean"]  = _mean(rainfall_cols)
    df["Temp_Mean"]      = _mean(temp_cols)
    df["Humidity_Mean"]  = _mean(hum_cols)

    keep = ["Year","Month","Rainfall_Mean","Temp_Mean","Humidity_Mean"]
    env = df[keep].dropna(subset=["Year","Month"]).reset_index(drop=True)
    print(f"  → Env: {len(env)} rows | "
          f"{int(env['Year'].min())} → {int(env['Year'].max())}")
    return env


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
def build_features(malaria, env):
    df = malaria.merge(env, on=["Year", "Month"], how="left").copy()
    df = df.sort_values(["District", "Date"]).reset_index(drop=True)

    # Seasonal encoding
    df["Month_Sin"] = np.sin(2*np.pi*df["Month"]/12)
    df["Month_Cos"] = np.cos(2*np.pi*df["Month"]/12)

    # Lagged cases & rolling stats (past-only)
    for lag in (1, 2, 3, 6):
        df[f"Cases_Lag{lag}"] = df.groupby("District")["Malaria_Cases"].shift(lag)
    for w in (3, 6):
        df[f"Cases_Roll{w}"] = (df.groupby("District")["Malaria_Cases"]
                                .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1)))

    # Lagged env
    for v in ("Rainfall_Mean", "Temp_Mean", "Humidity_Mean"):
        if v in df.columns:
            for lag in (1, 2):
                df[f"{v}_Lag{lag}"] = df.groupby("District")[v].shift(lag)

    return df


def add_target(df, h):
    df = df.copy()
    df[f"Target_h{h}"] = df.groupby("District")["Malaria_Cases"].shift(-h)
    return df


# =============================================================================
# 3. MODELS
# =============================================================================
def run_sarima(train_df, test_df, h):
    preds = []
    for d, g in test_df.groupby("District"):
        hist = train_df[train_df["District"] == d].sort_values("Date")["Malaria_Cases"].values
        if len(hist) < 12:
            preds.extend([np.nan]*len(g)); continue
        try:
            fit = SARIMAX(hist, order=(1,1,1), seasonal_order=(1,1,1,12),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            preds.extend(np.asarray(fit.forecast(steps=len(g))))
        except Exception:
            preds.extend([float(np.nanmean(hist))]*len(g))
    return np.array(preds)


def run_rf(Xtr, ytr, Xte):
    m = RandomForestRegressor(n_estimators=300, max_depth=12,
                              min_samples_leaf=2, random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr); return m, m.predict(Xte)


def run_xgb(Xtr, ytr, Xva, yva, Xte):
    m = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                         subsample=0.85, colsample_bytree=0.85,
                         random_state=42, n_jobs=-1, verbosity=0)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=40, verbose=False)
    return m, m.predict(Xte)


def build_lstm(seq, n_feat, units=48, drop=0.3):
    inp = Input(shape=(seq, n_feat))
    x = LSTM(units, return_sequences=True)(inp); x = Dropout(drop)(x)
    x = LSTM(units//2)(x); x = Dropout(drop)(x)
    x = Dense(24, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out); m.compile(Adam(1e-3), loss="mse", metrics=["mae"]); return m


def build_tft(seq, n_feat, d=48, heads=4, drop=0.2):
    inp = Input(shape=(seq, n_feat))
    g = Dense(d, activation="elu")(inp); g = Dropout(drop)(g)
    a = MultiHeadAttention(num_heads=heads, key_dim=d//heads, dropout=drop)(g, g)
    a = LayerNormalization()(Add()([g, a]))
    f = Dense(d, activation="relu")(a); f = Dropout(drop)(f)
    f = LayerNormalization()(Add()([a, f]))
    x = GlobalAveragePooling1D()(f)
    x = Dense(24, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out); m.compile(Adam(1e-3), loss="mse", metrics=["mae"]); return m


def seq_data(X, y, seq):
    Xs, ys = [], []
    for i in range(len(X)-seq):
        Xs.append(X[i:i+seq]); ys.append(y[i+seq])
    return np.array(Xs), np.array(ys)


def run_dl(model, Xtr, ytr, Xva, yva, Xte, seq, sx, sy):
    Xtr_s = sx.transform(Xtr); Xva_s = sx.transform(Xva); Xte_s = sx.transform(Xte)
    ytr_s = sy.fit_transform(ytr.reshape(-1,1)).ravel()
    yva_s = sy.transform(yva.reshape(-1,1)).ravel()

    Xtr_q, ytr_q = seq_data(Xtr_s, ytr_s, seq)
    Xva_q, yva_q = seq_data(Xva_s, yva_s, seq)
    Xte_q, _     = seq_data(Xte_s, np.zeros(len(Xte_s)), seq)

    cb = [EarlyStopping(patience=20, restore_best_weights=True),
          ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)]
    model.fit(Xtr_q, ytr_q, validation_data=(Xva_q, yva_q),
              epochs=150, batch_size=16, callbacks=cb, verbose=0)
    pred_s = model.predict(Xte_q, verbose=0).ravel()
    return sy.inverse_transform(pred_s.reshape(-1,1)).ravel()


# =============================================================================
# 4. METRICS
# =============================================================================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) < 2:
        return dict(MAE=np.nan, MSE=np.nan, RMSE=np.nan, MAPE=np.nan, R2=np.nan)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    nz   = y_true != 0
    mape = np.mean(np.abs((y_true[nz]-y_pred[nz])/y_true[nz]))*100 if nz.any() else np.nan
    r2   = r2_score(y_true, y_pred)
    return dict(MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, R2=r2)


# =============================================================================
# 5. MAIN EXPERIMENT
# =============================================================================
def run_experiment_1(filepath, horizon=1, seq_len=3, test_months=4):
    print("\n" + "="*70)
    print(f"  EXPERIMENT 1 — HORIZON h = {horizon} month(s)")
    print("="*70)

    # --- Load ---
    print("[1/6] Loading malaria cases ...")
    malaria = load_malaria_cases(filepath)
    print("[2/6] Loading environmental features ...")
    env = load_national_env(filepath)

    # --- Features ---
    print("[3/6] Building features ...")
    df = build_features(malaria, env)
    df = add_target(df, horizon)
    df = df.dropna(subset=[f"Target_h{horizon}"]).reset_index(drop=True)
    print(f"      → after target drop: {df.shape}")

    # Features
    drop = {"District","Date","Malaria_Cases","Target_h1","Target_h2","Target_h3"}
    feat_cols = [c for c in df.columns if c not in drop and df[c].dtype != "O"]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

    # --- Split per district ---
    tr_parts, te_parts, va_parts, tr2_parts = [], [], [], []
    for d, g in df.groupby("District"):
        g = g.sort_values("Date")
        if len(g) < seq_len + test_months + 3:
            continue
        tr_parts.append(g.iloc[:-test_months])
        te_parts.append(g.iloc[-test_months:])
    if not tr_parts:
        raise RuntimeError("Not enough data per district. "
                           "Reduce seq_len / test_months.")
    train = pd.concat(tr_parts, ignore_index=True)
    test  = pd.concat(te_parts, ignore_index=True)
    print(f"      → train {train.shape} | test {test.shape}")

    # Inner val (last 2 months of train)
    for d, g in train.groupby("District"):
        g = g.sort_values("Date")
        if len(g) < seq_len + 3:
            continue
        va_parts.append(g.iloc[-2:])
        tr2_parts.append(g.iloc[:-2])
    tr2 = pd.concat(tr2_parts, ignore_index=True)
    val = pd.concat(va_parts, ignore_index=True)

    Xtr, ytr = tr2[feat_cols].values, tr2[f"Target_h{horizon}"].values
    Xva, yva = val[feat_cols].values, val[f"Target_h{horizon}"].values
    Xte, yte = test[feat_cols].values, test[f"Target_h{horizon}"].values

    sx = StandardScaler().fit(Xtr)
    sy = StandardScaler().fit(ytr.reshape(-1,1))

    results, preds = {}, {}

    # ---- SARIMA ----
    print("[4/6] Fitting models ...")
    print("      → SARIMA")
    try:
        p = run_sarima(train, test, horizon)
        preds["SARIMA"] = p
        results["SARIMA"] = compute_metrics(yte, p)
    except Exception as e:
        print(f"        SARIMA failed: {e}"); results["SARIMA"] = {}

    # ---- RF ----
    print("      → Random Forest")
    _, p = run_rf(Xtr, ytr, Xte)
    preds["Random Forest"] = p; results["Random Forest"] = compute_metrics(yte, p)

    # ---- XGBoost ----
    print("      → XGBoost")
    _, p = run_xgb(Xtr, ytr, Xva, yva, Xte)
    preds["XGBoost"] = p; results["XGBoost"] = compute_metrics(yte, p)

    # ---- LSTM ----
    print("      → LSTM")
    try:
        m_lstm = build_lstm(seq_len, len(feat_cols))
        p_lstm = run_dl(m_lstm, Xtr, ytr, Xva, yva, Xte, seq_len, sx, sy)
        preds["LSTM"] = p_lstm
        results["LSTM"] = compute_metrics(yte[-len(p_lstm):], p_lstm)
    except Exception as e:
        print(f"        LSTM failed: {e}"); p_lstm = np.full(len(yte), np.nan)

    # ---- TFT ----
    print("      → TFT (simplified attention)")
    try:
        m_tft = build_tft(seq_len, len(feat_cols))
        p_tft = run_dl(m_tft, Xtr, ytr, Xva, yva, Xte, seq_len, sx, sy)
        preds["TFT"] = p_tft
        results["TFT"] = compute_metrics(yte[-len(p_tft):], p_tft)
    except Exception as e:
        print(f"        TFT failed: {e}"); p_tft = np.full(len(yte), np.nan)

    # ---- LSTM+TFT ----
    print("      → LSTM + TFT hybrid")
    if not np.isnan(p_lstm).all() and not np.isnan(p_tft).all():
        L = min(len(p_lstm), len(p_tft))
        p_hy = 0.5*p_lstm[-L:] + 0.5*p_tft[-L:]
        preds["LSTM+TFT"] = p_hy
        results["LSTM+TFT"] = compute_metrics(yte[-L:], p_hy)
    else:
        L = 0; p_hy = np.array([])

    # ---- Full ensemble ----
    print("      → LSTM + TFT + XGBoost + RF ensemble")
    rf_pred = preds["Random Forest"]; xgb_pred = preds["XGBoost"]
    if L > 0:
        # Validation errors for adaptive weights
        try:
            val_lstm = m_lstm.predict(seq_data(sx.transform(Xva),
                                     np.zeros(len(Xva)), seq_len)[0], verbose=0).ravel()
            val_lstm = sy.inverse_transform(val_lstm.reshape(-1,1)).ravel()
            val_tft  = m_tft.predict(seq_data(sx.transform(Xva),
                                     np.zeros(len(Xva)), seq_len)[0], verbose=0).ravel()
            val_tft  = sy.inverse_transform(val_tft.reshape(-1,1)).ravel()
            yv = yva[-min(len(val_lstm), len(val_tft)):]
            errs = {"lstm": mean_absolute_error(yv, val_lstm[-len(yv):]),
                    "tft":  mean_absolute_error(yv, val_tft[-len(yv):]),
                    "xgb":  mean_absolute_error(yva, xgb.predict(Xva)),
                    "rf":   mean_absolute_error(yva, RandomForestRegressor(
                             n_estimators=200, max_depth=12, random_state=42)
                             .fit(Xtr, ytr).predict(Xva))}
            inv = {k: 1/(v+1e-6) for k, v in errs.items()}
            tot = sum(inv.values()); w = {k: v/tot for k, v in inv.items()}
        except Exception:
            w = {"lstm":0.25,"tft":0.25,"xgb":0.25,"rf":0.25}

        y_al = yte[-L:]
        ens = (w["lstm"]*p_lstm[-L:] + w["tft"]*p_tft[-L:] +
               w["xgb"] *xgb_pred[-L:] + w["rf"] *rf_pred[-L:])
        preds["Ensemble"] = ens
        results["Ensemble"] = compute_metrics(y_al, ens)
        results["Ensemble"]["weights"] = w

    # ---- Assemble preds dataframe ----
    meta_cols = ["District", "Date"]
    if L > 0:
        meta = test.iloc[-L:][meta_cols].reset_index(drop=True)
        y_al = yte[-L:]
    else:
        meta = test[meta_cols].reset_index(drop=True); y_al = yte
    pdf = meta.copy(); pdf["Actual"] = y_al
    for k, v in preds.items():
        if len(v) >= len(pdf): pdf[k] = v[-len(pdf):]
        else: pdf[k] = np.nan

    print("[5/6] Results:")
    print(pd.DataFrame(results).T.round(4).to_string())

    print("[6/6] Saving tables and figures ...")
    save_outputs(results, pdf, horizon)
    return results, pdf


# =============================================================================
# 6. OUTPUTS
# =============================================================================
def save_outputs(results, pdf, h):
    # ---- Table ----
    rows = [{"Model": m, "MAE": r.get("MAE", np.nan),
             "MSE": r.get("MSE", np.nan), "RMSE": r.get("RMSE", np.nan),
             "MAPE (%)": r.get("MAPE", np.nan), "R²": r.get("R2", np.nan)}
            for m, r in results.items() if r]
    t = pd.DataFrame(rows).sort_values("RMSE")
    t.to_csv(os.path.join(TAB_DIR, f"Table_Metrics_h{h}.csv"), index=False)
    try:
        t.to_latex(os.path.join(TAB_DIR, f"Table_Metrics_h{h}.tex"),
                   index=False, float_format="%.3f",
                   caption=f"Experiment 1 — Metrics at h={h}",
                   label=f"tab:metrics_h{h}")
    except Exception: pass

    # ---- Fig 1 : metric bars ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=t, x="Model", y="RMSE", ax=ax[0], palette="viridis")
    ax[0].set_title(f"RMSE by model (h={h})"); ax[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=t, x="Model", y="MAE",  ax=ax[1], palette="magma")
    ax[1].set_title(f"MAE by model (h={h})");  ax[1].tick_params(axis="x", rotation=30)
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"Fig1_MetricBars_h{h}.{ext}"), dpi=300)
    plt.close()

    # ---- Fig 2 : time-series overlay ----
    dists = pdf["District"].unique()[:4]
    fig, axes = plt.subplots(len(dists), 1, figsize=(11, 2.4*len(dists)), sharex=True)
    if len(dists) == 1: axes = [axes]
    for ax, d in zip(axes, dists):
        s = pdf[pdf["District"] == d].sort_values("Date")
        ax.plot(s["Date"], s["Actual"], "k-o", label="Actual", markersize=4)
        for col, c in zip(["Random Forest","XGBoost","LSTM","TFT","Ensemble"],
                          ["tab:blue","tab:orange","tab:green","tab:red","tab:brown"]):
            if col in s.columns:
                ax.plot(s["Date"], s[col], "--", label=col, linewidth=1.2)
        ax.set_title(d, fontsize=10); ax.set_ylabel("Cases"); ax.grid(alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"Fig2_TimeSeries_h{h}.{ext}"), dpi=300)
    plt.close()

    # ---- Fig 3 : scatter ensemble ----
    if "Ensemble" in pdf.columns and pdf["Ensemble"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.scatter(pdf["Actual"], pdf["Ensemble"], alpha=0.55,
                   s=28, edgecolor="k", linewidth=0.3)
        lim = [0, max(pdf["Actual"].max(), pdf["Ensemble"].max())*1.05]
        ax.plot(lim, lim, "r--", label="Perfect fit")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted (Ensemble)")
        ax.set_title(f"Actual vs Predicted — Ensemble (h={h})")
        ax.legend(); ax.grid(alpha=0.3)
        for ext in ("png","pdf"):
            plt.savefig(os.path.join(FIG_DIR, f"Fig3_Scatter_h{h}.{ext}"), dpi=300)
        plt.close()

    # ---- Fig 4 : residuals ----
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for col in ["Random Forest","XGBoost","LSTM","TFT","Ensemble"]:
        if col in pdf.columns:
            sns.kdeplot(pdf["Actual"] - pdf[col], ax=ax, label=col, linewidth=1.4)
    ax.axvline(0, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Residual (Actual − Predicted)"); ax.set_title(f"Residuals (h={h})")
    ax.legend(); ax.grid(alpha=0.3)
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"Fig4_Residuals_h{h}.{ext}"), dpi=300)
    plt.close()

    # ---- Fig 5 : heatmap ----
    heat = t.set_index("Model")[["MAE","RMSE","MAPE (%)","R²"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"Performance heatmap (h={h})")
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"Fig5_Heatmap_h{h}.{ext}"), dpi=300)
    plt.close()

    pdf.to_csv(os.path.join(TAB_DIR, f"Predictions_h{h}.csv"), index=False)
    print(f"      → saved to {TAB_DIR} and {FIG_DIR}")


# =============================================================================
# 7. RUN
# =============================================================================
if __name__ == "__main__":
    all_results, all_preds = {}, {}
    for h in (1, 2, 3):
        try:
            res, preds = run_experiment_1(FILEPATH, horizon=h,
                                          seq_len=3, test_months=4)
            all_results[h] = res; all_preds[h] = preds
        except Exception as e:
            print(f"Horizon h={h} failed: {e}")

    # Combined table
    if all_results:
        combined = []
        for h, res in all_results.items():
            for m, r in res.items():
                if r:
                    combined.append({"Horizon": f"h={h}", "Model": m,
                                     "MAE": r.get("MAE"), "MSE": r.get("MSE"),
                                     "RMSE": r.get("RMSE"),
                                     "MAPE (%)": r.get("MAPE"),
                                     "R²": r.get("R2")})
        c = pd.DataFrame(combined)
        c.to_csv(os.path.join(TAB_DIR, "Table_AllHorizons.csv"), index=False)
        try:
            c.to_latex(os.path.join(TAB_DIR, "Table_AllHorizons.tex"),
                       index=False, float_format="%.3f",
                       caption="Experiment 1 — Metrics across horizons.",
                       label="tab:allhorizons")
        except Exception: pass
        print("\n" + "="*70)
        print("  FINAL SUMMARY")
        print("="*70)
        print(c.round(4).to_string(index=False))
        print(f"\nAll outputs in: {OUT_DIR}")