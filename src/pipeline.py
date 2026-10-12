"""
================================================================================
 CPST-MalariaNet — Experiment 1 (FINAL, works with XGBoost 1.x and 2.x)
================================================================================
"""
# ---------- Headless matplotlib — must come BEFORE pyplot ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, math, warnings, traceback, re
import numpy as np
import pandas as pd
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

# ============================================================================
# PATHS
# ============================================================================
FILEPATH = "BdiDATASETS.xlsx"
OUT_DIR  = "Experiment1_Outputs"
FIG_DIR  = os.path.join(OUT_DIR, "Figures")
TAB_DIR  = os.path.join(OUT_DIR, "Tables")
DATA_DIR = os.path.join(OUT_DIR, "ExtractedData")
DIAG_DIR = os.path.join(OUT_DIR, "Diagnostics")
for d in (OUT_DIR, FIG_DIR, TAB_DIR, DATA_DIR, DIAG_DIR):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({"font.family":"serif","font.size":11,"axes.titlesize":13,
                     "axes.labelsize":12,"figure.dpi":300,"savefig.dpi":300,
                     "savefig.bbox":"tight"})

def L(msg):  print(msg, flush=True)


# ============================================================================
# HELPERS
# ============================================================================
def clean_ym(df):
    if df is None or df.empty:  return df
    df = df.copy()
    df["Year"]  = pd.to_numeric(df["Year"],  errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df = df.dropna(subset=["Year","Month"])
    df["Year"]  = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df = df[(df["Month"]>=1) & (df["Month"]<=12)]
    df = df[(df["Year"]>=1980) & (df["Year"]<=2030)]
    return df.reset_index(drop=True)


def make_date(df):
    return pd.to_datetime(dict(year=df["Year"].astype(int),
                               month=df["Month"].astype(int), day=1))


_NUM_CLEAN = re.compile(r"[\s\u00a0\u202f]+")
def to_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):  return None
    if isinstance(v, (int, float)):  return float(v)
    s = str(v).strip()
    if not s or s.lower() in ("nan","nd","none"):  return None
    s = _NUM_CLEAN.sub("", s).replace(",", ".")
    try:    return float(s)
    except (ValueError, TypeError):  return None


_DS_RE = re.compile(r"^\s*DS[\s\u00a0\u202f]+(.+)$", re.IGNORECASE)
def is_ds(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):  return None
    s = str(v).strip()
    m = _DS_RE.match(s)
    if not m:  return None
    name = m.group(1).strip()
    if not name or len(name) > 80:  return None
    if name.lower() in ("total","totaux"):  return None
    return f"DS {name}"


# ============================================================================
# DIAGNOSTIC DUMP
# ============================================================================
def dump_sheet(filepath, sheet):
    try:
        df = pd.read_excel(filepath, sheet_name=sheet, header=None, nrows=20)
    except Exception as e:
        L(f"  {sheet}: FAILED read -> {e}"); return None
    fn = os.path.join(DIAG_DIR, f"diag_{re.sub(r'[^A-Za-z0-9]+','_',sheet)}.csv")
    df.to_csv(fn, index=False)
    L(f"  {sheet}: shape={df.shape} → {os.path.basename(fn)}")
    return df


def dump_all(filepath):
    L("\n" + "="*70 + "\n  STEP 0 — DIAGNOSTIC\n" + "="*70)
    xl = pd.ExcelFile(filepath)
    L(f"  Sheets: {xl.sheet_names}")
    for s in xl.sheet_names:
        dump_sheet(filepath, s)


# ============================================================================
# MALARIA EXTRACTION
# ============================================================================
def extract_malaria_sheet(filepath, sheet, year):
    df = pd.read_excel(filepath, sheet_name=sheet, header=None)
    L(f"  [{sheet}] shape={df.shape}")
    hits = []
    for i in range(len(df)):
        for c in range(df.shape[1]):
            d = is_ds(df.iloc[i, c])
            if d is not None:
                hits.append((i, c, d))
    L(f"  [{sheet}] DS cells: {len(hits)}")

    if not hits:
        L(f"  [{sheet}] fallback: keyword scan ...")
        keywords = ["Bubanza","Mpanda","Isare","Kabezi","Rwibaga","Bujumbura",
                    "Bururi","Matana","Rutovu","Cankuzo","Murore","Cibitoke",
                    "Mabayi","Gitega","Kibuye","Mutaho","Ryansoro","Buhiga",
                    "Nyabikere","Gahombo","Kayanza","Musema","Busoni","Kirundo",
                    "Mukenke","Vumbi","Makamba","Nyanza","Kiganda","Muramvya",
                    "Gashoho","Giteranyi","Muyinga","Fota","Kibumbu","Buye",
                    "Kiremba","Ngozi","Bugarama","Rumonge","Gihofi","Rutana",
                    "Butezi","Gisuru","Kinyinya","Ruyigi"]
        for i in range(len(df)):
            for c in range(df.shape[1]):
                v = df.iloc[i, c]
                if pd.isna(v):  continue
                s = str(v).strip()
                for kw in keywords:
                    if kw.lower() in s.lower() and 3 < len(s) < 60:
                        hits.append((i, c, s)); break
        L(f"  [{sheet}] fallback hits: {len(hits)}")

    records = []
    used_rows = set()
    for i, c, d in hits:
        if i in used_rows:  continue
        nums = []
        for k in range(c + 1, df.shape[1]):
            v = to_num(df.iloc[i, k])
            if v is None:  continue
            if 0 <= v <= 5_000_000:  nums.append(v)
            if len(nums) >= 12:  break
        if len(nums) < 12:  continue
        used_rows.add(i)
        for m, x in enumerate(nums[:12], 1):
            records.append({"District": d, "Year": year,
                            "Month": m, "Malaria_Cases": x})
    out = pd.DataFrame(records)
    L(f"  [{sheet}] extracted {len(out)} rows | "
      f"{out['District'].nunique() if not out.empty else 0} districts")
    return out


def load_malaria(filepath):
    L("\n[STEP 1] Malaria")
    parts = []
    for sheet, year in [("2022", 2022), ("2023", 2023), ("2024", 2024)]:
        try:
            p = extract_malaria_sheet(filepath, sheet, year)
            if not p.empty:  parts.append(p)
        except Exception as e:
            L(f"  {sheet} FAILED: {e}"); traceback.print_exc()

    if not parts:
        L("!!! NO malaria rows extracted.");  return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["District","Year","Month"], keep="first")
    df = clean_ym(df)
    if df.empty:
        L("!!! Malaria df empty after cleaning.");  return pd.DataFrame()
    df["Date"] = make_date(df)
    df = df.sort_values(["District","Date"]).reset_index(drop=True)
    L(f"  → TOTAL malaria: {len(df)} rows | "
      f"{df['District'].nunique()} districts | "
      f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ============================================================================
# ENVIRONMENT EXTRACTION
# ============================================================================
def extract_precipitation(fp):
    df = pd.read_excel(fp, sheet_name="precipitationmensuel30years", header=0)
    df.columns = [str(c).strip() for c in df.columns]
    months = ['Janv','Fev','Mars','Avril','Mai','Jn','Jt','At','Sept','Oct','Nov','Dec']
    months = [c for c in months if c in df.columns]
    long = df.melt(id_vars=[c for c in ["Station_Name","Lat","Lon","Elev","YY"] if c in df.columns],
                   value_vars=months, var_name="Month_Name", value_name="Rainfall")
    mmap = {'Janv':1,'Fev':2,'Mars':3,'Avril':4,'Mai':5,'Jn':6,
            'Jt':7,'At':8,'Sept':9,'Oct':10,'Nov':11,'Dec':12}
    long["Month"] = long["Month_Name"].map(mmap)
    long = long.rename(columns={"YY":"Year","Station_Name":"Station"})
    long = long[["Station","Year","Month","Rainfall"]].dropna(subset=["Year","Month"])
    long["Rainfall"] = long["Rainfall"].apply(to_num)
    long = long.dropna(subset=["Rainfall"])
    long = clean_ym(long)
    L(f"  precipitation: {len(long)} rows | {long['Station'].nunique() if not long.empty else 0} stations")
    return long


def extract_temperature(fp):
    df = pd.read_excel(fp, sheet_name="Temperaturemoyen", header=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"station_Name":"Station","YY":"Year","MM":"Month"})
    df["TMPMAX"] = df["TMPMAX"].apply(to_num)
    df["TMPMIN"] = df["TMPMIN"].apply(to_num)
    df["Temp_Mean"] = (df["TMPMAX"] + df["TMPMIN"]) / 2
    df = df[["Station","Year","Month","TMPMAX","TMPMIN","Temp_Mean"]].dropna(subset=["Temp_Mean"])
    df = clean_ym(df)
    L(f"  temperature: {len(df)} rows | {df['Station'].nunique() if not df.empty else 0} stations")
    return df


def extract_humidity(fp):
    df = pd.read_excel(fp, sheet_name="Humiditerelatives  stations", header=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"station_Name":"Station","YY":"Year","MM":"Month"})
    df["RHMAX"] = df["RHMAX"].apply(to_num)
    df["RHMIN"] = df["RHMIN"].apply(to_num)
    df["Humidity_Mean"] = (df["RHMAX"] + df["RHMIN"]) / 2
    df = df[["Station","Year","Month","RHMAX","RHMIN","Humidity_Mean"]].dropna(subset=["Humidity_Mean"])
    df = clean_ym(df)
    L(f"  humidity: {len(df)} rows | {df['Station'].nunique() if not df.empty else 0} stations")
    return df


def extract_lake(fp):
    df = pd.read_excel(fp, sheet_name="Niveau du Lac Tanganyika", header=0)
    df.columns = [str(c).strip() for c in df.columns]
    lake_cols = [c for c in df.columns if "moyenne" in c.lower()]
    lake_col = lake_cols[0] if lake_cols else df.columns[-1]
    df = df.rename(columns={lake_col:"Lake_Level"})
    df = df[["Year","Month","Lake_Level"]].dropna(subset=["Year","Month"])
    df["Lake_Level"] = df["Lake_Level"].apply(to_num)
    df = df.dropna(subset=["Lake_Level"])
    df = clean_ym(df)
    L(f"  lake level: {len(df)} rows")
    return df


def extract_river(fp):
    df = pd.read_excel(fp, sheet_name="Niveau d'eau de la RiviereRusiz", header=0)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c.lower().startswith("name"):                     rename[c] = "Station"
        elif c.upper() == "YY":                              rename[c] = "Year"
        elif c.upper() == "MM":                              rename[c] = "Month"
        elif "haut" in c.lower():                            rename[c] = "River_Level"
        elif "debit" in c.lower() or "débit" in c.lower():   rename[c] = "River_Discharge"
    df = df.rename(columns=rename)
    keep = [c for c in ["Year","Month","River_Level","River_Discharge"] if c in df.columns]
    df = df[keep].dropna(subset=["Year","Month"])
    df["River_Level"] = df["River_Level"].apply(to_num)
    if "River_Discharge" in df.columns:
        df["River_Discharge"] = df["River_Discharge"].apply(to_num)
    df = df.dropna(subset=["River_Level"])
    df = clean_ym(df)
    L(f"  river level: {len(df)} rows")
    return df


def build_env(fp):
    L("\n[STEP 2] Environment")
    try: precip = extract_precipitation(fp)
    except Exception as e: L(f"  precip failed: {e}");  precip = pd.DataFrame()
    try: temp   = extract_temperature(fp)
    except Exception as e: L(f"  temp failed: {e}");    temp = pd.DataFrame()
    try: hum    = extract_humidity(fp)
    except Exception as e: L(f"  hum failed: {e}");     hum = pd.DataFrame()
    try: lake   = extract_lake(fp)
    except Exception as e: L(f"  lake failed: {e}");    lake = pd.DataFrame()
    try: river  = extract_river(fp)
    except Exception as e: L(f"  river failed: {e}");   river = pd.DataFrame()

    def _mean(df, col, name):
        if df.empty or col not in df.columns:
            return pd.DataFrame(columns=["Year","Month",name])
        return df.groupby(["Year","Month"])[col].mean().reset_index().rename(columns={col:name})

    env = _mean(precip, "Rainfall",      "Rainfall_Mean")
    for other in (_mean(temp, "Temp_Mean", "Temp_Mean"),
                  _mean(hum,  "Humidity_Mean","Humidity_Mean"),
                  _mean(lake, "Lake_Level",  "Lake_Level"),
                  river.groupby(["Year","Month"])[["River_Level","River_Discharge"]].mean().reset_index()
                  if not river.empty else pd.DataFrame()):
        if not other.empty:
            env = env.merge(other, on=["Year","Month"], how="outer")
    env = env.sort_values(["Year","Month"]).reset_index(drop=True)
    env = clean_ym(env)
    L(f"  → env merged: {env.shape}")
    return env, precip, temp, hum, lake, river


# ============================================================================
# RAW PLOTS
# ============================================================================
def plot_raw(precip, temp, hum, lake, river, malaria):
    L("\n[STEP 3] Raw plots")

    def _save(fig, name):
        for ext in ("png","pdf"):
            fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300)
        plt.close(fig)
        L(f"  saved {name}.png + .pdf")

    if not precip.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        for st, g in precip.groupby("Station"):
            g = g.sort_values(["Year","Month"]).reset_index(drop=True)
            ax.plot(make_date(g), g["Rainfall"], linewidth=1, alpha=0.7, label=st)
        ax.set_title("Rainfall by station"); ax.set_ylabel("Rainfall (mm)")
        if precip["Station"].nunique() > 1: ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3); _save(fig, "Sheet_Rainfall")

    if not temp.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        for st, g in temp.groupby("Station"):
            g = g.sort_values(["Year","Month"]).reset_index(drop=True)
            ax.plot(make_date(g), g["Temp_Mean"], linewidth=1, alpha=0.7, label=st)
        ax.set_title("Mean temperature by station"); ax.set_ylabel("°C")
        if temp["Station"].nunique() > 1: ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3); _save(fig, "Sheet_Temperature")

    if not hum.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        for st, g in hum.groupby("Station"):
            g = g.sort_values(["Year","Month"]).reset_index(drop=True)
            ax.plot(make_date(g), g["Humidity_Mean"], linewidth=1, alpha=0.7, label=st)
        ax.set_title("Mean relative humidity by station"); ax.set_ylabel("%")
        if hum["Station"].nunique() > 1: ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3); _save(fig, "Sheet_Humidity")

    if not lake.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        lake = lake.sort_values(["Year","Month"]).reset_index(drop=True)
        ax.plot(make_date(lake), lake["Lake_Level"], color="tab:blue", linewidth=1.2)
        ax.set_title("Lake Tanganyika — monthly mean level")
        ax.set_ylabel("Level (m)"); ax.grid(alpha=0.3)
        _save(fig, "Sheet_LakeLevel")

    if not river.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        river = river.sort_values(["Year","Month"]).reset_index(drop=True)
        date = make_date(river)
        ax.plot(date, river["River_Level"], color="tab:green", linewidth=1.2, label="Level")
        if "River_Discharge" in river.columns:
            ax2 = ax.twinx()
            ax2.plot(date, river["River_Discharge"], color="tab:red",
                     linewidth=1, alpha=0.6, label="Discharge")
            ax2.set_ylabel("Discharge (m³/s)", color="tab:red")
        ax.set_title("Rusizi River — level & discharge")
        ax.set_ylabel("Level (m)"); ax.grid(alpha=0.3)
        _save(fig, "Sheet_RiverLevel")

    if not malaria.empty:
        fig, ax = plt.subplots(figsize=(13, 6))
        for d, g in malaria.groupby("District"):
            g = g.sort_values("Date")
            ax.plot(g["Date"], g["Malaria_Cases"], linewidth=1, alpha=0.7,
                    label=d.replace("DS ",""))
        ax.set_title("Malaria cases by district")
        ax.set_ylabel("Cases"); ax.legend(fontsize=6, ncol=3, loc="upper right")
        ax.grid(alpha=0.3); _save(fig, "Sheet_MalariaByDistrict")


# ============================================================================
# FEATURES
# ============================================================================
def build_features(malaria, env):
    df = malaria.merge(env, on=["Year","Month"], how="left")
    df = df.sort_values(["District","Date"]).reset_index(drop=True)
    df["Month_Sin"] = np.sin(2*np.pi*df["Month"]/12)
    df["Month_Cos"] = np.cos(2*np.pi*df["Month"]/12)
    for lag in (1, 2, 3):
        df[f"Cases_Lag{lag}"] = df.groupby("District")["Malaria_Cases"].shift(lag)
    for w in (3, 6):
        df[f"Cases_Roll{w}"] = (df.groupby("District")["Malaria_Cases"]
                                .transform(lambda x: x.rolling(w, min_periods=1)
                                           .mean().shift(1)))
    for var in ["Rainfall_Mean","Temp_Mean","Humidity_Mean",
                "Lake_Level","River_Level","River_Discharge"]:
        if var in df.columns:
            for lag in (1, 2):
                df[f"{var}_Lag{lag}"] = df.groupby("District")[var].shift(lag)
            df[var] = df.groupby("District")[var].transform(lambda x: x.ffill().bfill())
    return df


def add_target(df, h):
    df = df.copy()
    df[f"Target_h{h}"] = df.groupby("District")["Malaria_Cases"].shift(-h)
    return df


# ============================================================================
# MODELS
# ============================================================================
def run_sarima(train_df, test_df):
    preds = []
    for d, g in test_df.groupby("District"):
        hist = train_df[train_df["District"]==d].sort_values("Date")["Malaria_Cases"].values
        if len(hist) < 12:
            preds.extend([np.nan]*len(g));  continue
        try:
            fit = SARIMAX(hist, order=(1,1,1), seasonal_order=(1,1,1,12),
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            preds.extend(np.asarray(fit.forecast(steps=len(g))))
        except Exception:
            preds.extend([float(np.nanmean(hist))]*len(g))
    return np.array(preds)


def run_rf(Xtr, ytr, Xte):
    m = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr);  return m, m.predict(Xte)


# ---------- XGBoost — version-agnostic ----------
def run_xgb(Xtr, ytr, Xva, yva, Xte):
    base = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                random_state=42, n_jobs=-1, verbosity=0)
    try:
        # XGBoost >= 2.0 — early_stopping_rounds is a constructor arg
        m = xgb.XGBRegressor(**base, early_stopping_rounds=30)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    except TypeError:
        # XGBoost < 2.0 — early_stopping_rounds is a fit() kwarg
        m = xgb.XGBRegressor(**base)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              early_stopping_rounds=30, verbose=False)
    return m, m.predict(Xte)


def build_lstm(seq, n_feat, units=48, drop=0.3):
    inp = Input(shape=(seq, n_feat))
    x = LSTM(units, return_sequences=True)(inp); x = Dropout(drop)(x)
    x = LSTM(units//2)(x);                        x = Dropout(drop)(x)
    x = Dense(24, activation="relu")(x); out = Dense(1)(x)
    m = Model(inp, out); m.compile(Adam(1e-3), loss="mse", metrics=["mae"]); return m


def build_tft(seq, n_feat, d=48, heads=4, drop=0.2):
    inp = Input(shape=(seq, n_feat))
    g = Dense(d, activation="elu")(inp); g = Dropout(drop)(g)
    a = MultiHeadAttention(num_heads=heads, key_dim=d//heads, dropout=drop)(g, g)
    a = LayerNormalization()(Add()([g, a]))
    f = Dense(d, activation="relu")(a); f = Dropout(drop)(f)
    f = LayerNormalization()(Add()([a, f]))
    x = GlobalAveragePooling1D()(f); x = Dense(24, activation="relu")(x)
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


def metrics(yt, yp):
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    m = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    if len(yt) < 2:
        return dict(MAE=np.nan, MSE=np.nan, RMSE=np.nan, MAPE=np.nan, R2=np.nan)
    mae  = mean_absolute_error(yt, yp)
    mse  = mean_squared_error(yt, yp)
    rmse = math.sqrt(mse)
    nz   = yt != 0
    mape = np.mean(np.abs((yt[nz]-yp[nz])/yt[nz]))*100 if nz.any() else np.nan
    return dict(MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, R2=r2_score(yt, yp))


# ============================================================================
# EXPERIMENT
# ============================================================================
def run_experiment(env, malaria, h=1, seq_len=3, test_months=4):
    L(f"\n{'='*70}\n  h = {h} month(s)\n{'='*70}")
    df = build_features(malaria, env)
    df = add_target(df, h)
    df = df.dropna(subset=[f"Target_h{h}"]).reset_index(drop=True)
    L(f"  df after target: {df.shape}")
    if df.empty:
        raise RuntimeError("Empty df.")

    drop = {"District","Date","Malaria_Cases","Target_h1","Target_h2","Target_h3"}
    feats = [c for c in df.columns if c not in drop and df[c].dtype != "O"]
    for c in feats: df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feats] = df[feats].fillna(df[feats].median())
    L(f"  features: {len(feats)}")

    tr_parts, te_parts = [], []
    for d, g in df.groupby("District"):
        g = g.sort_values("Date")
        if len(g) < seq_len + test_months + 3:  continue
        tr_parts.append(g.iloc[:-test_months]); te_parts.append(g.iloc[-test_months:])
    if not tr_parts:
        raise RuntimeError("Not enough months per district.")
    train = pd.concat(tr_parts, ignore_index=True)
    test  = pd.concat(te_parts, ignore_index=True)
    L(f"  train {train.shape} | test {test.shape}")

    va_parts, tr2_parts = [], []
    for d, g in train.groupby("District"):
        g = g.sort_values("Date")
        if len(g) < seq_len + 3:  continue
        va_parts.append(g.iloc[-2:]); tr2_parts.append(g.iloc[:-2])
    tr2 = pd.concat(tr2_parts, ignore_index=True); val = pd.concat(va_parts, ignore_index=True)

    Xtr, ytr = tr2[feats].values, tr2[f"Target_h{h}"].values
    Xva, yva = val[feats].values, val[f"Target_h{h}"].values
    Xte, yte = test[feats].values, test[f"Target_h{h}"].values

    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(ytr.reshape(-1,1))
    results, preds = {}, {}

    L("  SARIMA ...")
    try:
        p = run_sarima(train, test)
        preds["SARIMA"] = p; results["SARIMA"] = metrics(yte, p)
    except Exception as e:
        L(f"    SARIMA failed: {e}")
        preds["SARIMA"] = np.full(len(yte), np.nan)
        results["SARIMA"] = metrics(yte, preds["SARIMA"])

    L("  Random Forest ...")
    _, p = run_rf(Xtr, ytr, Xte); preds["Random Forest"] = p
    results["Random Forest"] = metrics(yte, p)

    L("  XGBoost ...")
    try:
        _, p = run_xgb(Xtr, ytr, Xva, yva, Xte)
        preds["XGBoost"] = p; results["XGBoost"] = metrics(yte, p)
    except Exception as e:
        L(f"    XGBoost failed: {e}")
        preds["XGBoost"] = np.full(len(yte), np.nan)
        results["XGBoost"] = metrics(yte, preds["XGBoost"])

    L("  LSTM ...")
    try:
        m_lstm = build_lstm(seq_len, len(feats))
        p_lstm = run_dl(m_lstm, Xtr, ytr, Xva, yva, Xte, seq_len, sx, sy)
    except Exception as e:
        L(f"    LSTM failed: {e}"); p_lstm = np.full(len(yte), np.nan)
    preds["LSTM"] = p_lstm
    results["LSTM"] = metrics(yte[-len(p_lstm):], p_lstm)

    L("  TFT ...")
    try:
        m_tft = build_tft(seq_len, len(feats))
        p_tft = run_dl(m_tft, Xtr, ytr, Xva, yva, Xte, seq_len, sx, sy)
    except Exception as e:
        L(f"    TFT failed: {e}"); p_tft = np.full(len(yte), np.nan)
    preds["TFT"] = p_tft
    results["TFT"] = metrics(yte[-len(p_tft):], p_tft)

    L("  LSTM+TFT ...")
    if not np.isnan(p_lstm).all() and not np.isnan(p_tft).all():
        Llen = min(len(p_lstm), len(p_tft))
        p_hy = 0.5*p_lstm[-Llen:] + 0.5*p_tft[-Llen:]
    else:
        Llen = 0; p_hy = np.full(len(yte), np.nan)
    preds["LSTM+TFT"] = p_hy
    results["LSTM+TFT"] = metrics(yte[-len(p_hy):], p_hy)

    L("  Ensemble ...")
    rf_p = preds["Random Forest"]; xgb_p = preds["XGBoost"]
    if Llen > 0:
        try:
            Xva_q = seq_data(sx.transform(Xva), np.zeros(len(Xva)), seq_len)[0]
            vl = sy.inverse_transform(m_lstm.predict(Xva_q, verbose=0).reshape(-1,1)).ravel()
            vt = sy.inverse_transform(m_tft.predict(Xva_q, verbose=0).reshape(-1,1)).ravel()
            Lv = min(len(vl), len(vt)); yv = yva[-Lv:]
            errs = {"lstm": mean_absolute_error(yv, vl[-Lv:]),
                    "tft":  mean_absolute_error(yv, vt[-Lv:]),
                    "xgb":  mean_absolute_error(yva, xgb_p[:len(yva)] if len(xgb_p) >= len(yva) else xgb_p),
                    "rf":   mean_absolute_error(yva, rf_p[:len(yva)] if len(rf_p) >= len(yva) else rf_p)}
            inv = {k: 1/(v+1e-6) for k, v in errs.items()}
            tot = sum(inv.values()); w = {k: v/tot for k, v in inv.items()}
        except Exception as e:
            L(f"    weights fallback: {e}")
            w = {"lstm":0.25,"tft":0.25,"xgb":0.25,"rf":0.25}
        ens = (w["lstm"]*p_lstm[-Llen:] + w["tft"]*p_tft[-Llen:] +
               w["xgb"]*xgb_p[-Llen:] + w["rf"]*rf_p[-Llen:])
        preds["Ensemble"] = ens
        results["Ensemble"] = metrics(yte[-Llen:], ens)
        L(f"    weights: {w}")
    else:
        ens = 0.5*rf_p + 0.5*xgb_p
        preds["Ensemble"] = ens
        results["Ensemble"] = metrics(yte, ens)

    if Llen > 0:
        meta = test.iloc[-Llen:][["District","Date"]].reset_index(drop=True); y_al = yte[-Llen:]
    else:
        meta = test[["District","Date"]].reset_index(drop=True); y_al = yte
    pdf = meta.copy(); pdf["Actual"] = y_al
    for k, v in preds.items():
        pdf[k] = v[-len(pdf):] if len(v) >= len(pdf) else np.nan

    L("\n  RESULTS:")
    L(pd.DataFrame(results).T.round(4).to_string())
    save_outputs(results, pdf, h)
    return results, pdf


# ============================================================================
# SAVE OUTPUTS
# ============================================================================
def save_outputs(results, pdf, h):
    rows = [{"Model": m, "MAE": r.get("MAE"), "MSE": r.get("MSE"),
             "RMSE": r.get("RMSE"), "MAPE (%)": r.get("MAPE"),
             "R²": r.get("R2")} for m, r in results.items() if r]
    t = pd.DataFrame(rows).sort_values("RMSE")
    t.to_csv(os.path.join(TAB_DIR, f"Table_Metrics_h{h}.csv"), index=False)
    try:
        t.to_latex(os.path.join(TAB_DIR, f"Table_Metrics_h{h}.tex"),
                   index=False, float_format="%.3f",
                   caption=f"Experiment 1 — Metrics at h={h}",
                   label=f"tab:metrics_h{h}")
    except Exception: pass

    def _save(fig, name):
        for ext in ("png","pdf"):
            fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300)
        plt.close(fig)
        L(f"  saved {name}.png + .pdf")

    # Fig1
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=t, x="Model", y="RMSE", ax=ax[0], palette="viridis")
    ax[0].set_title(f"RMSE (h={h})"); ax[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=t, x="Model", y="MAE",  ax=ax[1], palette="magma")
    ax[1].set_title(f"MAE (h={h})");  ax[1].tick_params(axis="x", rotation=30)
    plt.tight_layout(); _save(fig, f"Fig1_MetricBars_h{h}")

    # Fig2
    dists = pdf["District"].unique()[:6]
    fig, axes = plt.subplots(len(dists), 1, figsize=(11, 2.3*len(dists)), sharex=True)
    if len(dists) == 1: axes = [axes]
    for ax, d in zip(axes, dists):
        s = pdf[pdf["District"]==d].sort_values("Date")
        ax.plot(s["Date"], s["Actual"], "k-o", label="Actual", markersize=5)
        for col, c in zip(["Random Forest","XGBoost","LSTM","TFT","Ensemble"],
                          ["tab:blue","tab:orange","tab:green","tab:red","tab:brown"]):
            if col in s.columns and s[col].notna().any():
                ax.plot(s["Date"], s[col], "--", label=col, linewidth=1.2)
        ax.set_title(d, fontsize=10); ax.set_ylabel("Cases"); ax.grid(alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Date")
    plt.tight_layout(); _save(fig, f"Fig2_TimeSeries_h{h}")

    # Fig3
    if "Ensemble" in pdf.columns and pdf["Ensemble"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.scatter(pdf["Actual"], pdf["Ensemble"], alpha=0.55, s=28,
                   edgecolor="k", linewidth=0.3)
        lim = [0, max(pdf["Actual"].max(), pdf["Ensemble"].max())*1.05]
        ax.plot(lim, lim, "r--"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted (Ensemble)")
        ax.set_title(f"Actual vs Ensemble (h={h})"); ax.grid(alpha=0.3)
        plt.tight_layout(); _save(fig, f"Fig3_Scatter_h{h}")

    # Fig4
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for col in ["Random Forest","XGBoost","LSTM","TFT","Ensemble"]:
        if col in pdf.columns and pdf[col].notna().any():
            sns.kdeplot(pdf["Actual"]-pdf[col], ax=ax, label=col, linewidth=1.4)
    ax.axvline(0, color="k", linestyle=":"); ax.set_xlabel("Residual")
    ax.set_title(f"Residuals (h={h})"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); _save(fig, f"Fig4_Residuals_h{h}")

    # Fig5
    heat = t.set_index("Model")[["MAE","RMSE","MAPE (%)","R²"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"Performance heatmap (h={h})")
    plt.tight_layout(); _save(fig, f"Fig5_Heatmap_h{h}")

    pdf.to_csv(os.path.join(TAB_DIR, f"Predictions_h{h}.csv"), index=False)
    L(f"  saved outputs h={h}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    dump_all(FILEPATH)

    L("\n[STEP 1] Malaria")
    try:
        malaria = load_malaria(FILEPATH)
        if not malaria.empty:
            malaria.to_csv(os.path.join(DATA_DIR, "Extracted_Malaria.csv"), index=False)
    except Exception as e:
        L(f"!!! Malaria failed: {e}"); traceback.print_exc()
        malaria = pd.DataFrame()

    L("\n[STEP 2] Environment")
    try:
        env, precip, temp, hum, lake, river = build_env(FILEPATH)
        if not env.empty:
            env.to_csv(os.path.join(DATA_DIR, "Extracted_Env_Merged.csv"), index=False)
        if not precip.empty: precip.to_csv(os.path.join(DATA_DIR, "Extracted_Rainfall.csv"), index=False)
        if not temp.empty:   temp.to_csv(os.path.join(DATA_DIR, "Extracted_Temperature.csv"), index=False)
        if not hum.empty:    hum.to_csv(os.path.join(DATA_DIR, "Extracted_Humidity.csv"), index=False)
        if not lake.empty:   lake.to_csv(os.path.join(DATA_DIR, "Extracted_LakeLevel.csv"), index=False)
        if not river.empty:  river.to_csv(os.path.join(DATA_DIR, "Extracted_RiverLevel.csv"), index=False)
    except Exception as e:
        L(f"!!! Env failed: {e}"); traceback.print_exc()
        env, precip, temp, hum, lake, river = [pd.DataFrame()]*6

    try:
        plot_raw(precip, temp, hum, lake, river, malaria)
    except Exception as e:
        L(f"!!! Plot failed: {e}"); traceback.print_exc()

    if not malaria.empty and not env.empty:
        all_results = {}
        for h in (1, 2, 3):
            try:
                res, _ = run_experiment(env, malaria, h=h, seq_len=3, test_months=4)
                all_results[h] = res
            except Exception as e:
                L(f"!!! h={h} failed: {e}"); traceback.print_exc()

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
            L("\n" + "="*70 + "\n  FINAL SUMMARY\n" + "="*70)
            L(c.round(4).to_string(index=False))
    else:
        L("\n!!! Cannot train: malaria or env is empty.")
        L("    Look inside Experiment1_Outputs/Diagnostics/*.csv")

    L(f"\nAll outputs in: {os.path.abspath(OUT_DIR)}")