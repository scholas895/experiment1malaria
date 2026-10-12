# Methodology - CPST-MalariaNet

This document is **plain documentation**. It is not code and must never be
executed inside Jupyter. Open it with a Markdown viewer or on GitHub.

## 1. Problem statement

Let y(d,t) denote the number of confirmed malaria cases in health district d
at month t, for d = 1 ... 49 districts of Burundi and t covering
Jan 2022 - Dec 2024. Given a covariate vector x(d,t), we solve two tasks:

- Forecasting (Experiment 1): estimate y_hat(d, t+h) for h in {1, 2, 3}.
- Early warning (Experiment 2): classify whether a district-month is in an
  outbreak state, defined as y(d,t) > tau, where tau is the 75th percentile
  of the observed case distribution at that horizon.

## 2. Data

Three Excel sheets (2022, 2023, 2024) each contain one row per district with
12 monthly case counts. Deduplicated on (District, Year, Month): 1 764
district-months across 49 districts.

Environmental drivers aggregated to national monthly means:
rainfall, temperature, humidity, lake level, river level/discharge.

## 3. Feature engineering

Per district, past-only features:

- Month_Sin, Month_Cos
- Cases_Lag1, Cases_Lag2, Cases_Lag3
- Cases_Roll3, Cases_Roll6
- {Env}_Lag1, {Env}_Lag2
- Target_h{h} = cases shifted -h months

## 4. Models

- SARIMA(1,1,1)(1,1,1,12) per district
- Random Forest (300 trees, depth 12)
- XGBoost (500 trees, depth 6, early-stopped)
- LSTM (48 + 24 units, dropout 0.3)
- Simplified TFT (Dense + MultiHeadAttention + residual + pooling)
- LSTM+TFT (arithmetic mean)
- Inverse-error weighted ensemble of LSTM, TFT, XGBoost, RF

## 5. Evaluation

Splits per district (chronological, no shuffling):

- test: last 4 months
- train: remaining months
- inner validation: last 2 months of train
- train': remaining months of train

Regression: MAE, MSE, RMSE, MAPE, R2.

Early warning: Sensitivity, Specificity, Precision, F1, False-Alarm Rate,
Missed-Alert Rate, Brier, AUC-ROC, AUPRC, Lead Time, Detection Rate.
The continuous prediction is mapped to a probability via
p_hat = sigmoid((y_hat - tau) / s), with s = std of observed cases.

## 6. Reproducibility

Seeds: NumPy and TensorFlow seeded with 42.
Entry point: python scripts/run_all.py

## 7. Limitations

1. National environmental aggregates (no station-to-district mapping).
2. Only 36 months of malaria data (short history for deep learning).
3. P75 outbreak threshold is population-dependent.
4. Districts modelled independently (no spatial structure).
