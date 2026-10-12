# CPST-MalariaNet

Hybrid deep-learning and statistical framework for malaria forecasting and
early-warning in Burundi.

## Overview

Combines SARIMA, Random Forest, XGBoost, LSTM, a simplified TFT, and a
weighted ensemble to forecast monthly malaria incidence across the 49 health
districts of Burundi.

## Layout

    CPST-MalariaNet/
    ├── src/            reusable Python modules
    ├── scripts/        command-line entry points
    ├── docs/           methodology notes
    ├── data/raw/       place BdiDATASETS.xlsx here
    ├── outputs/        generated tables, figures, predictions
    └── figures/        curated publication figures

## Installation

    pip install -r requirements.txt

## Run

    python scripts/run_all.py --data data/raw/BdiDATASETS.xlsx --out outputs

## License

MIT. See LICENSE.

## Contact

GitHub: https://github.com/scholas895
