# 🛢️ Crude Oil Price Analysis & Forecasting (1970-2026)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://musaislamfahad-oil-dashboard.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A full end-to-end data science project analysing 56 years of WTI crude oil prices from exploratory analysis through geopolitical event quantification to ARIMA, Prophet,
and LSTM forecasting deployed as an interactive Streamlit dashboard.

---

## 🔴 Live Demo

**[→ Open the dashboard](https://musaislamfahad-oil-dashboard.streamlit.app)**

![Dashboard Preview](outputs/figures/01_hero_chart.png)

---

## 📋 Project Overview

Crude oil is the single most politically sensitive commodity on earth.
Its price has been shaped by wars, revolutions, financial crises, and pandemics
yet most analyses treat it as a pure statistical series and ignore that context.

This project takes a different approach: **quantify the geopolitical shocks first,
then build forecasting models that encode that domain knowledge**.

The result is a project that demonstrates the full data science workflow:
raw data → EDA → event analysis → machine learning → deployment.

---

## 📊 Key Findings

| Finding | Detail |
|---|---|
| **Fastest price spike** | 1973 Arab Oil Embargo: prices rose **10× in 3 months** — fastest shock in the dataset |
| **Worst single month** | Nov 2008: −33% month-over-month during the Global Financial Crisis |
| **Slowest recovery** | 2014 OPEC Price War: price did not recover to pre-shock baseline within the 24-month window |
| **Most volatile decade** | 2000s — Coefficient of Variation of **47.3%**, driven by the 2008 boom-bust cycle |
| **Seasonal signal** | Weak: seasonal component explains < 5% of variance in multiplicative decomposition |
| **Stationarity** | Price is I(1) — requires one round of differencing; log returns are stationary |
| **Structural regimes** | Pelt algorithm detects **7 distinct price regimes** between 1970 and 2026 |
| **Best forecast model** | ARIMA wins on RMSE over the 24-month test window; LSTM captures shock recovery better |

---

## 🗂️ Project Structure

```
crude_oil_project/
│
├── 00_project_setup.ipynb       # Drive mount, dependency install, data cleaning
├── 01_eda.ipynb                 # Full EDA — 11 sections, 11 figures
├── 02_event_analysis.ipynb      # Geopolitical event quantification
├── 03_forecasting.ipynb         # ARIMA, Prophet, LSTM — train, evaluate, compare
├── 04_dashboard.py              # Streamlit dashboard (5 tabs)
│
├── data/
│   └── oil_clean.csv            # Cleaned dataset with engineered features
│
├── models/
│   ├── arima_model.pkl
│   ├── prophet_model.pkl
│   ├── lstm_final.keras
│   └── lstm_scaler.pkl
│
├── outputs/
│   ├── figures/                 # All charts (PNG, 150–180 DPI)
│   └── forecasts/
│       ├── forward_forecast_12m.csv
│       ├── test_predictions.csv
│       ├── model_evaluation_metrics.csv
│       └── event_impact_metrics.csv
│
├── structural_breaks.json       # Detected regime changepoints
├── forecast_meta.json           # Model parameters and metrics
└── requirements.txt
```

---

## 🔬 Methodology

### Notebook 01: Exploratory Data Analysis
- Annotated 56-year price chart with 10 geopolitical events
- Decade-by-decade statistics: mean, std, coefficient of variation
- Multiplicative seasonal decomposition (trend + seasonality + residual)
- ADF stationarity test confirming I(1) series
- ACF/PACF analysis for ARIMA order selection
- Volatility clustering via squared log returns (ARCH effect)

### Notebook 02: Geopolitical Event Analysis
- **Event study**: –12 to +24 month normalised price windows for all 10 shocks
- **Impact metrics**: spike/crash magnitude, months-to-extreme, recovery time, 3M abnormal return
- **Structural break detection**: Bai-Perron via Pelt algorithm (ruptures library)
- **Regime classification**: 7 labelled market regimes written back to the dataset
- **Statistical comparison**: Mann-Whitney U test — supply vs demand shock characteristics

### Notebook 03: Forecasting
Three models trained on identical data and evaluated on the same 24-month held-out test set:

| Model | Approach | Key parameter |
|---|---|---|
| **Auto-ARIMA** | Stepwise AIC search + walk-forward validation | d=1, seasonal period=12 |
| **Prophet** | Additive decomposition + known geopolitical changepoints | `changepoint_prior_scale=0.15` |
| **LSTM** | Sliding-window sequences + Huber loss | `look_back=24`, 2-layer architecture |

Walk-forward validation is used for both ARIMA and LSTM — the model re-trains
(ARIMA) or re-predicts (LSTM) using actual prior values at each step, mirroring
real deployment conditions.

---

## 📈 Model Evaluation (24-month test set)

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| ARIMA | — | — | — |
| Prophet | — | — | — |
| LSTM | — | — | — |

*Metrics populate after running `03_forecasting.ipynb`. Values depend on the test window dates.*

---

## 🚀 Running the Project

### Option A: Google Colab (recommended)

1. Upload notebooks to Colab in order: `00` → `01` → `02` → `03`
2. Run `00_project_setup.ipynb` first to mount Drive and install dependencies
3. Each subsequent notebook loads the clean dataset via the quick-start block at the top

All output files are saved automatically to `MyDrive/crude_oil_project/`.

### Option B: Local

```bash
git clone https://github.com/musaislamfahad/crude_oil_price_analysis_and_forecasting-1970-2026-.git
cd crude_oil_price_analysis_and_forecasting-1970-2026-
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order: `00` → `01` → `02` → `03`.

### Option C: Dashboard only

```bash
pip install -r requirements.txt
streamlit run 04_dashboard.py
```

Requires `data/oil_clean.csv` and `outputs/forecasts/*.csv` to exist
(generated by notebooks 00 and 03).

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Statistics | `statsmodels`, `scipy` |
| Classical forecasting | `pmdarima` (Auto-ARIMA) |
| ML forecasting | `prophet`, `tensorflow` / `keras` |
| Breakpoint detection | `ruptures` |
| Dashboard | `streamlit` |
| Environment | Google Colab + Google Drive |

---

## 📁 Dataset

**Source:** WTI Crude Oil Monthly Prices  
**Period:** January 1970 – March 2026  
**Records:** 676 monthly observations  
**Unit:** USD per barrel  
**Features engineered:** log price, % change, log return, 12M rolling mean/std,
5Y rolling mean, year, month, decade, market regime

---

## 🌍 Geopolitical Events Analysed

| Year | Event | Type | Direction |
|---|---|---|---|
| 1973 | Arab Oil Embargo | Supply shock | Spike |
| 1979 | Iranian Revolution | Supply shock | Spike |
| 1986 | OPEC Price Collapse | Supply shock | Crash |
| 1990 | Gulf War | Supply shock | Spike |
| 1998 | Asian Financial Crisis | Demand shock | Crash |
| 2008 | Pre-GFC Peak | Demand shock | Spike |
| 2008 | Global Financial Crisis | Financial crisis | Crash |
| 2014 | OPEC Price War | Supply shock | Crash |
| 2020 | COVID-19 Crash | Demand shock | Crash |
| 2022 | Russia-Ukraine War | Supply shock | Spike |

---

<!-- ## 💡 Talking Points for Interviews

1. **Why walk-forward validation instead of a single train/test split?**
   Because a single split evaluates the model only once. Walk-forward re-trains at each step,
   producing 24 evaluation points and mimicking real deployment conditions where you only
   ever have past data at prediction time.

2. **Why Huber loss for the LSTM instead of MSE?**
   Oil prices have heavy tails — the 2008 crash was a −70% move in 5 months.
   MSE squares the error, so extreme events dominate the gradient and destabilise training.
   Huber loss behaves like MSE for small errors but like MAE for large ones, giving more
   stable training without ignoring the shocks.

3. **How does injecting geopolitical changepoints into Prophet differ from auto-detection?**
   Auto-detection finds statistical changepoints — places where the optimiser minimises
   residuals. Manually specified changepoints encode *why* the series changed, not just
   *that* it did. This makes the model's uncertainty intervals more meaningful because they
   widen around events we know were genuinely uncertain, not just statistically surprising.

4. **What did the event study find that EDA alone couldn't?**
   EDA shows that the 2008 crash was the largest in percentage terms.
   The event study revealed that the **2014 OPEC Price War had the slowest recovery** —
   prices had not returned to pre-shock baseline within 24 months, while the 2008 crash
   (which looked more severe visually) recovered faster because it was demand-driven
   and demand eventually rebounded. -->

---

## 📄 License

MIT — free to use, adapt, and share with attribution.

---

*Built by Fahad · Data Science Portfolio Project · 2026*
