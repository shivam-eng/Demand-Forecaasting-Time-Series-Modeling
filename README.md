# 📦 Demand Forecasting & Time Series Modeling
### Walmart M5 Sales Forecasting — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Walmart%20M5-orange)
![Models](https://img.shields.io/badge/Models-SARIMA%20%7C%20Prophet%20%7C%20LSTM-purple)

---

## 📋 Overview

This is my first end-to-end demand forecasting project, built as part of a structured learning path in data science and time series modeling. The goal was to take real-world retail sales data from Walmart's M5 Forecasting Competition and build a complete pipeline — from raw data ingestion all the way through model comparison and business impact analysis.

The project follows the full lifecycle of a forecasting system:

- Loading and transforming 58 million rows of transactional data across 10 Walmart stores
- Performing thorough exploratory data analysis to understand trends, seasonality, and anomalies
- Building and comparing four forecasting models of increasing sophistication
- Translating model accuracy into concrete business value
- Designing how the system would operate in a real production environment

This project was built entirely using free and open-source tools. No paid APIs, platforms, or services were used at any stage.

---

## ❓ Business Questions Answered

| # | Question | Answer Found |
|---|---|---|
| 1 | How much does Walmart sell on an average day across all stores? | ~38,000 units/day across 10 stores |
| 2 | Which day of the week drives the most sales? | Saturday — consistently the peak day across all years |
| 3 | Which product category dominates sales volume? | FOODS (~70% of all units sold) |
| 4 | Do SNAP food assistance days affect sales? | Yes — approximately 12.7% lift on SNAP days |
| 5 | Which events have the biggest impact on demand? | Labor Day and Super Bowl drive the highest lifts |
| 6 | What is the most accurate forecasting model for this dataset? | Prophet — 4.71% MAPE over a 28-day horizon |
| 7 | How much does better forecasting save in inventory costs? | ~$3,290/day or ~$1.2M/year vs naive approach |
| 8 | Does model complexity always mean better accuracy? | No — LSTM (most complex) was outperformed by Prophet and SARIMA |

---

## 🛠️ Tools & Libraries Used

| Tool / Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core programming language |
| pandas | 2.x | Data manipulation and transformation |
| numpy | 1.24+ | Numerical operations |
| matplotlib | Latest | Visualisation and plotting |
| seaborn | Latest | Statistical visualisation |
| statsmodels | 0.14+ | SARIMA model, ADF test, decomposition |
| prophet | 1.1+ | Facebook Prophet forecasting model |
| tensorflow / keras | 2.13+ | LSTM deep learning model |
| scikit-learn | 1.3+ | Metrics (MAE, RMSE, MAPE), preprocessing |
| scipy | Latest | Z-score anomaly detection |
| jupyter | Latest | Notebook environment |
| VS Code | Latest | Primary development environment |

---

## 📂 Dataset

**Source:** [Walmart M5 Forecasting Competition — Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

The M5 dataset is one of the most widely used real-world retail forecasting benchmarks. It covers daily sales data from 10 Walmart stores across 3 US states.

| File | Rows | Columns | Description |
|---|---|---|---|
| `sales_train_validation.csv` | 30,490 | 1,919 | One row per product-store pair, one column per day (wide format) |
| `calendar.csv` | 1,969 | 14 | Maps day index (d_1 … d_1969) to calendar dates, events, SNAP flags |
| `sell_prices.csv` | ~6.8M | 4 | Weekly sell price per product per store |

**After transformation:**
- Melted to long format: **58,327,370 rows × 19 columns**
- Date range: **29 January 2011 → June 2016**
- Stores: **10 (CA×4, TX×3, WI×3)**
- Products: **3,049 unique SKUs**
- Categories: **FOODS, HOUSEHOLD, HOBBIES**

> ⚠️ The data files are not included in this repository due to file size. Download them directly from the Kaggle link above and place them in the `/data` folder before running the notebook.

---

## 📊 Key Results

### Model Performance — 28-Day Forecast Horizon

| Model | MAE | RMSE | MAPE | Daily Error (units) | Daily Inventory Risk |
|---|---|---|---|---|---|
| 🥇 Prophet | 2,068.6 | 2,679.3 | 4.71% | ~1,790 units | ~$3,580 |
| 🥈 SARIMA | 2,309.8 | 2,794.9 | 5.44% | ~2,067 units | ~$4,134 |
| 🥉 LSTM | 3,288.3 | 3,793.7 | 7.58% | ~2,879 units | ~$5,758 |
| Naive Baseline | 4,061.0 | 6,160.1 | 9.04% | ~3,435 units | ~$6,870 |

*Inventory risk calculated at $2/unit holding cost*

### Key Findings

- **Prophet outperformed all models** including the more complex LSTM — demonstrating that model fit to data characteristics matters more than model complexity
- **Adding external regressors** (SNAP days + event calendar) improved Prophet's MAPE by **1.32 percentage points** — approximately 500 fewer forecast errors per day
- **All three trained models beat the naive baseline** — confirming genuine learning occurred
- **LSTM underperformed** at this scale due to limited training data (1,885 days), no external regressors, and 20 epochs being insufficient for convergence
- **7 anomalies detected** — 5 Christmas Day store closures (z = -4.67) and 2 genuine demand spikes in March/April 2016 (z ≈ +3.1)
- **Estimated annual saving** from Prophet vs naive approach: **~$1.2 million** across 10 stores

### EDA Highlights

- Saturday is the highest-selling day — ~45,000 avg units vs ~29,000 on Tuesday
- FOODS category accounts for ~70% of all units sold
- HOBBIES appeared flat on shared-axis charts but sells 2,000–4,000 units/day with real growth
- A structural trend break is visible in late 2012/early 2013 across all three categories simultaneously — likely a store expansion event
- Labor Day and Super Bowl generate the highest positive sales lifts of all calendar events

---

## 📁 Folder Structure

```
Demand-Forecasting-Time-Series-Modeling/
│
├── data/                          ← Place Kaggle M5 files here (not tracked by git)
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   └── sell_prices.csv
│
├── notebooks/
│   └── demand_forecasting.ipynb   ← Main project notebook (all steps)
│
├── outputs/
│   ├── plots/                     ← All saved EDA and forecast charts
│   └── results/
│       └── model_comparison.csv   ← MAE, RMSE, MAPE for all models
│
├── README.md                      ← This file
├── requirements.txt               ← All Python dependencies
└── .gitignore                     ← Excludes data files and environment folders
```

---

## ⚙️ Local Setup & Installation

Follow these steps exactly in order to run the project on your own machine.

### Prerequisites

- Python 3.10 or higher installed
- VS Code installed with the **Python** and **Jupyter** extensions
- A free Kaggle account (to download the dataset)
- Git installed on your machine

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Demand-Forecasting-Time-Series-Modeling.git
cd Demand-Forecasting-Time-Series-Modeling
```

---

### Step 2 — Create and Activate Virtual Environment

**Windows (VS Code terminal):**
```bash
python -m venv ts_env
ts_env\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv ts_env
source ts_env/bin/activate
```

You should see `(ts_env)` appear at the start of your terminal prompt confirming the environment is active.

---

### Step 3 — Install All Dependencies

```bash
pip install -r requirements.txt
```

This installs all required libraries in one command. Expected install time: 3–5 minutes depending on connection speed.

---

### Step 4 — Download the Dataset

1. Go to [https://www.kaggle.com/competitions/m5-forecasting-accuracy/data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
2. Accept the competition rules
3. Download these three files:
   - `sales_train_validation.csv`
   - `calendar.csv`
   - `sell_prices.csv`
4. Place all three files inside the `/data` folder in the project directory

> ⚠️ Do not rename the files. The notebook references them by their exact original names.

---

### Step 5 — Select the Correct Kernel in VS Code

1. Open `notebooks/demand_forecasting.ipynb` in VS Code
2. Click **"Select Kernel"** in the top right corner
3. Choose **ts_env** from the dropdown list
4. If ts_env does not appear, press `Ctrl+Shift+P` → type **"Python: Select Interpreter"** → navigate to `ts_env/Scripts/python.exe`

---

### Step 6 — Run the Notebook

Run all cells from top to bottom using **Run All** (`Ctrl+F9`) or run each cell individually with `Shift+Enter`.

Expected total runtime: **15–25 minutes** depending on hardware (LSTM training is the slowest step).

---

### Common Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `MemoryError` on price merge | Wrong join keys | Ensure merge uses `['store_id','item_id','wm_yr_wk']` — all three keys |
| `ValueError: merging datetime64 and str` | Calendar date not converted | Add `calendar['date'] = pd.to_datetime(calendar['date'])` before merge |
| `TypeError: fillna() method argument` | pandas 2.x removed method param | Replace `.fillna(method='bfill')` with `.bfill()` |
| Prophet import error | Wrong package name | Install `prophet` not `fbprophet` — package was renamed |
| TensorFlow import warning in VS Code | Pylance static analysis limitation | Use `import tensorflow as tf` then `tf.keras.layers.LSTM(...)` |
| ts_env not showing in kernel list | Interpreter not registered | Run `python -m ipykernel install --user --name=ts_env` in terminal |

---

## 📄 Requirements File

The `requirements.txt` file contains:

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
prophet>=1.1.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
scipy>=1.10.0
jupyter>=1.0.0
notebook>=7.0.0
kaggle>=1.5.0
ipykernel>=6.0.0
```

---

## 📖 Project Notebook Structure

The main notebook `demand_forecasting.ipynb` is organised into 10 sequential steps:

| Step | Title | Description |
|---|---|---|
| 1 | Environment Setup | Library imports, seed setting, display config |
| 2 | Data Acquisition & Transformation | Load 3 files, melt wide→long, merge calendar and prices |
| 3 | Exploratory Data Analysis | 8 plots covering trends, seasonality, events, SNAP, anomalies |
| 4 | Train / Test Split | Temporal 80/20 split — last 28 days as test |
| 5 | SARIMA Model | ADF test, ACF/PACF, SARIMA(1,1,1)(1,1,1,7) fit and forecast |
| 6 | Prophet Model | Fit with SNAP + event regressors, component plots |
| 7 | LSTM Model | Sequence windowing, architecture, training, evaluation |
| 8 | Model Comparison | MAE/RMSE/MAPE table, combined forecast plot, analysis |
| 9 | Business Impact | Cost quantification, team stakeholders, savings estimate |
| 10 | Production Design | Pipeline, retraining, drift detection, alerting, serving |

---

## 🙋 About This Project

This project was completed as part of a structured self-learning curriculum in data science and machine learning. It is my first time working with a dataset of this scale (58 million rows), implementing time series models from scratch, and thinking about how a model would operate beyond the notebook environment.

Key personal learnings from this project:
- Data preparation takes longer than modelling — the melt, merge, and type conversion steps required careful debugging
- Model complexity does not guarantee better results — Prophet (simpler) outperformed LSTM (more complex) on this dataset
- Aggregation level matters enormously — several charts showed misleading ~1.0 averages until I fixed the groupby level
- Anomalies tell a story — the Christmas closures and 2016 spikes were not data errors but genuine business events worth understanding

---

*Dataset credit: Walmart M5 Forecasting Competition, hosted on Kaggle*
*Tutorial reference: https://youtu.be/OfkYUaCp3mc — Segment 4:36–6:49*
