# USD/UZS Moving Average Crossover Backtester

A quantitative trading strategy backtester for the USD/UZS (US Dollar / Uzbek Som) exchange rate — built to demonstrate practical skills in emerging-market FX data sourcing, systematic trading strategy design, and risk-adjusted performance evaluation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75)
![Status](https://img.shields.io/badge/Status-Educational%20Research%20Project-yellow)

---

## Project Overview

This project builds a fully reproducible backtesting pipeline for a moving-average (MA) crossover trading strategy applied to the USD/UZS exchange rate — one of Central Asia's least-covered, least-liquid currency pairs in mainstream financial datasets. The notebook pulls ~5 years of official daily exchange rate data, engineers technical features, simulates a rules-based trading strategy day-by-day with no look-ahead bias, and benchmarks it against a buy-and-hold baseline using institutional-grade risk metrics (Sharpe ratio, max drawdown, win rate).

The project is built entirely in **Google Colab** with zero paid APIs and zero API keys required.

## Real-World Finance Use Case

Emerging-market (EM) currencies like the Uzbek Som present a different risk/return profile than G10 pairs (EUR/USD, USD/JPY, etc.):

- **Managed float regimes**: many EM central banks (including the CBU) intervene more actively, producing trend-like behavior that momentum/trend-following strategies are specifically designed to capture.
- **Thin data infrastructure**: unlike majors, EM pairs are often absent from free-tier financial APIs entirely, forcing analysts to build their own data pipelines from primary central bank sources — a skill directly transferable to EM trading desks, sovereign wealth funds, and frontier-market hedge funds.
- **Devaluation risk and trend persistence**: currencies like the Som have experienced sustained multi-year depreciation trends against the USD, punctuated by sharp step-devaluations (as in 2017) — exactly the kind of regime where trend-following strategies can meaningfully outperform a static buy-and-hold position, while also carrying real drawdown risk that must be measured, not assumed away.
- **Relevance to Uzbekistan's financial development**: as Uzbekistan's capital markets liberalize and local asset management grows, tools like this are directly applicable to corporate treasury hedging, remittance timing, and local investment fund strategy design.

This project demonstrates the exact workflow a junior quant analyst or treasury associate would be asked to prototype: source the data, build the signal, prove it (or disprove it) with a rigorous backtest, and present the results clearly.

## System Architecture

```
┌───────────────────────┐     ┌───────────────────────┐     ┌─────────────────────┐
│   DATA COLLECTION     │ --> │   DATA PROCESSING     │ --> │   STRATEGY ENGINE   │
│  CBU.uz API (primary) │     │  Clean, forward-fill, │     │  MA crossover signal│
│  exchangerate.host    │     │  feature engineering  │     │  generation         │
│  (fallback)           │     │  (returns, SMAs, vol) │     │                     │
│  Synthetic (last      │     │                       │     │                     │
│  resort, for demo)    │     │                       │     │                     │
└───────────────────────┘     └───────────────────────┘     └─────────┬───────────┘
                                                                      │
                                                                      v
┌───────────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│     VISUALIZATION     │ <-- │  PERFORMANCE METRICS  │ <-- │  BACKTEST ENGINE     │
│  Matplotlib (static)  │     │  Sharpe, drawdown,    │     │  No look-ahead bias, │
│  Plotly (interactive) │     │  win rate, returns    │     │  signal lag, txn     │
│  dashboard            │     │                       │     │  costs, equity curve │
└───────────────────────┘     └───────────────────────┘     └──────────────────────┘
```

## Required APIs and Data Sources

| Priority | Source | Notes |
|---|---|---|
| **Primary** | [Central Bank of Uzbekistan (cbu.uz)](https://cbu.uz/en/arkhiv-kursov-valyut/veb-masteram/) | Free, no API key. Official daily rate. Endpoint only supports **one date per request** (`/json/{CCY}/{YYYY-MM-DD}/`), so the pipeline loops over each calendar day. |
| **Secondary (fallback)** | [exchangerate.host](https://exchangerate.host) | Free timeseries endpoint, no key required for basic use; mirrors central-bank reference rates including UZS. |
| **Tertiary (fallback)** | Synthetic data generator (built into the notebook) | Calibrated random-walk simulation so the **entire pipeline still runs end-to-end** even with no internet access — useful for demos, grading, or offline development. Clearly labeled as non-real data everywhere it appears. |
| **Manual alternative** | [investing.com](https://www.investing.com) / [xe.com](https://www.xe.com) historical CSV export | Download manually, then `from google.colab import files; files.upload()` to bring it into the notebook if both APIs are unavailable. |

## Required Python Libraries

```
requests      # HTTP calls to the CBU API
pandas        # time series data manipulation
numpy         # numerical computation (returns, log returns, stats)
matplotlib    # static charts
plotly        # interactive dashboard
scipy         # (available for any future statistical extensions)
```

All are pre-installed in Google Colab; the notebook also `pip install`s them explicitly for portability to any local Jupyter environment.

## Folder / File Structure

Even though this is built and run inside a single Google Colab notebook, the logical structure is:

```
usd-uzs-backtester/
├── USDUZS_Backtester.py        # Main notebook (cell-segmented .py — paste into Colab)
├── README.md                   # This file
├── usduzs_backtest_results.csv # Generated on run: full daily backtest output
└── usduzs_performance_metrics.csv # Generated on run: summary metrics table
```

## Step-by-Step Build Guide

1. **Open Google Colab** → New Notebook.
2. **Copy each `# %%` block** from `USDUZS_Backtester.py` into its own Colab cell (or paste the whole file into one cell and use Colab's "Split cell" feature at each `# %%` marker).
3. **Run cells top to bottom** (`Runtime > Run all`). No credentials needed.
4. **(Optional) Adjust `CONFIG`** at the top — date range, MA windows, starting capital, transaction costs — and re-run to test different strategy parameters.
5. **Review the printed metrics table and charts** at the bottom of the notebook.
6. **Download the CSV outputs** for use in a report, slide deck, or GitHub repo.

> **Heads-up on runtime**: pulling 5 years of real CBU data takes 20–40+ minutes because the API only supports one date per HTTP request. For a quick test run, temporarily shrink the date range (e.g. to 6 months) before running the full pipeline.

## Data Collection Pipeline

The pipeline tries three sources in order, falling back automatically:

1. Loop over every calendar day in the configured range, hitting the CBU endpoint once per day, with retry logic and a circuit-breaker (stop after 30 consecutive failures rather than burning through the whole range).
2. If CBU returns insufficient data, fall back to `exchangerate.host`'s batch timeseries endpoint.
3. If both fail, generate a calibrated synthetic series so the rest of the notebook can still be demonstrated — clearly labeled as synthetic in all output and chart titles.

## Data Cleaning & Feature Engineering

- Sort, deduplicate, and reindex to every calendar day; forward-fill non-publishing days (weekends/holidays) — standard convention for FX and bond data.
- **Daily returns** and **log returns** (log returns used for risk metrics since they're time-additive).
- **Short and long simple moving averages** (default: 20-day / 50-day).
- **Rolling 20-day annualized volatility** for EM risk context.

## Core Strategy Logic / Backtesting Engine

- **Signal**: +1 (long USD) when the short MA is above the long MA; -1 (flat) otherwise.
- **No look-ahead bias**: positions are lagged by one day — a signal generated at today's close is acted on starting tomorrow.
- **Transaction costs**: configurable basis-point cost applied on every position change.
- **Equity curve**: capital compounds daily based on realized strategy returns, simulated alongside a buy-and-hold baseline over the identical period.

## Visualizations & Dashboard Components

1. Price chart with both moving averages and buy/sell signal markers.
2. Equity curve: strategy vs. buy-and-hold portfolio value over time.
3. Drawdown chart: peak-to-trough decline over time for both approaches.
4. A combined interactive Plotly dashboard (all three, stacked, zoomable, hoverable).

## Performance Metrics

- Total Return (%)
- Annualized Return (%)
- Annualized Volatility (%)
- Sharpe Ratio (vs. configurable annual risk-free rate)
- Max Drawdown (%)
- Win Rate (%)
- Number of Trades

## Final Deliverables

- A complete, runnable Colab notebook (`USDUZS_Backtester.py`)
- Two exportable CSVs: full daily backtest results and a summary metrics table
- Three static charts + one interactive dashboard
- This README, ready to drop into a GitHub repository

## Resume Description

> Built a Python-based quantitative backtesting engine for the USD/UZS exchange rate, sourcing 5 years of official daily rates directly from the Central Bank of Uzbekistan's public API; engineered a moving-average crossover strategy and benchmarked it against buy-and-hold using Sharpe ratio, max drawdown, and win-rate analysis, with transaction-cost modeling and bias-free signal lagging.

*(Trim to 1–2 lines depending on your resume format; the above is ~2 lines at standard resume font sizes.)*

## Potential Upgrades

- **More currency pairs**: extend to EUR/UZS, RUB/UZS, or a basket-relative strategy.
- **Machine learning signals**: replace the MA crossover with a logistic regression or gradient-boosted classifier trained on lagged returns, volatility regimes, and macro features.
- **Walk-forward optimization**: re-tune MA windows on rolling training windows instead of one fixed in-sample choice, to test robustness out-of-sample.
- **Realistic execution modeling**: bid/ask spread modeling instead of a flat bps cost, and slippage based on the CBU's actual two-way quote (if/when published).
- **Macro overlay**: incorporate Uzbekistan CPI, CBU policy rate decisions, or remittance flow data as additional model features.
- **Live paper-trading dashboard**: deploy as a Streamlit app that refreshes daily from the CBU API and tracks live (simulated) P&L.

---

### Disclaimer

This project is for educational and research purposes only. It does not constitute investment advice. Backtested performance on historical data does not guarantee future results, and the strategy has not been evaluated for live trading with real capital, real bid/ask spreads, or regulatory considerations relevant to FX trading in Uzbekistan.
