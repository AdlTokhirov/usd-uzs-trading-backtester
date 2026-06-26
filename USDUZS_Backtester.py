# %% [markdown]
"""
# USD/UZS Moving-Average Crossover Backtester
### An Emerging-Market FX Trading Strategy Research Project

**Author:** [Your Name]
**Built in:** Google Colab
**Pair:** USD/UZS (US Dollar / Uzbek Som)
**Data window:** ~5 years of daily official rates

---

### What this notebook does
1. Pulls historical USD/UZS rates from the Central Bank of Uzbekistan (with a
   synthetic-data fallback so the notebook NEVER breaks, even with no internet).
2. Cleans the data and engineers features (moving averages, daily returns, volatility).
3. Builds a moving-average crossover trading strategy.
4. Backtests it against a buy-and-hold baseline.
5. Computes professional risk/return metrics (Sharpe ratio, max drawdown, win rate).
6. Produces clean, presentation-ready charts.

### How to run this in Google Colab
1. Open https://colab.research.google.com/ and create a new notebook.
2. Copy each `# %%` block below into its own Colab cell (Colab auto-splits on
   `# %%` if you paste the whole file into one cell and then "Split cell",
   or just copy block-by-block — either works).
3. Run cells top to bottom (Runtime > Run all).
4. No API keys are required. Everything here uses free, public data sources.

If you've never used Colab: a "cell" is just a chunk of code you can run by
itself with Shift+Enter. Markdown cells (like this one) are just text/notes —
they don't run code.
"""

# %% [markdown]
"""
## 1. Setup — Installing and Importing Libraries

We need:
- `requests` → to call the Central Bank of Uzbekistan's API over HTTP
- `pandas` → to store and manipulate our time series data (like Excel, but in code)
- `numpy` → for fast numerical math (returns, log calculations, etc.)
- `matplotlib` → for static, publication-quality charts
- `plotly` → for interactive charts (zoomable, hoverable — great for a portfolio demo)

Colab already has all of these pre-installed, but we run `pip install` anyway
with `-q` (quiet mode) so the notebook is self-contained and reproducible on
any machine, not just Colab.
"""

# %%
# Install dependencies (Colab usually has these, but this makes the notebook portable)
!pip install -q requests pandas numpy matplotlib plotly scipy

# %%
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")  # suppress noisy pandas/numpy warnings for clean output

# Make matplotlib charts look more professional out of the box
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.size"] = 11

print("Libraries loaded successfully.")

# %% [markdown]
"""
## 2. Configuration

Centralizing all the "knobs" of the project in one place is a habit every
quant desk follows — it means you can re-run the entire backtest with
different settings (date range, moving average windows, starting capital)
without hunting through the code.

**Why these defaults?**
- `START_DATE` / `END_DATE`: ~5 years gives us enough history to see multiple
  market regimes (UZS had a major devaluation in 2017, gradual depreciation
  since, and periods of relative central-bank-managed stability).
- `SHORT_WINDOW = 20`, `LONG_WINDOW = 50`: a classic "20/50" crossover, common
  in retail and prop-desk momentum strategies. Short enough to react to trend
  changes, long enough to filter out single-day noise.
- `INITIAL_CAPITAL`: arbitrary starting USD amount for the simulated portfolio.
- `TRANSACTION_COST_BPS`: basis points (1 bps = 0.01%) charged per trade, to
  make the backtest realistic. Real FX spreads/commissions are not zero.
"""

# %%
CONFIG = {
    "BASE_CURRENCY": "USD",
    "QUOTE_CURRENCY": "UZS",
    "START_DATE": "2021-01-01",
    "END_DATE": "2026-01-01",          # adjust to "today" if you want the latest data
    "SHORT_WINDOW": 20,                 # fast moving average (in trading days)
    "LONG_WINDOW": 50,                  # slow moving average (in trading days)
    "INITIAL_CAPITAL": 10_000.0,        # starting capital in USD
    "TRANSACTION_COST_BPS": 5,          # 5 bps = 0.05% cost per trade (realistic FX spread)
    "RISK_FREE_RATE_ANNUAL": 0.05,      # ~5% annualized, used in Sharpe ratio calc
    "TRADING_DAYS_PER_YEAR": 252,
}

print("Configuration set:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
"""
## 3. Data Collection Pipeline

### Why this is the hardest part of the project (and why that's OK)

Unlike EUR/USD or GBP/USD, USD/UZS is **not** available on most free
financial APIs (Yahoo Finance, Alpha Vantage's free tier, etc. — UZS is too
illiquid / minor a currency for them to carry full daily history). The
correct, professional move — and the one a real EM trading desk would make —
is to go to the **primary source**: the Central Bank of Uzbekistan (CBU),
which publishes the *official* daily USD/UZS rate it uses for accounting,
customs, and statistical purposes.

### The CBU API

The Central Bank of Uzbekistan exposes a free, public, no-API-key-required
JSON endpoint:

```
https://cbu.uz/en/arkhiv-kursov-valyut/json/{CURRENCY}/{YYYY-MM-DD}/
```

This returns the official rate for **one single date**. There is no "give me
a date range" endpoint — so to build a 5-year daily series, we have to loop
over every date and make one HTTP request per day. This is a real-world
lesson in working with imperfect, non-batch-friendly EM data sources — you
won't get a clean `yfinance.download()` one-liner here, and that's exactly
the kind of plumbing problem you'll hit on a real EM trading desk.

### Our approach
1. Loop through each calendar day in the date range.
2. Call the CBU endpoint for that date.
3. Parse out the rate.
4. Be polite to the server: small delay between requests, retry on failure.
5. If the API is unreachable (rate-limited, blocked, network issue, CBU
   downtime) — which **will** happen sometimes in Colab — **fall back to a
   realistic synthetic dataset** so the rest of the notebook still runs
   end-to-end. This is a production-engineering best practice: external data
   dependencies should never crash your whole pipeline.

### Alternative data sources (in case CBU.uz is blocked/unreachable)
- **exchangerate.host** or **Frankfurter.app** — free, no-key APIs that mirror
  central bank rates for many currencies including UZS (USD/UZS via ECB
  reference cross-rates).
- **investing.com / xe.com** historical download (manual CSV export, then
  upload to Colab with `files.upload()`).
- **World Bank / IMF International Financial Statistics** — monthly (not
  daily) official rates, good for a lower-frequency version of this project.
We implement the CBU method as primary and exchangerate.host as a secondary
fallback before resorting to synthetic data.
"""

# %%
def fetch_cbu_rate_single_date(date_obj, currency="USD", max_retries=2, timeout=6):
    """
    Fetch the official CBU exchange rate for ONE specific date.

    Parameters
    ----------
    date_obj : datetime.date
        The date we want the rate for.
    currency : str
        3-letter currency code (e.g. "USD").
    max_retries : int
        How many times to retry on a failed request before giving up.
    timeout : int
        Seconds to wait for a response before giving up on this request.

    Returns
    -------
    float or None
        The exchange rate (UZS per 1 unit of `currency`), or None if the
        request failed after all retries.
    """
    date_str = date_obj.strftime("%Y-%m-%d")
    url = f"https://cbu.uz/en/arkhiv-kursov-valyut/json/{currency}/{date_str}/"

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()  # raises an error if status code is 4xx/5xx
            data = response.json()

            # CBU returns a LIST with one dict inside, e.g. [{"Rate": "12705.00", ...}]
            if isinstance(data, list) and len(data) > 0:
                rate = float(data[0]["Rate"])
                return rate
            else:
                return None  # no data for this date (e.g. weekend/holiday)

        except (requests.exceptions.RequestException, KeyError, ValueError, IndexError):
            if attempt < max_retries:
                time.sleep(0.5)  # brief pause before retrying
                continue
            return None  # give up after max_retries

    return None


def fetch_cbu_history(start_date, end_date, currency="USD", request_delay=0.05, verbose=True):
    """
    Loop over every calendar day between start_date and end_date and pull the
    official CBU rate for each day. This is necessarily slow (one HTTP call
    per day) because CBU's public API has no date-RANGE endpoint.

    Parameters
    ----------
    start_date, end_date : str ("YYYY-MM-DD")
    currency : str
    request_delay : float
        Seconds to sleep between requests — being a "good citizen" to the
        server and avoiding rate-limit blocks.
    verbose : bool
        Print progress every ~100 days.

    Returns
    -------
    pandas.DataFrame with columns ["Date", "Rate"], or empty DataFrame on total failure.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end - start).days + 1

    records = []
    current = start
    day_count = 0
    consecutive_failures = 0

    print(f"Fetching {currency}/UZS rates from CBU.uz for {total_days} calendar days...")
    print("(This can take a while since the API only supports one date per request.)")

    while current <= end:
        rate = fetch_cbu_rate_single_date(current, currency=currency)

        if rate is not None:
            records.append({"Date": current, "Rate": rate})
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        # If we get 30 failures in a row, the API is probably down/blocked —
        # stop early rather than burning through the whole date range slowly.
        if consecutive_failures >= 30:
            print(f"  Stopping early: {consecutive_failures} consecutive failed requests "
                  f"— CBU API appears unreachable from this environment.")
            break

        day_count += 1
        if verbose and day_count % 200 == 0:
            print(f"  ...processed {day_count}/{total_days} days "
                  f"({len(records)} valid rates collected so far)")

        current += timedelta(days=1)
        time.sleep(request_delay)

    df = pd.DataFrame(records)
    print(f"Done. Collected {len(df)} valid daily rates out of {day_count} days attempted.")
    return df


def fetch_exchangerate_host_history(start_date, end_date, base="USD", quote="UZS"):
    """
    SECONDARY fallback data source: exchangerate.host's free timeseries
    endpoint, which mirrors central-bank reference rates for many currencies.
    No API key required for basic usage.

    Returns
    -------
    pandas.DataFrame with columns ["Date", "Rate"], or empty DataFrame on failure.
    """
    url = "https://api.exchangerate.host/timeseries"
    params = {"start_date": start_date, "end_date": end_date, "base": base, "symbols": quote}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data.get("success", True) or "rates" not in data:
            return pd.DataFrame()

        records = [
            {"Date": datetime.strptime(date_str, "%Y-%m-%d").date(), "Rate": rates.get(quote)}
            for date_str, rates in data["rates"].items()
            if rates.get(quote) is not None
        ]
        return pd.DataFrame(records)

    except (requests.exceptions.RequestException, ValueError, KeyError):
        return pd.DataFrame()


def generate_synthetic_usduzs(start_date, end_date, seed=42):
    """
    FINAL fallback: generate a realistic SYNTHETIC USD/UZS series so the rest
    of the notebook (cleaning, strategy, backtest, charts) can still run and
    be demonstrated even with zero internet access.

    This is NOT real data — it's a geometric random walk with a gentle upward
    drift, calibrated to loosely resemble UZS's real-world depreciation trend
    against the USD (the Som has weakened fairly steadily since the 2017
    currency liberalization). We clearly label this as synthetic everywhere
    it's used, both in print statements and in chart titles.

    Returns
    -------
    pandas.DataFrame with columns ["Date", "Rate"]
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    # Calibration: UZS depreciates ~6-9%/year on average against USD historically,
    # with daily volatility much lower than typical EM equities but bursty around
    # known devaluation events. We simulate a mild upward drift + occasional jumps.
    daily_drift = 0.0003       # ~7.5%/year average depreciation of UZS vs USD
    daily_vol = 0.0015         # daily volatility
    jump_prob = 0.01           # 1% chance per day of a larger "step" move
    jump_size_std = 0.01

    log_returns = rng.normal(loc=daily_drift, scale=daily_vol, size=n)
    jumps = (rng.random(n) < jump_prob) * rng.normal(0, jump_size_std, n)
    log_returns += jumps

    log_rate = np.cumsum(log_returns) + np.log(10500)  # start near a realistic 2021 level
    rate = np.exp(log_rate)

    return pd.DataFrame({"Date": dates.date, "Rate": rate})


def get_usduzs_data(config):
    """
    Master data-collection function: tries CBU first, falls back to
    exchangerate.host, and finally falls back to synthetic data. Always
    returns a usable DataFrame and tells you exactly which source was used —
    transparency about data provenance matters in finance.
    """
    print("=" * 70)
    print("STEP 1: Attempting PRIMARY source (Central Bank of Uzbekistan)...")
    print("=" * 70)
    df = fetch_cbu_history(config["START_DATE"], config["END_DATE"], currency=config["BASE_CURRENCY"])

    # Require a reasonably complete series before trusting the primary source
    expected_days = (datetime.strptime(config["END_DATE"], "%Y-%m-%d")
                      - datetime.strptime(config["START_DATE"], "%Y-%m-%d")).days
    if len(df) >= expected_days * 0.5:  # at least half the days came through
        df["Source"] = "CBU.uz (official)"
        return df

    print("\nPrimary source returned insufficient data.")
    print("=" * 70)
    print("STEP 2: Attempting SECONDARY source (exchangerate.host)...")
    print("=" * 70)
    df2 = fetch_exchangerate_host_history(config["START_DATE"], config["END_DATE"],
                                           base=config["BASE_CURRENCY"], quote=config["QUOTE_CURRENCY"])
    if len(df2) >= expected_days * 0.5:
        df2["Source"] = "exchangerate.host"
        return df2

    print("\nSecondary source also insufficient.")
    print("=" * 70)
    print("STEP 3: Falling back to SYNTHETIC data for demonstration purposes.")
    print("NOTE: Results below are NOT real market data. Re-run this cell with")
    print("internet access to CBU.uz for genuine USD/UZS history.")
    print("=" * 70)
    df3 = generate_synthetic_usduzs(config["START_DATE"], config["END_DATE"])
    df3["Source"] = "SYNTHETIC (demo only)"
    return df3

# %%
# Run the data collection pipeline.
# NOTE: If using real CBU data over a 5-year window, this cell can take
# 20-40+ minutes because the API requires one request PER DAY. For a faster
# test run, temporarily shrink CONFIG["START_DATE"] to something like
# "2024-01-01" before running this cell.
raw_data = get_usduzs_data(CONFIG)

print(f"\nData source used: {raw_data['Source'].iloc[0]}")
print(f"Rows collected: {len(raw_data)}")
raw_data.head()

# %% [markdown]
"""
## 4. Data Cleaning & Feature Engineering

Raw exchange rate data is never analysis-ready. Before we can build a
strategy, we need to:

1. **Sort and de-duplicate** by date (CBU doesn't publish on weekends/holidays,
   so we'll have gaps — that's expected for FX, since the Som isn't traded
   24/7 like a major pair).
2. **Forward-fill gaps** — on non-publishing days, the most recent official
   rate is the best available "current" value (this is standard practice for
   FX rates and bond prices on non-trading days).
3. **Compute daily returns** — percentage change day-over-day. This is the
   fundamental building block of every metric we'll calculate later (Sharpe
   ratio, volatility, etc.).
4. **Compute moving averages** — the short and long windows our strategy will
   compare against each other.
5. **Compute rolling volatility** — how "jumpy" the exchange rate has been
   recently, which is useful context for an EM currency.

### Why moving averages?
A simple moving average (SMA) smooths out day-to-day noise so we can see the
underlying *trend*. If the **fast** (short-window) average is above the
**slow** (long-window) average, that means recent prices are higher than
older prices on average → an uptrend → in our strategy, "USD is strengthening
against UZS, so hold/buy USD." When the fast average dips below the slow
average, the trend has turned down → "sell USD, hold UZS instead."
"""

# %%
def clean_and_engineer_features(df, config):
    """
    Takes the raw [Date, Rate] DataFrame and returns a fully-featured,
    analysis-ready DataFrame indexed by date.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    df = df.set_index("Date")

    # Reindex to EVERY calendar day in range, then forward-fill non-publishing days
    # (weekends/holidays carry forward the last official rate, standard FX convention)
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)
    df["Rate"] = df["Rate"].ffill()
    df.index.name = "Date"

    # Drop any leading NaNs (in case the series starts with a gap before the first valid rate)
    df = df.dropna(subset=["Rate"])

    # --- Feature engineering ---
    # Daily simple return: (today - yesterday) / yesterday
    df["Daily_Return"] = df["Rate"].pct_change()

    # Log returns are used for risk metrics because they're additive over time
    # (a common quant convention — Sharpe ratios are typically computed on log returns)
    df["Log_Return"] = np.log(df["Rate"] / df["Rate"].shift(1))

    # Moving averages — the core inputs to our trading signal
    df["SMA_Short"] = df["Rate"].rolling(window=config["SHORT_WINDOW"]).mean()
    df["SMA_Long"] = df["Rate"].rolling(window=config["LONG_WINDOW"]).mean()

    # Rolling 20-day annualized volatility — gives context on how turbulent
    # the currency has been recently (useful for EM risk commentary)
    df["Rolling_Volatility_Annualized"] = (
        df["Daily_Return"].rolling(window=20).std() * np.sqrt(config["TRADING_DAYS_PER_YEAR"])
    )

    # Drop rows where the long moving average isn't yet defined (the first
    # LONG_WINDOW days have no valid SMA_Long — can't trade until we have it)
    df = df.dropna(subset=["SMA_Long"])

    return df

# %%
clean_data = clean_and_engineer_features(raw_data, CONFIG)

print(f"Clean dataset shape: {clean_data.shape}")
print(f"Date range: {clean_data.index.min().date()} to {clean_data.index.max().date()}")
print(f"\nSample of engineered features:")
clean_data[["Rate", "SMA_Short", "SMA_Long", "Daily_Return", "Rolling_Volatility_Annualized"]].tail()

# %% [markdown]
"""
## 5. Strategy Logic — Moving Average Crossover

### The rule, in plain English
- **Signal = +1 (long USD)** when the short-term moving average is ABOVE the
  long-term moving average — the trend favors USD strengthening, so we hold USD.
- **Signal = -1 (short USD / hold UZS)** when the short-term average is BELOW
  the long-term average — the trend favors USD weakening relative to recent
  history, so we exit USD (in our simplified backtest, this means holding
  cash/UZS-equivalent instead of USD).
- **A trade is triggered** only when the signal CHANGES from one day to the
  next (this is the "crossover" moment) — not on every day the signal is
  active. This avoids over-counting "trades" for days we're just sitting in
  the same position.

### A note on what "long/short USD" means here
Because USD/UZS isn't a typical two-way tradable retail product (you can't
easily short the Som the way you'd short a stock), we treat this as a
**directional allocation strategy**: when signal = +1, our hypothetical
portfolio is 100% in USD; when signal = -1, it's 100% in UZS-equivalent cash
(effectively "out of the trade," earning ~0% in this simplified model). This
is a common simplification for retail-style FX trend strategies and keeps
the backtest logic clean and explainable.
"""

# %%
def generate_signals(df, config):
    """
    Adds a 'Signal' column (+1 = long USD, -1 = flat/out of USD) and a
    'Position_Change' column marking the exact days a crossover happens
    (i.e. an actual trade is executed).
    """
    df = df.copy()

    # Core crossover logic: are we above or below the trend line?
    df["Signal"] = np.where(df["SMA_Short"] > df["SMA_Long"], 1, -1)

    # A "trade" only happens the day the signal flips from the previous day
    df["Position_Change"] = df["Signal"].diff().fillna(0) != 0

    return df

# %%
signal_data = generate_signals(clean_data, CONFIG)

num_trades = signal_data["Position_Change"].sum()
print(f"Total crossover events (trades) over the period: {int(num_trades)}")
signal_data[signal_data["Position_Change"]][["Rate", "SMA_Short", "SMA_Long", "Signal"]].head(10)

# %% [markdown]
"""
## 6. Backtesting Engine

This is the heart of the project: simulating what would have happened to a
real portfolio if we'd followed this strategy mechanically, day by day, with
no hindsight bias.

### Core backtesting principles we follow here (important for credibility!)
1. **No look-ahead bias**: today's trading decision uses only data available
   *up to and including* today's close. We never let the strategy "peek" at
   tomorrow's price.
2. **Signal lag**: in real trading you can't react to a signal and execute in
   the same instant it appears — there's always some execution delay. We
   shift our position by one day (`.shift(1)`) so that a signal generated at
   the close of day T is acted on starting day T+1. This is standard
   backtesting practice and avoids an unrealistically "perfect" strategy.
3. **Transaction costs**: every time we flip positions, we pay a small cost
   (modeled as basis points of the trade value). Ignoring this is one of the
   most common mistakes in amateur backtests — it makes mediocre strategies
   look artificially profitable.
4. **Compounding**: portfolio value compounds daily based on the return earned
   *while in a given position*, not a simple sum of profits.

### Buy-and-hold baseline
To know whether our strategy actually adds value, we need a benchmark: what
if we'd simply bought USD on day one and held it for the entire period,
doing nothing else? If our strategy can't beat this simple baseline (after
costs), it's not worth the added complexity and trading costs.
"""

# %%
def run_backtest(df, config):
    """
    Simulates the moving-average crossover strategy and a buy-and-hold
    baseline over the same period, tracking portfolio value day by day.
    """
    df = df.copy()

    # --- Strategy returns ---
    # Shift signal by 1 day: act on YESTERDAY's signal today (no look-ahead bias)
    df["Strategy_Position"] = df["Signal"].shift(1).fillna(0)

    # Strategy daily return = position * underlying daily return
    # If position = -1 ("flat"/out of USD), we treat the return as 0 (sitting
    # in cash) rather than -1 * return, since we're not actually short the Som.
    df["Strategy_Daily_Return"] = np.where(
        df["Strategy_Position"] == 1, df["Daily_Return"], 0.0
    )

    # Apply transaction costs on days a trade actually executes
    cost_per_trade = config["TRANSACTION_COST_BPS"] / 10_000
    df["Trade_Executed"] = df["Signal"].shift(1).diff().fillna(0) != 0
    df["Strategy_Daily_Return"] = np.where(
        df["Trade_Executed"], df["Strategy_Daily_Return"] - cost_per_trade, df["Strategy_Daily_Return"]
    )

    # --- Build equity curves (portfolio value over time) ---
    initial_capital = config["INITIAL_CAPITAL"]
    df["Strategy_Equity"] = initial_capital * (1 + df["Strategy_Daily_Return"]).cumprod()
    df["BuyHold_Equity"] = initial_capital * (1 + df["Daily_Return"].fillna(0)).cumprod()

    return df

# %%
backtest_results = run_backtest(signal_data, CONFIG)

print("Backtest complete. Final equity values:")
print(f"  Strategy final value:   ${backtest_results['Strategy_Equity'].iloc[-1]:,.2f}")
print(f"  Buy & Hold final value: ${backtest_results['BuyHold_Equity'].iloc[-1]:,.2f}")
print(f"  (Starting capital was: ${CONFIG['INITIAL_CAPITAL']:,.2f})")

# %% [markdown]
"""
## 7. Performance Metrics

A backtest without proper risk-adjusted metrics is just a chart — it doesn't
tell you whether the strategy was actually *good*, just whether it happened
to make money over one specific historical period. Here's what each metric
tells us and why it matters:

| Metric | What it measures | Why it matters |
|---|---|---|
| **Total Return %** | Overall % gain/loss over the full period | The headline number, but easy to be misled by alone |
| **Annualized Return** | Return normalized to a 1-year basis | Lets you compare strategies tested over different time spans |
| **Volatility (annualized)** | How much the returns swing around | High volatility = higher risk for the same return |
| **Sharpe Ratio** | Return earned per unit of risk taken, vs. a risk-free rate | The single most-cited risk-adjusted performance metric in finance |
| **Max Drawdown** | The worst peak-to-trough decline in portfolio value | Answers "what's the worst pain I'd have had to sit through?" |
| **Win Rate** | % of trades that were profitable | Doesn't tell you about size of wins/losses, but useful color |
| **Number of Trades** | How often the strategy actually traded | More trades = more transaction costs = needs bigger edge to justify |

### Sharpe Ratio formula
$$ Sharpe = \\frac{\\overline{R_p} - R_f}{\\sigma_p} \\times \\sqrt{252} $$
where $\\overline{R_p}$ is the mean daily strategy return, $R_f$ is the daily
risk-free rate, and $\\sigma_p$ is the standard deviation of daily strategy
returns. We annualize by multiplying by $\\sqrt{252}$ (the number of trading
days in a year), which is the standard convention for daily-frequency data.

### Max Drawdown formula
At every point in time, compare the current portfolio value to its running
historical peak. Drawdown = (current value / peak value) − 1. Max drawdown is
the most negative value this ever reaches across the whole backtest.
"""

# %%
def calculate_performance_metrics(df, config, equity_col, return_col, label):
    """
    Computes the full suite of performance metrics for a given equity curve.

    Parameters
    ----------
    df : DataFrame with the backtest results
    equity_col : str — column name of the equity curve to analyze
    return_col : str — column name of the daily returns feeding that equity curve
    label : str — human-readable name for this strategy (for printing)

    Returns
    -------
    dict of computed metrics
    """
    equity = df[equity_col]
    daily_returns = df[return_col].fillna(0)

    # --- Total & annualized return ---
    total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    num_years = (df.index[-1] - df.index[0]).days / 365.25
    annualized_return_pct = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / num_years) - 1) * 100

    # --- Volatility (annualized) ---
    annualized_vol_pct = daily_returns.std() * np.sqrt(config["TRADING_DAYS_PER_YEAR"]) * 100

    # --- Sharpe Ratio ---
    daily_rf = config["RISK_FREE_RATE_ANNUAL"] / config["TRADING_DAYS_PER_YEAR"]
    excess_returns = daily_returns - daily_rf
    sharpe_ratio = (
        (excess_returns.mean() / excess_returns.std()) * np.sqrt(config["TRADING_DAYS_PER_YEAR"])
        if excess_returns.std() > 0 else np.nan
    )

    # --- Max Drawdown ---
    running_max = equity.cummax()
    drawdown_series = (equity / running_max) - 1
    max_drawdown_pct = drawdown_series.min() * 100

    # --- Win rate & trade count ---
    # "Win rate" here means: of the days this approach was actually exposed to
    # the market (in a position), what % had a positive return? For the MA
    # strategy that's days with Strategy_Position == 1; for buy & hold it's
    # every day in the sample (always in the trade), so we don't filter.
    if return_col == "Strategy_Daily_Return" and "Strategy_Position" in df.columns:
        num_trades = int(df["Trade_Executed"].sum())
        active_days = daily_returns[df["Strategy_Position"] == 1]
    else:
        num_trades = "N/A"  # buy & hold isn't "trading" in the same sense — one trade on day 1
        active_days = daily_returns

    win_rate_pct = (active_days > 0).mean() * 100 if len(active_days) > 0 else np.nan

    metrics = {
        "Strategy": label,
        "Total Return (%)": round(total_return_pct, 2),
        "Annualized Return (%)": round(annualized_return_pct, 2),
        "Annualized Volatility (%)": round(annualized_vol_pct, 2),
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Max Drawdown (%)": round(max_drawdown_pct, 2),
        "Win Rate (%)": round(win_rate_pct, 2) if not np.isnan(win_rate_pct) else "N/A",
        "Number of Trades": num_trades,
    }
    return metrics, drawdown_series

# %%
strategy_metrics, strategy_drawdown = calculate_performance_metrics(
    backtest_results, CONFIG, "Strategy_Equity", "Strategy_Daily_Return", "MA Crossover Strategy"
)
buyhold_metrics, buyhold_drawdown = calculate_performance_metrics(
    backtest_results, CONFIG, "BuyHold_Equity", "Daily_Return", "Buy & Hold (USD)"
)

metrics_comparison = pd.DataFrame([strategy_metrics, buyhold_metrics]).set_index("Strategy")
print("=" * 70)
print("PERFORMANCE COMPARISON: Strategy vs. Buy & Hold")
print("=" * 70)
metrics_comparison

# %% [markdown]
"""
## 8. Visualizations

A backtest is only as good as your ability to communicate it. We'll build
three charts that any finance interviewer or hiring manager will immediately
understand:

1. **Price chart with signals** — the exchange rate over time, with the two
   moving averages and little markers showing exactly where the strategy
   bought/sold.
2. **Equity curve** — strategy portfolio value vs. buy-and-hold portfolio
   value, side by side, over time. This is the single most important chart
   in the whole project — it answers "did this actually work?" at a glance.
3. **Drawdown chart** — visualizing the pain (peak-to-trough declines) an
   investor would have experienced holding each approach.

We build static `matplotlib` versions (good for a PDF report / README image)
AND an interactive `plotly` version (great for a live portfolio demo where
you can zoom and hover over individual data points).
"""

# %%
# --- CHART 1: Price chart with moving averages and buy/sell signals (matplotlib) ---
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(backtest_results.index, backtest_results["Rate"],
        label="USD/UZS Rate", color="#1f2937", linewidth=1.2, alpha=0.85)
ax.plot(backtest_results.index, backtest_results["SMA_Short"],
        label=f"SMA {CONFIG['SHORT_WINDOW']}-day", color="#2563eb", linewidth=1.3)
ax.plot(backtest_results.index, backtest_results["SMA_Long"],
        label=f"SMA {CONFIG['LONG_WINDOW']}-day", color="#dc2626", linewidth=1.3)

# Mark the exact days a crossover (trade) happened
buy_signals = backtest_results[(backtest_results["Position_Change"]) & (backtest_results["Signal"] == 1)]
sell_signals = backtest_results[(backtest_results["Position_Change"]) & (backtest_results["Signal"] == -1)]

ax.scatter(buy_signals.index, buy_signals["Rate"], marker="^", color="green",
           s=110, zorder=5, label="Buy USD Signal", edgecolors="black", linewidths=0.5)
ax.scatter(sell_signals.index, sell_signals["Rate"], marker="v", color="red",
           s=110, zorder=5, label="Sell USD Signal", edgecolors="black", linewidths=0.5)

ax.set_title("USD/UZS Exchange Rate — Moving Average Crossover Signals",
              fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("UZS per 1 USD", fontsize=12)
ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
plt.show()

# %%
# --- CHART 2: Equity curve comparison (matplotlib) ---
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(backtest_results.index, backtest_results["Strategy_Equity"],
        label="MA Crossover Strategy", color="#2563eb", linewidth=1.8)
ax.plot(backtest_results.index, backtest_results["BuyHold_Equity"],
        label="Buy & Hold (USD)", color="#6b7280", linewidth=1.8, linestyle="--")

ax.set_title("Portfolio Equity Curve — Strategy vs. Buy & Hold",
              fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Portfolio Value (USD)", fontsize=12)
ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
fig.tight_layout()
plt.show()

# %%
# --- CHART 3: Drawdown comparison (matplotlib) ---
fig, ax = plt.subplots(figsize=(15, 5))

ax.fill_between(backtest_results.index, strategy_drawdown * 100, 0,
                 color="#2563eb", alpha=0.4, label="MA Crossover Strategy")
ax.fill_between(backtest_results.index, buyhold_drawdown * 100, 0,
                 color="#6b7280", alpha=0.4, label="Buy & Hold (USD)")

ax.set_title("Drawdown Over Time — How Much Pain Would You Have Felt?",
              fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Drawdown (%)", fontsize=12)
ax.legend(loc="lower left", framealpha=0.9, fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
plt.show()

# %% [markdown]
"""
### Interactive Dashboard (Plotly)

This combines all three charts into a single scrollable, zoomable dashboard
— the kind of thing you can screen-record or embed directly in a portfolio
website. Hover over any point to see exact values.
"""

# %%
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.45, 0.35, 0.20],
    subplot_titles=(
        "USD/UZS Rate with Moving Average Crossover Signals",
        "Portfolio Equity Curve: Strategy vs. Buy & Hold",
        "Drawdown (%)"
    )
)

# Row 1: Price + SMAs + signals
fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["Rate"],
                          name="USD/UZS Rate", line=dict(color="#1f2937", width=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["SMA_Short"],
                          name=f"SMA {CONFIG['SHORT_WINDOW']}", line=dict(color="#2563eb", width=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["SMA_Long"],
                          name=f"SMA {CONFIG['LONG_WINDOW']}", line=dict(color="#dc2626", width=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Rate"], mode="markers", name="Buy Signal",
                          marker=dict(symbol="triangle-up", size=11, color="green")), row=1, col=1)
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["Rate"], mode="markers", name="Sell Signal",
                          marker=dict(symbol="triangle-down", size=11, color="red")), row=1, col=1)

# Row 2: Equity curves
fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["Strategy_Equity"],
                          name="Strategy Equity", line=dict(color="#2563eb", width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["BuyHold_Equity"],
                          name="Buy & Hold Equity", line=dict(color="#6b7280", width=2, dash="dash")), row=2, col=1)

# Row 3: Drawdowns
fig.add_trace(go.Scatter(x=backtest_results.index, y=strategy_drawdown * 100,
                          name="Strategy Drawdown", fill="tozeroy",
                          line=dict(color="#2563eb")), row=3, col=1)
fig.add_trace(go.Scatter(x=backtest_results.index, y=buyhold_drawdown * 100,
                          name="Buy & Hold Drawdown", fill="tozeroy",
                          line=dict(color="#6b7280")), row=3, col=1)

fig.update_layout(
    height=950, width=1100,
    title_text=f"USD/UZS Moving Average Crossover Backtest Dashboard "
               f"(Data source: {backtest_results.attrs.get('source', raw_data['Source'].iloc[0])})",
    title_font_size=16,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_yaxes(title_text="UZS per USD", row=1, col=1)
fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

fig.show()

# %% [markdown]
"""
## 9. Final Summary Table

A clean, single printed summary — the kind of output you'd screenshot
directly into a project writeup or include as a results table in your README.
"""

# %%
print("=" * 70)
print(f"  USD/UZS MOVING AVERAGE CROSSOVER BACKTEST — FINAL RESULTS")
print(f"  Period: {backtest_results.index[0].date()} to {backtest_results.index[-1].date()}")
print(f"  Data source: {raw_data['Source'].iloc[0]}")
print(f"  Strategy params: SMA({CONFIG['SHORT_WINDOW']}) / SMA({CONFIG['LONG_WINDOW']}) crossover, "
      f"{CONFIG['TRANSACTION_COST_BPS']} bps cost/trade")
print("=" * 70)
print(metrics_comparison.to_string())
print("=" * 70)

outperformance = (strategy_metrics["Total Return (%)"] - buyhold_metrics["Total Return (%)"])
verdict = "OUTPERFORMED" if outperformance > 0 else "UNDERPERFORMED"
print(f"\nThe MA Crossover strategy {verdict} Buy & Hold by {abs(outperformance):.2f} percentage points "
      f"over this period.")
print("\nNOTE: Past performance on historical data does not guarantee future results.")
print("This is an educational/research backtest, not investment advice.")

# %% [markdown]
"""
## 10. (Optional) Export Results

Saves the full backtest dataset and the metrics table as CSVs you can attach
to a GitHub repo, a resume project writeup, or a slide deck.
"""

# %%
backtest_results.to_csv("usduzs_backtest_results.csv")
metrics_comparison.to_csv("usduzs_performance_metrics.csv")
print("Saved: usduzs_backtest_results.csv, usduzs_performance_metrics.csv")

# In Colab, download them with:
# from google.colab import files
# files.download("usduzs_backtest_results.csv")
# files.download("usduzs_performance_metrics.csv")
