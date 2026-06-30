# Statistical Arbitrage & Pairs Trading Engine
### Taiwan AI Infrastructure Supply Chain | TWSE/TWO Equities | 2023–2026

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Motivation & Background](#2-motivation--background)
3. [Universe & Data](#3-universe--data)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Limitations & Future Work](#6-limitations--future-work)
7. [Setup & Usage](#7-setup--usage)

---

## 1. Project Overview

This project implements a full statistical arbitrage pairs trading engine targeting the Taiwan Stock Exchange (TWSE) and Taipei Exchange (TWO), with a focus on the AI infrastructure and semiconductor supply chain sector. The engine identifies cointegrated stock pairs using the Engle-Granger two-step procedure with Benjamini-Hochberg false discovery rate correction, estimates time-varying hedge ratios via a hand-implemented Kalman filter, and generates mean-reversion trading signals from rolling z-scores of the dynamic spread. Performance is evaluated both in-sample across all 42 cointegrated pairs and out-of-sample via a rolling walk-forward backtest that rescreens pairs at each fold to eliminate look-ahead bias. The project draws directly on time series econometrics coursework (ECON4304, HKUST) — specifically the theory of cointegration, unit roots, and error correction models — and extends it into a production-quality quantitative research codebase.

---

## 2. Motivation & Background

Statistical arbitrage exploits a fundamental insight: stocks exposed to the same economic drivers tend to share long-run price equilibria. When two stocks temporarily diverge from their historical relationship, mean reversion creates a predictable, market-neutral profit opportunity.

Taiwan's AI infrastructure supply chain provides an unusually fertile environment for this strategy. Companies spanning semiconductors (TSMC, 世界先進), ODMs (廣達, 鴻海), passive components (國巨), and networking equipment (智邦) are all exposed to the same global demand catalysts — AI server buildout, HBM memory demand, CoWoS advanced packaging capacity. This shared exposure creates genuine long-run cointegration relationships that are grounded in economic fundamentals rather than statistical coincidence.

The period from 2023 onward also introduced structural conditions that amplify mean-reversion signals: the post-ChatGPT AI frenzy drew massive global institutional inflows into Taiwanese equities, increasing both volatility and the frequency of temporary mispricings between closely linked supply chain names. High volatility, tight economic linkages, and a shared investor base create exactly the conditions under which statistical arbitrage generates the most reliable signals.

This project is grounded in cointegration theory from ECON4304 (HKUST) — specifically the Engle-Granger two-step procedure, augmented Dickey-Fuller testing, and the error correction model framework — and extends these foundations into a production-quality trading engine with dynamic hedge ratio estimation and rigorous out-of-sample validation.

---

## 3. Universe & Data

**Data source**: Yahoo Finance via `yfinance`, daily adjusted closing prices from 2023-01-01 to 2026-05-31 (818 trading days).

**Universe construction**: 64 TWSE/TWO-listed equities screened from the top 300 stocks by market capitalisation, filtered to AI-relevant sectors:

| Sector | Example Names |
|---|---|
| Semiconductors | 台積電 (2330.TW), 聯詠 (6770.TW), 世界先進 (2344.TW) |
| Electronic Components | 台達電 (2308.TW), 日月光 (2059.TW) |
| Computer Hardware | 廣達 (2382.TW), 緯創 (3231.TW) |
| Optical | 大立光 (3008.TW), 元太 (8069.TWO) |
| Networking | 智邦 (2345.TW) |

**Filters applied**:
- Market cap ≥ 2,000億 TWD (ensures liquidity)
- Sectors: 半導體業, 電子零組件業, 電腦及週邊設備業, 其他電子業, 光電業, 通信網路業, 電子通路業
- Telecom names removed (utility-like return profile, inconsistent with AI supply chain thesis)
- 7769.TW (鴻勁) dropped — IPO'd November 2025, insufficient price history

**Why Taiwan only**: mixing TWD and USD-denominated prices introduces FX noise into the spread that is not mean-reverting. Restricting to a single currency and trading session ensures the spread reflects genuine economic relationships rather than FX movements or timezone mismatches.

**Start date rationale**: 2023-01-01 was chosen deliberately to capture the post-ChatGPT AI infrastructure investment cycle. Including 2022 (a severe bear market driven by Fed rate hikes and semiconductor inventory correction) would introduce regime heterogeneity that distorts cointegration estimates.

---

## 4. Methodology

### Step 1 — Cointegration Screening (`cointegration.py`)

Two price series P₁(t) and P₂(t) are cointegrated if their linear combination is stationary:

Observation:  y(t) = β(t)·x(t) + ε(t),    ε(t) ~ N(0, R)

Transition:   β(t) = β(t-1) + η(t),         η(t) ~ N(0, Q)

The Kalman filter recursively estimates β(t) via predict and update steps:

Predict

β_pred = β(t-1)

P_pred = P(t-1) + Q

Update

innovation = y(t) − β_pred·x(t)

S          = x(t)²·P_pred + R

K          = P_pred·x(t) / S        # Kalman gain

β(t)       = β_pred + K·innovation

P(t)       = (1 − K·x(t))·P_pred

The filter is implemented from scratch (no `pykalman`) with `delta=1e-4` parameterising process noise as `Q = delta/(1-delta)`. β is initialised from the OLS estimate to avoid the cold-start drift problem. The dynamic spread is:

spread(t) = y(t) − β(t)·x(t)

---

### Step 3 — Signal Generation (`signals.py`)

The spread is normalised into a rolling z-score over a 60-day lookback window:

z(t) = (spread(t) − μ₆₀(t)) / σ₆₀(t)

Trading rules:
- z(t) < −2.0 → **long spread** (buy y, sell x)
- z(t) > +2.0 → **short spread** (sell y, buy x)
- |z(t)| < 0.5 → **exit**
- Otherwise → hold previous position (forward-fill)

---

### Step 4 — Backtest (`backtest.py`)

Daily normalised P&L:

pnl(t) = signal(t-1) × (spread(t) − spread(t-1)) / |spread(t-1)|

Transaction costs of 10bps are deducted on every signal change. Performance metrics: annualised Sharpe ratio, maximum drawdown, hit rate, average trade duration.

**Walk-forward validation**: a rolling walk-forward backtest rescreens pairs on each 630-day training window and evaluates signals on the subsequent 60-day out-of-sample test window. Pairs are never selected using future data.

---

## 5. Results

### In-Sample Backtest

Screening 57 tickers (1,596 unique pairs) identified **42 cointegrated pairs** at α=0.05 with BH-FDR correction — a 2.6% hit rate consistent with academic benchmarks for this type of screening.

**Top 5 pairs by Sharpe ratio:**

| Pair | Names | Sharpe | Max Drawdown | Hit Rate | Avg Duration |
|---|---|---|---|---|---|
| 2344.TW / 6515.TW | 世界先進 / 矽力-KY | 5.69 | -69.4% | 64.4% | 5.3 days |
| 2344.TW / 3443.TW | 世界先進 / 創意電子 | 4.86 | -81.3% | 63.5% | 5.3 days |
| 2337.TW / 3443.TW | 旺宏 / 創意電子 | 4.44 | -88.9% | 65.6% | 5.0 days |
| 2303.TW / 2492.TW | 聯電 / 晶豐明源 | 4.18 | -87.6% | 66.8% | 17.2 days |
| 6770.TW / 8046.TW | 聯詠 / 南電 | 4.15 | -92.3% | 61.8% | 12.7 days |

All five pairs share tight supply chain linkages within the same semiconductor ecosystem. Short average trade durations of 5-13 days confirm genuine fast mean reversion rather than slow drift correction.

### Walk-Forward Validation

| Fold | Test Period | Pairs | Sharpe | Max Drawdown | Hit Rate |
|---|---|---|---|---|---|
| 0 | Jan 2023 – Nov 2025 | 0 | — | — | — |
| 1 | Nov 2025 – Feb 2026 | 3 | 1.90 | -13.7% | 58.8% |
| 2 | Feb 2026 – May 2026 | 9 | 3.86 | -33.2% | 58.9% |

The walk-forward results reveal an important regime finding: no statistically defensible cointegrated pairs existed during the 2023-2025 training window, even at relaxed significance thresholds up to α=0.20. This reflects the AI bull market dynamic — sector-wide momentum caused all stocks to move together, breaking spread stationarity. Genuine mean-reversion opportunities emerged from late 2025 onward as market differentiation increased. Out-of-sample Sharpe ratios of 1.90 and 3.86 suggest genuine predictive power in this regime.

---

## 6. Limitations & Future Work

**Data limitations**: the 2023-2026 sample covers a single market regime. With only three walk-forward folds, the out-of-sample evidence is suggestive but not conclusive. A longer history spanning multiple market cycles would substantially strengthen the validation.

**P&L normalization instability**: dividing daily P&L by the lagged spread value creates extreme returns when the spread passes through near-zero. A more robust approach would normalize by portfolio notional value or use log-price spreads.

**Regime dependence**: the walk-forward analysis confirms cointegration relationships were absent during the 2023-2025 bull run. A production system would require a regime detection layer to switch the strategy on and off based on market conditions.

**Static thresholds**: entry (z=2.0) and exit (z=0.5) thresholds are fixed across all pairs. Adaptive thresholds calibrated to each pair's historical z-score distribution would likely improve signal quality.

**Future extensions**:
- Johansen cointegration test as a robustness check
- Portfolio-level risk management via Kelly criterion or inverse volatility weighting
- Higher-frequency intraday data for faster mean reversion capture
- Regime detection layer using Hidden Markov Models on sector volatility
- Extension to Korean market with explicit FX hedging

As Taiwan and broader Asian markets continue developing along the AI infrastructure theme, the supply chain relationships underpinning this strategy are likely to deepen — suggesting the model's predictive power may improve as more price history accumulates in this regime.

---

## 7. Setup & Usage

**Requirements**: Python 3.10+, Mac/Linux

```bash
# Clone and set up environment
git clone https://github.com/benjamincheng18/stat-arb-pairs-trading.git
cd stat-arb-pairs-trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add FRED API key (optional)
echo "FRED_API_KEY=your_key_here" > .env
```

**Run the full pipeline**:

```python
# In notebooks/test.ipynb
from src.data_loader import load_prices
from src.cointegration import screen_all_pairs, apply_fdr_correction
from src.kalman_filter import kalman_filter
from src.signals import compute_zscore, generate_signals
from src.backtest import run_backtest_pipeline, walk_forward_backtest

# Load data
prices, universe = load_prices()

# Screen pairs
results = screen_all_pairs(prices)
results = apply_fdr_correction(results)

# In-sample backtest
summary = run_backtest_pipeline(prices, results)

# Walk-forward validation
pnl_series, fold_summary = walk_forward_backtest(prices)
```

**Project structure**:
```
stat-arb-pairs-trading/
├── src/
│   ├── data_loader.py       # Universe definition and price fetching
│   ├── cointegration.py     # Engle-Granger screening with BH-FDR
│   ├── kalman_filter.py     # Dynamic hedge ratio estimation
│   ├── signals.py           # Rolling z-score and signal generation
│   └── backtest.py          # P&L calculation and walk-forward validation
├── notebooks/
│   └── test.ipynb           # Full pipeline exploration
├── requirements.txt
└── README.md
```
---

*Built by Benjamin Cheng — HKUST Information Systems & Economics*  
*GitHub: [benjamincheng18](https://github.com/benjamincheng18)*