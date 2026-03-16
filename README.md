# Alpha Engine — Systematic Equity Research Platform

A modular Python research platform for discovering and evaluating **alpha signals** on **London Stock Exchange (LSE)** listed equities. Alpha Engine implements the end-to-end workflow found in professional quantitative trading environments: factor generation, IC-based signal ranking, portfolio construction, rigorous backtesting, and an interactive research interface — all denominated in **GBP (£)**.

The platform is built around a **factor leaderboard** that continuously accumulates the best-performing signals, providing a structured environment for systematic alpha research.

---

## Contents

1. [Project Overview](#1-project-overview)
2. [Research Pipeline](#2-research-pipeline)
3. [Key Quant Concepts Implemented](#3-key-quant-concepts-implemented)
4. [System Architecture](#4-system-architecture)
5. [Factor Leaderboard](#5-factor-leaderboard)
6. [Supported Universes](#6-supported-universes)
7. [Technology Stack](#7-technology-stack)
8. [Example Research Workflow](#8-example-research-workflow)
9. [Running the Application](#9-running-the-application)
10. [Strategies Reference](#10-strategies-reference)
11. [Statistical Validation](#11-statistical-validation)
12. [Portfolio Construction Utilities](#12-portfolio-construction-utilities)
13. [UK Market Specifics & Transaction Costs](#13-uk-market-specifics--transaction-costs)
14. [Configuration](#14-configuration)
15. [Future Roadmap](#15-future-roadmap)
16. [Caveats & Limitations](#16-caveats--limitations)
17. [Dependencies](#17-dependencies)

---

## 1. Project Overview

Alpha Engine is a **systematic equity research platform** designed to mirror the research pipeline used in quantitative trading environments.

The core research question it addresses is: *given a candidate signal, does it contain genuine predictive power over future stock returns — and if so, how do we exploit it profitably after costs?*

The platform supports:

- **Factor generation** — create candidate signals from price and market data
- **Information Coefficient evaluation** — measure signal predictive power with statistical rigour
- **Factor ranking and leaderboard** — rank all tested signals; retain only the best
- **Multi-factor portfolio construction** — combine top signals into composite strategies
- **Historical backtesting** — simulate trading with realistic UK costs (commission, slippage, SDRT)
- **Walk-forward out-of-sample validation** — test robustness across rolling time periods
- **Paper trading** — run strategy signals live against current prices with simulated capital
- **Interactive Streamlit research interface** — explore every step through a browser UI

The architecture is modular: each component communicates through immutable DataFrames (`DatetimeIndex × tickers`) and can be replaced or extended independently.

---

## 2. Research Pipeline

The platform implements a structured, four-stage alpha research pipeline. This mirrors the workflow used in systematic equity funds to progress from raw signal ideas to deployable strategies.

```
generate_features()
        │
        ▼
rank_features_by_IC()
        │
        ▼
combine_top_features()
        │
        ▼
construct_risk_balanced_portfolio()
```

### Stage 1 — `generate_features()`

Computes candidate factor scores for every stock in the universe on each trading day. Current built-in factors:

| Factor | Signal construction | Hypothesis |
|---|---|---|
| **Momentum 12-1** | Cumulative return from t−252 to t−21 | Winners continue to outperform over medium-term horizons |
| **Mean Reversion 5d** | Negated 5-day return | Short-term losers revert; microstructure liquidity provision premium |
| **Earnings Revision** | Overnight gap proxy for earnings surprise | Post-Earnings Announcement Drift (PEAD) — markets underreact to surprises |

New factors are registered via a `@register` decorator and automatically appear in the research UI and factor portfolio strategy.

### Stage 2 — `rank_features_by_IC()`

For each candidate factor, computes the **Information Coefficient (IC)** — the Spearman rank correlation between daily factor scores and realised forward returns — across thousands of trading days. Produces a full IC summary:

- **IC Mean**: average signal quality across the sample
- **IC t-statistic**: hypothesis test for whether IC is distinguishable from noise
- **IC Information Ratio (ICIR)**: IC Mean ÷ IC Std × √252 — signal quality adjusted for consistency
- **Factor decay**: how IC evolves at t+1, t+5, t+10, t+20 — guides rebalancing frequency

Results are **automatically logged** to the factor leaderboard after every run, sorted by ICIR. Only the top-performing signals are retained.

### Stage 3 — `combine_top_features()`

The `FactorPortfolioStrategy` combines multiple ranked signals into a single composite factor using equal weighting. It constructs a cross-sectional long-short portfolio: long the top-decile stocks by composite score, short the bottom decile.

```python
from factors import get_factor
from strategies.factor_portfolio import FactorPortfolioStrategy

strategy = FactorPortfolioStrategy(
    config=strat_cfg,
    factor_names=["momentum_12_1", "mean_reversion_5d"],
    long_pct=0.10,
)
result = strategy.run(prices, returns)
```

### Stage 4 — `construct_risk_balanced_portfolio()`

The `PortfolioManager` applies risk controls and combines strategy signals:

- **Volatility targeting**: scales position sizes to achieve a target annual volatility (default 10%)
- **Leverage cap**: clips gross exposure (default 1.5×)
- **Max position size**: limits single-stock concentration (default 5%)
- **Multi-strategy combination**: equal-weight or volatility-scaled blending

```python
from portfolio.portfolio_manager import PortfolioManager, PortfolioConfig

pm = PortfolioManager(PortfolioConfig(), risk_cfg)
combined_weights = pm.combine_strategies(strategy_results)
controlled = pm.apply_risk_controls(combined_weights, returns)
```

---

## 3. Key Quant Concepts Implemented

### Information Coefficient (IC)

The IC is the **Spearman rank correlation** between a factor's cross-sectional ranking of stocks on day *t* and the realised returns *h* days later. It answers the question: *does ranking stocks by this factor tell us anything useful about which stocks will outperform?*

- IC = 0: the factor has no predictive power
- IC = 0.05: meaningful edge in practice for equity cross-sectional factors
- IC = 0.10: strong (rare for daily equity factors)
- |IC t-stat| > 2.0: statistically significant at ~95% confidence

The `factor_research/ic.py` module computes daily IC, rolling IC, and a full summary including t-statistic and ICIR over any factor-returns pair.

### IC Information Ratio (ICIR)

ICIR = IC Mean ÷ IC Std × √252 — the annualised signal-to-noise ratio of a factor. It penalises factors that have a high average IC but are highly inconsistent day-to-day, preferring stable predictors. ICIR > 0.5 is considered good; > 1.0 is strong.

### Factor Decay

The rate at which a factor's IC attenuates across forward horizons reveals its appropriate rebalancing frequency:

| Decay profile | Interpretation | Suitable rebalance |
|---|---|---|
| IC near zero by t+5 | Short-lived microstructure signal | Daily |
| IC persists to t+10 | Medium-term signal | Weekly |
| IC persists to t+20+ | Long-term signal | Monthly |

### Cross-Sectional Signals

Signals are computed as **ranks within the cross-section** on each day, not as time-series values. This eliminates look-ahead in volatility scaling, makes signals directly comparable across different market regimes, and ensures the long book and short book are always of equal size (dollar-neutral construction).

### Signal Combination

The `FactorPortfolioStrategy` combines multiple cross-sectional factors by averaging their daily rank signals before constructing the long-short portfolio. This diversifies across factor-specific noise and can improve ICIR over individual signals.

### Risk-Balanced Portfolios

Position sizing scales with the **inverse of realised volatility** to achieve consistent risk contribution from each stock. At the portfolio level, an exponentially-weighted volatility estimate targets a fixed annual volatility, scaling gross exposure up in low-volatility regimes and down in high-volatility regimes.

### Backtesting Methodology

- **Execution lag**: signals at close[t] execute at open[t+1] via `weights.shift(1)` — no look-ahead bias
- **Daily P&L**: `(weights_executed × returns).sum(axis=1)`
- **Realistic costs**: linear costs + stamp duty subtracted from daily returns
- **Vectorised**: all operations across the full date × ticker matrix; no Python loops over dates

---

## 4. System Architecture

Alpha Engine is structured as a **modular research pipeline**. Each layer has a single responsibility and communicates through immutable DataFrames:

```
data/data_loader.py          Downloads from Yahoo Finance, caches to data/cache/ as parquet
data/universe.py             FTSE 100 / 250 / All tickers with liquidity filters
         │
         ▼
factors/*.py                 Stateless factor computation (registered via @register)
         │
         ▼
factor_research/             IC analysis, factor decay, quantile portfolios
         │
         ▼
data/factor_research_log.py  Persistent leaderboard (parquet, ICIR-ranked, top-100)
         │
         ▼
strategies/*.py              Stateless signal generation + portfolio construction
         │
         ▼
portfolio/portfolio_manager  Multi-strategy combination (equal_weight / volatility_scale)
         │
         ▼
backtester/engine.py         Vectorised backtest with costs → BacktestResult
         │
         ▼
analytics/metrics.py         Sharpe, Sortino, Calmar, alpha/beta, VaR, ICIR…
analytics/visualizations.py  PNG charts saved to results/
```

### Project Structure

```
alpha-engine/
├── app.py                       # Streamlit navigation router
├── main.py                      # CLI entry point
├── config.py                    # DataConfig, StrategyConfig, RiskConfig dataclasses
├── requirements.txt
│
├── pages/
│   ├── factor_research.py       # IC analysis, factor decay, quantile portfolios UI
│   ├── leaderboard.py           # Factor leaderboard — ranked signals with badges
│   ├── backtest.py              # Historical backtest UI with risk profiles
│   ├── paper_trading.py         # Simulated live trading (no real money)
│   └── help.py                  # Reference documentation
│
├── factors/
│   ├── factor_registry.py       # @register decorator + get_factor() / list_factors()
│   ├── momentum_12_1.py         # 12-1 month cross-sectional momentum
│   └── mean_reversion_5d.py     # 5-day short-term reversal
│
├── factor_research/
│   ├── ic.py                    # IC daily/rolling/summary + t-statistic
│   ├── decay.py                 # IC at t+1/5/10/20 horizons
│   └── quantile.py              # Quantile portfolio analysis + L/S spread
│
├── data/
│   ├── data_loader.py           # yfinance batch download, universe-level parquet cache
│   ├── universe.py              # FTSE 100/250/All tickers with liquidity filters
│   ├── benchmark.py             # Vanguard-style passive benchmark
│   └── factor_research_log.py   # Leaderboard persistence layer
│
├── strategies/
│   ├── base.py                  # BaseStrategy ABC + StrategyResult
│   ├── momentum.py              # Cross-sectional 12-1 momentum
│   ├── mean_reversion.py        # 5-day short-term mean reversion
│   ├── earnings_revision.py     # PEAD overnight gap strategy
│   └── factor_portfolio.py      # Multi-factor composite strategy
│
├── backtester/
│   ├── engine.py                # Vectorised BacktestEngine (GBP output)
│   ├── walk_forward.py          # Expanding/rolling IS-OOS folds
│   └── parameter_sweep.py       # Grid search with multiprocessing
│
├── portfolio/
│   └── portfolio_manager.py     # Strategy combination, volatility targeting, risk controls
│
├── analytics/
│   ├── metrics.py               # Sharpe, Sortino, Calmar, VaR, alpha/beta, rolling Sharpe
│   └── visualizations.py        # Equity curve, drawdown, heatmap (£ labels)
│
├── validation/
│   ├── oos.py                   # Out-of-sample test + walk-forward framework
│   └── stability.py             # Parameter stability / overfitting detection
│
├── execution/
│   └── transaction_costs.py     # UK costs: commission + slippage + 0.5% SDRT
│
├── paper_trading/
│   └── portfolio.py             # PaperPortfolio: JSON-persisted simulated positions
│
├── data/cache/                  # Universe-level parquet price cache (auto-created)
├── data/factor_results/         # Factor leaderboard parquet (auto-created)
├── results/                     # Output charts and CSVs
└── logs/                        # Run logs
```

---

## 5. Factor Leaderboard

Every factor research run is **automatically logged** to a persistent parquet store at `data/factor_results/factor_leaderboard.parquet`. No manual save is required.

The leaderboard enforces:

1. **ICIR ranking** — always sorted by IC Information Ratio descending; the best-performing signals float to the top
2. **Update if improved** — if the same factor configuration is run again with a higher ICIR, the entry is updated; if the new run is weaker, it is ignored
3. **Top-100 cap** — only the top 100 entries by ICIR are retained; weak factors automatically fall off

Each leaderboard entry records:

| Field | Description |
|---|---|
| `factor_name` | Registered factor identifier |
| `universe` | FTSE 100 / FTSE 250 / FTSE All-Share |
| `ic_mean` | Mean daily IC over the evaluation period |
| `ic_ir` | Annualised IC Information Ratio |
| `ic_tstat` | t-statistic — significance of IC |
| `pct_positive_ic` | Fraction of days with positive IC |
| `spread_annual_return` | L/S spread annual return (pre-cost) |
| `monotonicity_score` | How cleanly quantile returns increase with factor rank |
| `timestamp` | When the analysis was run |

The **Leaderboard page** in the Streamlit UI visualises the ranked factor library with:

- **Robustness badges** (Robust / Promising / Weak) based on ICIR and t-stat thresholds
- **IC Mean vs |t-stat| scatter** — visual factor quality map
- **Top-10 bar chart** ranked by absolute t-statistic
- **"Use in Backtest"** — one click promotes any leaderboard factor to the Backtest page as a Factor Portfolio strategy

---

## 6. Supported Universes

| Universe | Tickers | Coverage |
|---|---|---|
| **FTSE 100** | ~100 large-cap UK equities | Blue-chip, highest liquidity |
| **FTSE 250** | ~250 mid-cap UK equities | Broader exposure, higher alpha potential |
| **FTSE All-Share** | ~350 tickers combined | Maximum cross-sectional breadth |

All tickers use the `.L` suffix convention for Yahoo Finance (e.g. `AZN.L`, `HSBA.L`, `LLOY.L`). Liquidity filters remove stocks below a minimum price (£0.50) and average daily volume threshold before each research run.

The universe module (`data/universe.py`) is designed for extension — adding a new index requires only a ticker list and a string identifier.

---

## 7. Technology Stack

| Technology | Role |
|---|---|
| **Python 3.10+** | Core language |
| **Pandas / NumPy** | Vectorised time-series operations across date × ticker matrices |
| **SciPy / scikit-learn** | Rank correlation (IC), statistical tests, cross-validation |
| **yfinance** | LSE equity price data, adjusted close + OHLCV |
| **pyarrow / parquet** | Compressed price cache and factor leaderboard persistence |
| **pandas-market-calendars** | LSE trading calendar — accurate business day handling |
| **Streamlit** | Interactive browser-based research interface |
| **Matplotlib / Seaborn** | Equity curves, drawdown charts, monthly return heatmaps |

The platform is designed for **systematic financial research**: vectorised operations throughout (no Python loops over dates), stateless strategies, immutable data pipelines, and configurable risk controls. All parameters live in typed dataclasses — no global state or mutable configuration.

---

## 8. Example Research Workflow

This is how a researcher interacts with Alpha Engine to go from a signal idea to a tested strategy:

### Step 1 — Select universe and date range

Open the **Factor Research** page in the Streamlit UI. Select FTSE 100 (100 stocks, 2018–2023) for an initial test. This gives ~1,500 trading days × 100 stocks = 150,000 signal-return pairs per IC computation.

### Step 2 — Generate and evaluate a candidate factor

Select the **Momentum 12-1** factor. Click **Run Factor Analysis**.

The platform computes:
- Daily Spearman IC between factor scores and 1-day forward returns
- Rolling 63-day IC to detect regime dependency
- Factor decay table at t+1, t+5, t+10, t+20
- 5-quantile portfolio returns and monotonicity score

Example output for momentum on FTSE 100 (2018–2023):

```
IC Mean:    0.028    IC Std:   0.071
IC t-stat:  4.21     IC IR:    0.63
% Positive: 58.3%    Obs:     1,474

L/S Spread Ann. Return:  8.2%
Monotonicity Score:      0.85
```

### Step 3 — Review and auto-log to leaderboard

The result is automatically written to the leaderboard. Navigate to the **Leaderboard** page to see it ranked against all previously tested signals, with a visual quality map and robustness badge.

### Step 4 — Promote to backtest

Click **Use in Backtest** on the Leaderboard page. The factor is pre-loaded into the Backtest page as a Factor Portfolio strategy.

### Step 5 — Run a full historical backtest

On the **Backtest** page:

```
Universe:   FTSE 100
Date range: 2015-01-01 → 2023-12-31
Strategy:   Factor Portfolio (Momentum 12-1 + Mean Reversion 5d)
Risk:       Balanced profile (vol target 10%, max leverage 1.5×)
```

Review equity curves, drawdowns, monthly return heatmap, rolling Sharpe, and benchmark comparison (vs Vanguard-style passive). Download performance metrics CSV.

### Step 6 — Validate robustness

Run walk-forward validation to confirm the strategy holds up on successive out-of-sample periods:

```bash
python main.py --walk-forward --strategy momentum
```

Check that OOS Sharpe is consistent across folds — a strategy that degrades sharply out-of-sample is overfit to in-sample noise.

---

## 9. Running the Application

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the interactive Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens a browser at `http://localhost:8501`. The app has five pages:

| Page | What it does |
|---|---|
| **Factor Research** | Evaluate any registered signal using IC, decay, and quantile analysis |
| **Leaderboard** | Browse all tested factors ranked by ICIR with robustness badges |
| **Backtest** | Full historical simulation with costs, drawdowns, and benchmark comparison |
| **Paper Trading** | Run live signals on current market data with simulated capital |
| **Help** | Reference documentation for all metrics, strategies, and parameters |

### Run from the command line

```bash
# All strategies on FTSE All-Share (~350 stocks), 2015–2024
python main.py

# Single strategy
python main.py --strategy momentum
python main.py --strategy mean_reversion

# Specific universe and date range
python main.py --universe ftse100 --start 2018-01-01 --end 2023-12-31

# Walk-forward out-of-sample validation
python main.py --walk-forward --strategy momentum

# Parameter grid search
python main.py --sweep --strategy momentum

# Force re-download of price data (clear cache)
python main.py --refresh-data
```

### Output files

After a CLI run, `results/` contains:

| File | Description |
|---|---|
| `performance_metrics.csv` | All strategies, all metrics |
| `{strategy}_dashboard.png` | 6-panel performance dashboard |
| `{strategy}_equity.png` | Equity curve (£) |
| `{strategy}_drawdown.png` | Drawdown chart |
| `{strategy}_monthly.png` | Monthly returns heatmap |
| `strategy_comparison.png` | Side-by-side bar chart |
| `equity_curves_all.png` | Overlaid normalised equity curves |

---

## 10. Strategies Reference

### Cross-Sectional Momentum (`strategies/momentum.py`)

Buys recent winners and shorts recent losers across the universe. Based on the Jegadeesh-Titman (1993) momentum anomaly, extended to the UK equity market.

| Parameter | Default | Description |
|---|---|---|
| `momentum_lookback` | 252 | Formation period in trading days |
| `momentum_skip` | 21 | Recent days excluded (avoids short-term reversal) |
| `momentum_long_pct` | 0.10 | Top/bottom decile per book |

**Signal**: `return(t−252, t−21)` — the "12-1" momentum factor.

### Short-Term Mean Reversion (`strategies/mean_reversion.py`)

Contrarian strategy. Buys the week's biggest losers, shorts the biggest winners. Exploits the liquidity provision premium and short-term price impact reversal.

| Parameter | Default | Description |
|---|---|---|
| `mean_reversion_lookback` | 5 | Formation period in trading days |
| `mean_reversion_long_pct` | 0.20 | Bottom/top quintile per book |

**Signal**: `−return(t−5, t)` — inverted so high signal = long bias.

### Earnings Revision Drift (`strategies/earnings_revision.py`)

Approximates Post-Earnings Announcement Drift (PEAD) using overnight price gaps as a proxy for earnings surprises. Markets systematically underreact to earnings news over the subsequent 10 trading days.

| Parameter | Default | Description |
|---|---|---|
| `earnings_gap_threshold` | 0.05 | Minimum \|gap\| to trigger (5%) |
| `earnings_hold_days` | 10 | Days to hold after signal |

**Signal**: `open[t] / close[t−1] − 1` where `|gap| > 5%`. Long on positive surprise, short on negative.

### Factor Portfolio (`strategies/factor_portfolio.py`)

Combines any set of registered factors into a composite signal, then constructs a long-short portfolio. Factors are combined by averaging their cross-sectional rank signals before decile selection.

### Adding a New Strategy

1. Create `strategies/my_strategy.py`:

```python
from strategies.base import BaseStrategy
from config import StrategyConfig
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig) -> None:
        super().__init__("MyStrategy", config)

    def generate_signals(self, prices: pd.DataFrame, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Return same-shape DataFrame: higher value = stronger long signal, NaN = no position
        ...

    def construct_portfolio(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        # Convert signals to signed weights using self._rank_cross_section() helpers
        ...
```

2. Register it in `strategies/__init__.py` and add to the strategy map in `main.py`.

### Adding a New Factor

```python
# factors/my_factor.py
from factors.base_factor import BaseFactor
from factors.factor_registry import register
import pandas as pd

@register(name="my_factor", description="My custom factor")
class MyFactor(BaseFactor):
    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        # Return a date × ticker DataFrame of raw signal values
        ...
```

The factor is automatically available in the Factor Research dropdown and `FactorPortfolioStrategy`.

---

## 11. Statistical Validation

The `validation/` module provides tools to test whether a strategy's performance is genuine or the product of overfitting.

### Out-of-Sample Test

```python
from validation.oos import run_oos_test
from strategies.momentum import CrossSectionalMomentum

strategy = CrossSectionalMomentum(StrategyConfig())

result = run_oos_test(
    weights_fn=lambda p, r: strategy.run(p, r).weights,
    prices=prices,
    returns=returns,
    train_start="2015-01-01", train_end="2019-12-31",
    test_start="2020-01-01", test_end="2023-12-31",
)
print(f"Train Sharpe: {result.train_metrics['sharpe_ratio']:.2f}")
print(f"Test Sharpe:  {result.test_metrics['sharpe_ratio']:.2f}")
print(f"Degradation:  {result.degradation['sharpe_ratio']:.2f}×")
```

### Walk-Forward Validation

Rolls the training window forward, testing on successive non-overlapping OOS periods (36-month in-sample / 12-month OOS, expanding window):

```bash
python main.py --walk-forward --strategy momentum
```

```python
from validation.oos import run_walk_forward

wf = run_walk_forward(
    weights_fn=lambda p, r: strategy.run(p, r).weights,
    prices=prices, returns=returns,
    train_months=36, test_months=12, expanding=True,
)
print(f"OOS Sharpe:            {wf.oos_metrics['sharpe_ratio']:.2f}")
print(f"Positive Sharpe folds: {wf.sharpe_consistency:.0%}")
```

### Parameter Stability

Tests whether performance is robust to small parameter changes. A strategy that only works for one specific lookback is likely overfit.

```python
from validation.stability import run_parameter_stability

result = run_parameter_stability(
    weights_fn_factory=lambda params: (
        lambda p, r: CrossSectionalMomentum(
            StrategyConfig(momentum_lookback=params["lookback"])
        ).run(p, r).weights
    ),
    prices=prices, returns=returns,
    param_grid={"lookback": [126, 189, 252, 315], "long_pct": [0.10, 0.15, 0.20]},
)
print(f"Best Sharpe:     {result.best_sharpe:.2f} at {result.best_params}")
print(f"Stability score: {result.stability_score:.2f}  (1.0 = perfectly stable)")
```

---

## 12. Portfolio Construction Utilities

The `portfolio/portfolio_manager.py` module exposes cross-sectional selection and weighting helpers:

```python
from portfolio import select_top_n, select_top_pct, signal_proportional_weights, rank_cross_section

# Select top 20 stocks by signal each day (boolean mask)
long_mask = select_top_n(signals, n=20, long=True)

# Select top 10% each day
long_mask_pct = select_top_pct(signals, pct=0.10, long=True)

# Signal-proportional weights (z-scored, gross = 1.0)
weights = signal_proportional_weights(signals, long_only=False)

# Cross-sectional percentile ranks
ranks = rank_cross_section(signals, ascending=True, pct=True)
```

Programmatic backtest example:

```python
from config import FrameworkConfig
from data.data_loader import DataLoader, clean_data, compute_returns
from data.universe import load_ftse_universe
from strategies.momentum import CrossSectionalMomentum
from backtester.engine import BacktestEngine
from analytics.metrics import compute_metrics

config = FrameworkConfig()
loader = DataLoader(cache_dir="data/cache", config=config.data)
tickers = load_ftse_universe("ftse100", top_n=100)
prices = loader.load_price_data(tickers, "2018-01-01", "2023-12-31")
prices = clean_data(prices)
returns = compute_returns(prices)

strategy = CrossSectionalMomentum(config.strategy)
result = strategy.run(prices, returns)

engine = BacktestEngine(config.risk, initial_capital=1_000_000)
bt = engine.run(result.weights, returns, strategy_name="Momentum")

metrics = compute_metrics(bt.daily_returns)
print(f"Sharpe:       {metrics['sharpe_ratio']:.2f}")
print(f"Max DD:       {metrics['max_drawdown']:.1%}")
print(f"Final equity: £{bt.equity_curve.iloc[-1]:,.0f}")
```

---

## 13. UK Market Specifics & Transaction Costs

### Market conventions

| Concern | Implementation |
|---|---|
| Ticker format | `.L` suffix throughout (e.g. `AZN.L`, `HSBA.L`) |
| Trading calendar | `pandas_market_calendars` LSE calendar — excludes UK bank holidays |
| Currency | GBP throughout; initial capital defaults to £1,000,000 |
| Stamp duty | 0.5% SDRT on gross long purchases, tracked separately |
| Slippage | 10 bps/side (wider than US to reflect LSE spreads) |
| Liquidity filter | Min £0.50 price, min £1M average daily volume |
| Max leverage | 1.5× (conservative; configurable) |

### Transaction cost model

Every backtest applies realistic UK trading costs:

```
Total cost per trade = commission (10 bps) + slippage (10 bps) + spread (5 bps)
                     = 25 bps per side on turnover

Stamp Duty Reserve Tax (SDRT) = 0.5% on gross long purchases only
```

Round-trip cost analysis:

| Position side | Buy + sell cost |
|---|---|
| **Long** | ~75 bps (buy incl. SDRT) + ~25 bps (sell) = ~100 bps round-trip |
| **Short** | ~25 bps (entry) + ~25 bps (cover) = ~50 bps round-trip |

`BacktestResult` separates `linear_costs` and `stamp_duty_costs` for independent analysis of SDRT drag.

High-frequency strategies (e.g. daily mean reversion) are highly sensitive to these costs — a primary reason factor research (IC evaluation) precedes any backtest.

---

## 14. Configuration

All parameters are defined in `config.py` as Python dataclasses:

```python
from config import FrameworkConfig, StrategyConfig, RiskConfig

config = FrameworkConfig()

# Tighten momentum decile
config.strategy.momentum_long_pct = 0.05

# Higher leverage tolerance
config.risk.max_leverage = 2.0

# Longer backtest window
config.data.start_date = "2010-01-01"
```

Key defaults: initial capital £1M, max leverage 1.5×, max position 5%, target vol 10%, commission + slippage + spread = 25 bps/side, SDRT 0.5%.

---

## 15. Future Roadmap

| Area | Enhancement |
|---|---|
| **Factor models** | Machine learning factor construction (random forests, gradient boosting, neural factor models) |
| **Alternative data** | News sentiment and NLP-driven signals via LLM-based text processing |
| **Regime detection** | Hidden Markov Models for market regime classification; regime-conditional factor selection |
| **Risk modelling** | Factor risk models (Barra-style) for ex-ante portfolio risk decomposition |
| **Transaction costs** | Almgren-Chriss market impact model; optimal execution scheduling |
| **Live monitoring** | Real-time signal dashboard with automated daily factor re-evaluation |
| **Universe expansion** | European (Eurostoxx), US (Russell 1000/3000), and EM equity universes |
| **Portfolio optimisation** | Mean-variance and Black-Litterman portfolio construction with factor views |
| **Short-selling costs** | Stock borrow rate integration for more accurate long-short cost modelling |

---

## 16. Caveats & Limitations

- **Survivorship bias**: The FTSE 100/250 universe is a static list of current constituents. Historical index changes, delistings, and additions are not modelled. Results will be optimistic vs true live performance.
- **No short-selling frictions**: Stock borrow costs for short positions are not modelled. Strategies with heavy short books will be more attractive than in practice.
- **Data quality**: Yahoo Finance data can contain errors, corporate action adjustment issues, and gaps. Inspect raw data before drawing conclusions.
- **Transaction costs are estimates**: Real costs depend on broker, order size, and market conditions. The 25 bps model is reasonable but not exact.
- **Past performance**: No guarantee of future results. Factor performance is non-stationary; many anomalies have decayed post-publication.

---

## 17. Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | LSE price data download |
| `pandas` / `numpy` | Data manipulation, vectorised operations |
| `pyarrow` | Parquet cache storage for prices and leaderboard |
| `pandas-market-calendars` | LSE trading calendar |
| `matplotlib` / `seaborn` | Charts and heatmaps |
| `scipy` / `scikit-learn` | Rank correlation, statistical tests, cross-validation |
| `streamlit` | Interactive browser-based research interface |
| `tqdm` | Progress bars for batch downloads |

---

## Licence

MIT
