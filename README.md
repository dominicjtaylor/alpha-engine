# Alpha Engine — UK Systematic Trading Research Framework

A modular Python framework for quantitative equity research on **London Stock Exchange (LSE)** listed securities. Backtest evidence-based strategies, combine them into a multi-strategy portfolio, and analyse performance — all denominated in **GBP (£)**.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the interactive app (recommended)

```bash
streamlit run app.py
```

Opens a browser UI where you can configure universes, strategies, date ranges, and risk parameters — no code required.

### 3. Run from the command line

```bash
# All three strategies on FTSE All (~350 stocks), 2015–2024
python main.py

# Single strategy
python main.py --strategy momentum

# Specific universe and date range
python main.py --universe ftse100 --start 2018-01-01 --end 2023-12-31

# Walk-forward out-of-sample validation
python main.py --walk-forward --strategy momentum

# Parameter grid search
python main.py --sweep --strategy momentum
```

---

## Project Structure

```
alpha-engine/
├── app.py                     # Streamlit home page
├── main.py                    # CLI entry point
├── config.py                  # All parameters (DataConfig, StrategyConfig, RiskConfig)
├── requirements.txt
│
├── pages/
│   ├── backtest.py            # Streamlit: historical backtest UI
│   ├── paper_trading.py       # Streamlit: simulated live trading (fake money)
│   └── factor_research.py     # Streamlit: IC, factor decay, quantile analysis
│
├── data/
│   ├── universe.py            # FTSE 100 / 250 ticker lists (.L suffix)
│   └── data_loader.py         # yfinance download + parquet cache + LSE calendar
│
├── strategies/
│   ├── base.py                # BaseStrategy ABC + StrategyResult
│   ├── momentum.py            # Cross-sectional 12-1 momentum
│   ├── mean_reversion.py      # 5-day short-term mean reversion
│   └── earnings_revision.py   # Overnight gap / PEAD drift
│
├── backtester/
│   ├── engine.py              # Vectorised BacktestEngine (GBP output)
│   ├── walk_forward.py        # Expanding / rolling IS-OOS validation
│   └── parameter_sweep.py     # Grid search with multiprocessing
│
├── portfolio/
│   └── portfolio_manager.py   # Strategy combination + risk controls
│                              # + ranking utils: select_top_n, signal_proportional_weights
│
├── execution/
│   └── transaction_costs.py   # UK costs: commission + slippage + 0.5% SDRT
│
├── analytics/
│   ├── metrics.py             # Sharpe, Sortino, Calmar, VaR, alpha/beta…
│   └── visualizations.py      # Equity curve, drawdown, heatmap (£ labels)
│
├── factor_research/            # ── NEW: Factor Research Toolkit ──
│   ├── ic.py                  # Information Coefficient (daily, rolling, summary)
│   ├── decay.py               # Factor decay at t+1/5/10/20 horizons
│   └── quantile.py            # Quantile portfolio analysis + L/S spread
│
├── validation/                 # ── NEW: Statistical Validation ──
│   ├── oos.py                 # Out-of-sample testing + walk-forward framework
│   └── stability.py           # Parameter stability sweep (overfitting detection)
│
├── paper_trading/
│   └── portfolio.py           # PaperPortfolio: JSON-persisted simulated positions
│
├── data/cache/                # Auto-created parquet price cache
├── results/                   # Output charts and CSVs
└── logs/                      # Run logs
```

---

## Strategies

### 1. Cross-Sectional Momentum (`strategies/momentum.py`)

Buys recent winners and shorts recent losers across the universe.

| Parameter | Default | Description |
|---|---|---|
| `momentum_lookback` | 252 | Formation period in trading days |
| `momentum_skip` | 21 | Recent days excluded (avoids reversal) |
| `momentum_long_pct` | 0.10 | Top/bottom decile per book |

Signal: `return(t−252, t−21)` — the "12-1" momentum factor.

### 2. Short-Term Mean Reversion (`strategies/mean_reversion.py`)

Contrarian strategy. Buys the week's biggest losers, shorts the biggest winners.

| Parameter | Default | Description |
|---|---|---|
| `mean_reversion_lookback` | 5 | Formation period in trading days |
| `mean_reversion_long_pct` | 0.20 | Bottom/top quintile per book |

Signal: `−return(t−5, t)` — inverted so high signal = long bias.

### 3. Earnings Revision Drift (`strategies/earnings_revision.py`)

Approximates Post-Earnings Announcement Drift (PEAD) using overnight price gaps as a proxy for earnings surprises.

| Parameter | Default | Description |
|---|---|---|
| `earnings_gap_threshold` | 0.05 | Minimum |gap| to trigger (5%) |
| `earnings_hold_days` | 10 | Days to hold after signal |

Signal: `open[t] / close[t−1] − 1` where `|gap| > 5%`. Long on positive surprise, short on negative.

---

## UK Market Specifics

| Concern | Implementation |
|---|---|
| Ticker format | All tickers use `.L` suffix (e.g. `AZN.L`, `HSBA.L`) |
| Trading calendar | `pandas_market_calendars` LSE calendar — excludes UK bank holidays |
| Currency | GBP throughout; initial capital defaults to £1,000,000 |
| Stamp duty | 0.5% SDRT on gross long purchases, tracked separately |
| Slippage | 10 bps/side (wider than US to reflect LSE spreads) |
| Liquidity filters | Min £0.50 price, min £1M average daily volume |
| Max leverage | 1.5× (conservative for retail accounts) |

---

## Transaction Costs

Every backtest applies realistic UK trading costs:

```
Total cost per trade = commission (10 bps) + slippage (10 bps) + spread (5 bps)
                     = 25 bps per side on turnover

Stamp duty = 0.5% on gross long purchases only (SDRT)
```

The `BacktestResult` separates `linear_costs` and `stamp_duty_costs` so you can see the SDRT impact independently.

---

## Configuration

All parameters live in `config.py`. Override defaults by editing the dataclasses or passing modified configs programmatically:

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

---

## Running the Backtest Programmatically

```python
from config import FrameworkConfig
from data.data_loader import DataLoader, clean_data, compute_returns
from data.universe import load_ftse_universe
from strategies.momentum import CrossSectionalMomentum
from backtester.engine import BacktestEngine
from analytics.metrics import compute_metrics

config = FrameworkConfig()

# Load data
loader = DataLoader(cache_dir="data/cache", config=config.data)
tickers = load_ftse_universe("ftse100", top_n=100)
prices = loader.load_price_data(tickers, "2018-01-01", "2023-12-31")
prices = clean_data(prices)
returns = compute_returns(prices)

# Run strategy
strategy = CrossSectionalMomentum(config.strategy)
result = strategy.run(prices, returns)

# Backtest
engine = BacktestEngine(config.risk, initial_capital=1_000_000)
bt = engine.run(result.weights, returns, strategy_name="Momentum")

# Metrics
metrics = compute_metrics(bt.daily_returns)
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Max DD: {metrics['max_drawdown']:.1%}")
print(f"Final equity: £{bt.equity_curve.iloc[-1]:,.0f}")
```

---

## Adding a New Strategy

1. Create `strategies/my_strategy.py`:

```python
from strategies.base import BaseStrategy
from config import StrategyConfig
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig) -> None:
        super().__init__("MyStrategy", config)

    def generate_signals(self, prices: pd.DataFrame, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Your signal logic here — return same-shape DataFrame
        # Higher value = stronger long signal, NaN = no position
        ...

    def construct_portfolio(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        # Convert signals to signed weights
        # Use self._rank_cross_section() and self._equal_weight_book() helpers
        ...
```

2. Register it in `strategies/__init__.py` and add it to the strategy map in `main.py` and `app.py`.

---

## Walk-Forward Validation

```bash
python main.py --walk-forward --strategy momentum
```

Splits history into 36-month in-sample / 12-month out-of-sample folds (expanding window). Reports OOS metrics only — the IS period is never evaluated. This gives a realistic picture of live trading performance.

---

## Parameter Sweep

```bash
python main.py --sweep --strategy momentum
```

Grid searches `momentum_lookback × momentum_skip × momentum_long_pct`. Results saved to `results/momentum_sweep_results.csv`, sorted by Sharpe ratio. **Always validate best parameters on out-of-sample data to avoid overfitting.**

---

## Output Files

After a run, `results/` contains:

| File | Description |
|---|---|
| `performance_metrics.csv` | All strategies, all metrics |
| `{strategy}_dashboard.png` | 6-panel performance dashboard |
| `{strategy}_equity.png` | Equity curve (£) |
| `{strategy}_drawdown.png` | Drawdown chart |
| `{strategy}_monthly.png` | Monthly returns heatmap |
| `strategy_comparison.png` | Side-by-side bar chart |
| `equity_curves_all.png` | Overlaid normalised curves |

---

---

## Factor Research Toolkit

The `factor_research/` module lets you evaluate a signal **as a factor** rather than running a full backtest.

### Information Coefficient (IC)

Measures the rank correlation between the factor value on day *t* and the realised return *t+h* days later.  IC near 0 = no edge; IC > 0.05 = meaningful signal in practice.

```python
from factor_research.ic import compute_ic, compute_rolling_ic, compute_ic_summary

# Daily IC between momentum signal and 1-day forward returns
daily_ic = compute_ic(signals, fwd_returns, method="spearman")

# Rolling 63-day IC
rolling_ic = compute_rolling_ic(signals, fwd_returns, window=63)

# Summary stats: mean, std, t-stat, ICIR
summary = compute_ic_summary(signals, fwd_returns)
print(f"IC mean={summary['ic_mean']:.3f}  t-stat={summary['ic_tstat']:.2f}")
```

### Factor Decay

Tests whether the factor edge persists across multiple forward horizons.

```python
from factor_research.decay import compute_factor_decay

decay = compute_factor_decay(signals, prices, horizons=(1, 5, 10, 20))
# Returns a DataFrame indexed by horizon with IC mean/std/tstat at each horizon
```

| Decay profile | Suitable strategy |
|---|---|
| Fast decay (IC near 0 by t+5) | Daily mean-reversion |
| Medium decay (IC persists to t+10) | Weekly rebalancing |
| Slow decay (IC persists to t+20+) | Monthly momentum |

### Quantile Portfolio Analysis

Sorts stocks into N quantile buckets by factor value each day and tracks the average return of each bucket.

```python
from factor_research.quantile import compute_quantile_portfolios

qr = compute_quantile_portfolios(signals, fwd_returns, n_quantiles=5)
print(f"Monotonicity score: {qr.monotonicity_score:.2f}")
print(f"L/S annual return: {qr.spread_annual_return:.1%}")
print(f"L/S Sharpe: {qr.spread_sharpe:.2f}")
```

---

## Statistical Validation

The `validation/` module provides tools to test whether a strategy's performance is genuine or overfit.

### Out-of-Sample Test

Fix a training period, then evaluate on a held-out test period with zero look-ahead.

```python
from validation.oos import run_oos_test
from strategies.momentum import CrossSectionalMomentum
from config import StrategyConfig

strategy = CrossSectionalMomentum(StrategyConfig())

result = run_oos_test(
    weights_fn=lambda p, r: strategy.run(p, r).weights,
    prices=prices,
    returns=returns,
    train_start="2015-01-01",
    train_end="2019-12-31",
    test_start="2020-01-01",
    test_end="2023-12-31",
)
print(f"Train Sharpe: {result.train_metrics['sharpe_ratio']:.2f}")
print(f"Test Sharpe:  {result.test_metrics['sharpe_ratio']:.2f}")
print(f"Degradation:  {result.degradation['sharpe_ratio']:.2f}×")
```

### Walk-Forward Validation

Rolls the training window forward, testing on successive non-overlapping OOS periods.

```python
from validation.oos import run_walk_forward

wf = run_walk_forward(
    weights_fn=lambda p, r: strategy.run(p, r).weights,
    prices=prices,
    returns=returns,
    train_months=36,
    test_months=12,
    expanding=True,
)
print(f"OOS Sharpe: {wf.oos_metrics['sharpe_ratio']:.2f}")
print(f"Positive Sharpe folds: {wf.sharpe_consistency:.0%}")
```

### Parameter Stability

Tests whether performance is robust to small parameter changes.  A strategy that only works for one specific lookback is likely overfit.

```python
from validation.stability import run_parameter_stability

result = run_parameter_stability(
    weights_fn_factory=lambda params: (
        lambda p, r: CrossSectionalMomentum(
            StrategyConfig(momentum_lookback=params["lookback"])
        ).run(p, r).weights
    ),
    prices=prices,
    returns=returns,
    param_grid={"lookback": [126, 189, 252, 315], "long_pct": [0.10, 0.15, 0.20]},
)
print(f"Best Sharpe: {result.best_sharpe:.2f} at {result.best_params}")
print(f"Stability score: {result.stability_score:.2f}  (1.0 = perfectly stable)")
```

---

## Portfolio Construction Utilities

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

---

## Caveats and Limitations

- **Survivorship bias**: The FTSE 100/250 universe is a static list of current constituents. Historical index changes are not modelled. Results are likely optimistic vs true live performance.
- **No short-selling frictions**: Borrow costs for short positions are not modelled.
- **Data quality**: Yahoo Finance data can have errors, corporate action adjustments, and gaps. Always inspect raw data before drawing conclusions.
- **Transaction costs are estimates**: Real costs depend on broker, order size, and market conditions.
- **Past performance**: No guarantee of future results.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | LSE price data download |
| `pandas` / `numpy` | Data manipulation |
| `pyarrow` | Parquet cache storage |
| `pandas-market-calendars` | LSE trading calendar |
| `matplotlib` / `seaborn` | Charts |
| `scipy` / `scikit-learn` | Statistics |
| `streamlit` | Interactive web UI |
| `tqdm` | Progress bars |

---

## Licence

MIT
