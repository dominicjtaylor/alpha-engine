# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest (all strategies, FTSE All ~350 stocks, 2015-2024)
python main.py

# Run specific strategy
python main.py --strategy momentum
python main.py --strategy mean_reversion
python main.py --strategy earnings

# Walk-forward OOS validation
python main.py --walk-forward --strategy momentum

# Parameter sweep (grid search)
python main.py --sweep --strategy momentum

# Launch interactive Streamlit UI
streamlit run app.py

# Run tests
pytest
```

## Architecture

The framework is a modular, vectorized pipeline for UK equity research. All components communicate through immutable DataFrames (DatetimeIndex × tickers) and configuration dataclasses.

**Core design rules:**
- Vectorized operations throughout — no Python loops over dates
- No look-ahead bias: signals at close[t] execute at open[t+1] via `shift(1)`
- Strategies are stateless (no mutable state between calls)
- All prices and returns are GBP-denominated
- UK transaction costs: commission + slippage + 0.5% Stamp Duty Reserve Tax (SDRT) on long buys

### Data flow

```
data/data_loader.py   → downloads from Yahoo Finance, caches to data/cache/ as parquet
data/universe.py      → FTSE100/250/All tickers with liquidity filters
         ↓
strategies/*.py       → stateless signal generation + portfolio construction → StrategyResult
         ↓
portfolio/portfolio_manager.py  → multi-strategy combination (equal_weight or volatility_scale)
         ↓
backtester/engine.py  → vectorized backtest with costs → BacktestResult
         ↓
analytics/metrics.py  → Sharpe, Sortino, Calmar, alpha/beta, etc.
analytics/visualizations.py → PNG charts saved to results/
```

### Configuration (`config.py`)

All parameters are in dataclasses — `DataConfig`, `StrategyConfig`, `RiskConfig` — wrapped by `FrameworkConfig`. To override a parameter:

```python
config = FrameworkConfig()
config.strategy.momentum_lookback = 189
strategy = CrossSectionalMomentum(config.strategy)
```

Key defaults: initial capital £1M, max leverage 1.5×, max position 5%, target vol 10%, commission+slippage+spread = 25bps total, SDRT 0.5%.

### Strategies (`strategies/`)

All inherit from `BaseStrategy` and implement `generate_signals()` and `construct_portfolio()`.

| Strategy | Signal | Rebalance |
|---|---|---|
| `momentum.py` | 12-1 month cross-sectional return (top/bottom decile) | Monthly |
| `mean_reversion.py` | Negated 5-day return (bottom/top quintile) | Daily |
| `earnings_revision.py` | Overnight gap proxy for PEAD (>5% threshold, 10-day hold) | Daily |

### Backtester (`backtester/engine.py`)

`BacktestEngine.run(weights, returns)` → `BacktestResult`

Key mechanics: execution lag via `weights.shift(1)`, position clipping, gross exposure cap, daily P&L as `(weights_executed * returns).sum(axis=1)`, linear costs + stamp duty subtracted from returns.

`backtester/walk_forward.py` — expanding-window IS/OOS folds (annual OOS periods from 2018 onward).
`backtester/parameter_sweep.py` — grid search over momentum lookback, skip, and long_pct.

### Factor Research (`factor_research/`)

Standalone statistical validation — no full backtest required:
- `ic.py` — daily rank-IC between signal and forward returns
- `decay.py` — IC persistence at t+1, t+5, t+10, t+20
- `quantile.py` — N-quantile bucket returns and long/short spread

Exposed via the Streamlit page at `pages/factor_research.py`.

### Paper Trading (`paper_trading/`)

Generates today's signals from latest cached/live prices. Portfolio state persisted to `paper_trading/portfolio_state.json`. Exposed via `pages/paper_trading.py`.

### Outputs

After a run, `results/` contains `performance_metrics.csv` and per-strategy PNG dashboards. Logs written to `logs/framework_YYYY-MM-DD.log` at DEBUG level.
