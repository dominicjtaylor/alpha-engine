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
├── app.py                     # Streamlit interactive UI
├── main.py                    # CLI entry point
├── config.py                  # All parameters (DataConfig, StrategyConfig, RiskConfig)
├── requirements.txt
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
│
├── execution/
│   └── transaction_costs.py   # UK costs: commission + slippage + 0.5% SDRT
│
├── analytics/
│   ├── metrics.py             # Sharpe, Sortino, Calmar, VaR, alpha/beta…
│   └── visualizations.py      # Equity curve, drawdown, heatmap (£ labels)
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
