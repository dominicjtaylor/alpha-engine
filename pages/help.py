"""
Help & Documentation page — Alpha Engine
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.markdown('<h1 class="ae-title"><i class="fa-solid fa-circle-question ae-icon"></i>Help & Documentation</h1>', unsafe_allow_html=True)
st.markdown("Reference guide for the Alpha Engine systematic trading research framework.")

st.divider()

# ---------------------------------------------------------------------------
# Pages overview
# ---------------------------------------------------------------------------

st.markdown('<h2 class="ae-header"><i class="fa-solid fa-layer-group ae-icon"></i>Pages</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Factor Research
    Evaluate alpha signals as quantitative factors before committing to a backtest.

    - **IC (Information Coefficient)** — daily Spearman/Pearson rank correlation between the factor signal and next-period stock returns
    - **Rolling IC** — rolling window IC to detect regime dependency
    - **Factor Decay** — IC at multiple forward horizons (1, 5, 10, 20 days) to guide rebalancing frequency
    - **Quantile Portfolios** — split the universe into N quantiles by factor score; check for monotonic return separation

    **Use this page to:** validate a new factor has genuine predictive power before running a full backtest.

    ---

    #### Backtest
    Full historical simulation of a long-short strategy with realistic transaction costs.

    - UK transaction costs: commission + slippage + **0.5% SDRT** on long purchases
    - Daily rebalancing with configurable lookback and portfolio size
    - Benchmark comparison vs. a Vanguard-style passive global ETF portfolio
    - Walk-forward performance, drawdown analysis, and monthly return heatmap
    """)

with col2:
    st.markdown("""
    #### Paper Trading
    Simulate live strategy execution with virtual capital.

    - Downloads latest LSE prices and generates today's signals
    - Previews the rebalance trade list before committing
    - Tracks cumulative paper P&L and position history in `paper_trading/portfolio_state.json`
    """)

st.divider()

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

st.markdown('<h2 class="ae-header"><i class="fa-solid fa-layer-group ae-icon"></i>Strategies</h2>', unsafe_allow_html=True)

st.markdown("""
| Strategy | Signal | Rebalance Frequency | Key Parameters |
|---|---|---|---|
| **Momentum (12-1)** | Cumulative return, t-252 to t-21 (skip last month) | Monthly | `lookback=252`, `skip=21`, `long_pct=0.10` |
| **Mean Reversion (5-day)** | Negated 5-day return (recent losers bounce) | Daily | `lookback=5`, `long_pct=0.20` |
| **Earnings Revision Drift** | Overnight gap > threshold (PEAD proxy) | Event-driven | `gap_threshold=0.05`, `hold_days=10` |
| **Factor Portfolio** | Composite of N weighted factors | Per-factor | `factor_names`, `factor_weights`, `long_pct` |

All strategies are **dollar-neutral long-short**: the long book and short book each carry equal gross exposure, so net market exposure is approximately zero.
""")

st.divider()

# ---------------------------------------------------------------------------
# Factor Registry
# ---------------------------------------------------------------------------

st.markdown('<h2 class="ae-header"><i class="fa-solid fa-brain ae-icon"></i>Factor Registry</h2>', unsafe_allow_html=True)

try:
    from factors import list_factors, factor_metadata
    meta = factor_metadata()
    factors = list_factors()

    rows = []
    for f in factors:
        m = meta.get(f, {})
        rows.append({
            "Factor Name": f"`{f}`",
            "Description": m.get("description", "—"),
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(rows).set_index("Factor Name"), width="stretch")
except Exception as e:
    st.warning(f"Could not load factor registry: {e}")

st.markdown("""
Factors are registered via the `@register` decorator in `factors/`. Any new factor placed in
`factors/` and imported in `factors/__init__.py` becomes automatically available in the
Factor Research page dropdown and the `FactorPortfolioStrategy`.
""")

st.divider()

# ---------------------------------------------------------------------------
# IC Interpretation Guide
# ---------------------------------------------------------------------------

st.markdown('<h2 class="ae-header"><i class="fa-solid fa-chart-line ae-icon"></i>IC Interpretation Guide</h2>', unsafe_allow_html=True)

st.markdown("""
| Metric | Rule of thumb |
|---|---|
| **IC Mean** | > 0.03 = weak; > 0.05 = useful; > 0.10 = strong (rare for equity factors) |
| **IC t-statistic** | > ±2.0 = statistically significant at ~95% confidence |
| **IC IR (annualised)** | IC Mean / IC Std × √252; > 0.5 is a good factor |
| **% Positive IC days** | > 52% suggests a persistent positive edge |
| **Monotonicity Score** | Spearman ρ of (quantile label vs mean return); 1.0 = perfect; < 0 = inverted |
| **L/S Spread** | Annualised return of long top quantile / short bottom quantile (pre-cost) |

**Factor decay guidance:**
- IC near-zero by t+5: suitable for **daily** rebalancing strategies only
- IC persisting through t+20: suitable for **monthly** rebalancing
""")

st.divider()

# ---------------------------------------------------------------------------
# Transaction Cost Model
# ---------------------------------------------------------------------------

st.markdown('<h2 class="ae-header"><i class="fa-solid fa-coins ae-icon"></i>UK Transaction Cost Model</h2>', unsafe_allow_html=True)

st.markdown("""
Each trade incurs costs on both the buy and sell side:

| Cost | Rate | Applied to |
|---|---|---|
| Commission | 10 bps | Both sides |
| Slippage / Market impact | 10 bps | Both sides |
| Bid-ask spread | 5 bps | Both sides |
| **Stamp Duty Reserve Tax (SDRT)** | **50 bps (0.5%)** | **Long purchases only** |

Total round-trip cost (long purchase + later sale):
- **Long side:** ~75 bps buy + ~25 bps sell = **~100 bps round-trip**
- **Short side:** ~25 bps entry + ~25 bps cover = **~50 bps round-trip**

High-frequency strategies (e.g. daily mean reversion) are particularly sensitive to these costs.
""")

st.divider()

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

st.markdown('<h2 class="ae-header"><i class="fa-solid fa-database ae-icon"></i>Data Pipeline</h2>', unsafe_allow_html=True)

st.markdown("""
1. **Universe** — FTSE 100 / FTSE 250 / combined tickers with `.L` suffix (Yahoo Finance LSE convention)
2. **Download** — `yfinance` downloads adjusted close prices and OHLCV data
3. **Cache** — prices cached to `data/cache/<ticker>.parquet` (Snappy-compressed); incremental updates on re-run
4. **Clean** — forward-fill gaps (max 5 days), drop tickers with insufficient history (< 252 trading days), apply liquidity filter (min £1M avg daily volume)
5. **Returns** — daily simple returns computed from adjusted close prices

The data loader validates cache coverage on both ends (start and end date). Missing historical or recent data is downloaded incrementally without full re-fetch.
""")

st.divider()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.markdown('<h2 class="ae-header"><i class="fa-solid fa-gear ae-icon"></i>Configuration</h2>', unsafe_allow_html=True)

st.markdown("""
All parameters are defined in `config.py` as Python dataclasses:

- `DataConfig` — universe, cache path, liquidity thresholds
- `StrategyConfig` — per-strategy signal and portfolio parameters
- `RiskConfig` — leverage limits, volatility targets, transaction costs
- `FrameworkConfig` — root config combining all sub-configs

No environment variables or external config files are required. Override parameters by constructing config objects directly in code or via the Streamlit UI sliders.
""")

st.divider()

# ---------------------------------------------------------------------------
# Benchmark reference
# ---------------------------------------------------------------------------

st.markdown('<h2 class="ae-header"><i class="fa-solid fa-scale-balanced ae-icon"></i>Passive Benchmark</h2>', unsafe_allow_html=True)

with st.expander("Vanguard-style Global Portfolio — details & live stats", expanded=False):
    st.markdown("""
    All strategies are measured against a **buy-and-hold passive benchmark**
    approximating a Vanguard global allocation. Annual rebalancing.

    | ETF | Allocation | Exposure |
    |-----|-----------|---------|
    | VTI | 58.89% | US Total Market |
    | VGK | 13.22% | Developed Europe |
    | BNDW | 10.00% | Global Aggregate Bonds |
    | VWO | 9.17% | Emerging Markets |
    | EWJ | 5.12% | Japan |
    | VPL | 3.60% | Asia Pacific ex-Japan |

    > **Note:** Benchmark returns are USD-denominated total returns.
    > Use relative metrics (alpha, tracking error, information ratio) for comparison.
    """)

    import pandas as pd

    @st.cache_data(show_spinner="Loading benchmark…", ttl=3600)
    def _load_benchmark_preview():
        from data.benchmark import load_benchmark_returns
        from analytics.metrics import annual_return, annual_volatility, sharpe_ratio, max_drawdown
        returns = load_benchmark_returns(start_date="2015-01-01")
        if returns.empty:
            return None, {}
        equity = (1 + returns).cumprod()
        metrics = {
            "CAGR": f"{annual_return(returns):.1%}",
            "Volatility": f"{annual_volatility(returns):.1%}",
            "Sharpe": f"{sharpe_ratio(returns):.2f}",
            "Max Drawdown": f"{max_drawdown(returns):.1%}",
            "Period": f"{returns.index[0].date()} → {returns.index[-1].date()}",
        }
        return equity, metrics

    try:
        bench_equity, bench_metrics = _load_benchmark_preview()
        if bench_equity is not None:
            m_cols = st.columns(len(bench_metrics))
            for col, (k, v) in zip(m_cols, bench_metrics.items()):
                col.metric(k, v)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(bench_equity.index, bench_equity.values,
                    color="#607D8B", linewidth=1.5, label="Vanguard-style benchmark")
            ax.set_title("Benchmark Growth of $1 (USD, 2015–present)", fontsize=10)
            ax.set_ylabel("Cumulative return")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.info(f"Benchmark data unavailable: {e}")
