"""
Backtest page — Alpha Engine

Historical backtesting of UK equity strategies with realistic costs.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Backtest — Alpha Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v:.1%}" if not np.isnan(v) else "—"

def _fmt_f2(v: float) -> str:
    return f"{v:.2f}" if not np.isnan(v) else "—"

def _fmt_gbp(v: float) -> str:
    return f"£{v:,.0f}" if not np.isnan(v) else "—"

def _delta_colour(v: float) -> str:
    """Return 'normal', 'inverse', or 'off' for st.metric delta_color."""
    return "normal"


# ---------------------------------------------------------------------------
# Cached data loading (avoids re-downloading on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading price data from Yahoo Finance…", ttl=3600)
def load_data(universe: str, universe_size: int, start: str, end: str):
    """Download and cache price data. Re-runs only when inputs change."""
    from config import DataConfig
    from data.data_loader import DataLoader, clean_data, compute_returns
    from data.universe import load_ftse_universe, apply_liquidity_filters

    data_cfg = DataConfig(
        universe=universe,
        universe_size=universe_size,
        cache_dir="data/cache",
    )
    tickers = load_ftse_universe(index=universe, top_n=universe_size)
    loader = DataLoader(cache_dir="data/cache", config=data_cfg)

    prices_raw = loader.load_price_data(tickers, start, end)
    ohlcv = loader.load_ohlcv(list(prices_raw.columns), start, end)

    prices = clean_data(
        prices_raw,
        min_history_days=data_cfg.min_history_days,
        max_forward_fill_days=data_cfg.max_forward_fill_days,
    )
    prices = prices[apply_liquidity_filters(
        list(prices.columns), prices, min_price_gbp=data_cfg.min_price
    )]
    returns = compute_returns(prices)
    return prices, returns, ohlcv


@st.cache_data(show_spinner="Running backtest…")
def run_backtest(
    universe: str,
    universe_size: int,
    start: str,
    end: str,
    strategies_selected: tuple[str, ...],
    momentum_lookback: int,
    momentum_skip: int,
    momentum_long_pct: float,
    mr_lookback: int,
    mr_long_pct: float,
    gap_threshold: float,
    hold_days: int,
    max_leverage: float,
    max_position_pct: float,
    target_vol: float,
    commission_bps: float,
    slippage_bps: float,
    stamp_duty_rate: float,
    initial_capital: float,
    combination: str,
):
    """Run the full pipeline. Cached by all input parameters."""
    from config import FrameworkConfig, StrategyConfig, RiskConfig
    from data.data_loader import DataLoader, clean_data, compute_returns
    from data.universe import load_ftse_universe, apply_liquidity_filters
    from strategies.momentum import CrossSectionalMomentum
    from strategies.mean_reversion import ShortTermMeanReversion
    from strategies.earnings_revision import EarningsRevisionDrift
    from backtester.engine import BacktestEngine
    from portfolio.portfolio_manager import PortfolioManager, PortfolioConfig, CombinationMethod
    from analytics.metrics import compute_metrics

    # Config
    strat_cfg = StrategyConfig(
        momentum_lookback=momentum_lookback,
        momentum_skip=momentum_skip,
        momentum_long_pct=momentum_long_pct,
        mean_reversion_lookback=mr_lookback,
        mean_reversion_long_pct=mr_long_pct,
        earnings_gap_threshold=gap_threshold,
        earnings_hold_days=hold_days,
    )
    risk_cfg = RiskConfig(
        max_leverage=max_leverage,
        max_position_pct=max_position_pct,
        target_volatility=target_vol,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        stamp_duty_rate=stamp_duty_rate,
    )

    # Data
    from config import DataConfig
    data_cfg = DataConfig(universe=universe, universe_size=universe_size)
    tickers = load_ftse_universe(index=universe, top_n=universe_size)
    loader = DataLoader(cache_dir="data/cache", config=data_cfg)
    prices_raw = loader.load_price_data(tickers, start, end)
    ohlcv = {}
    if "earnings" in strategies_selected:
        ohlcv = loader.load_ohlcv(list(prices_raw.columns), start, end)
    prices = clean_data(prices_raw, data_cfg.min_history_days, data_cfg.max_forward_fill_days)
    prices = prices[apply_liquidity_filters(list(prices.columns), prices,
                                             min_price_gbp=data_cfg.min_price)]
    returns = compute_returns(prices)

    # Strategies
    strategy_map = {
        "momentum": CrossSectionalMomentum(strat_cfg),
        "mean_reversion": ShortTermMeanReversion(strat_cfg),
        "earnings": EarningsRevisionDrift(strat_cfg),
    }
    strategy_results = {}
    for name in strategies_selected:
        kwargs = {"ohlcv": ohlcv} if name == "earnings" else {}
        strategy_results[name] = strategy_map[name].run(prices, returns, **kwargs)

    # Portfolio combination
    engine = BacktestEngine(risk_cfg, initial_capital=initial_capital)
    combo_method = CombinationMethod(combination)
    port_cfg = PortfolioConfig(combination_method=combo_method)
    pm = PortfolioManager(port_cfg, risk_cfg)

    backtest_results = {}
    for name, sr in strategy_results.items():
        bt = engine.run(sr.weights, returns, strategy_name=name)
        backtest_results[name] = bt

    # Combined portfolio (if more than one strategy)
    if len(strategy_results) > 1:
        combined_weights = pm.combine_strategies(strategy_results)
        combined_weights = pm.apply_risk_controls(combined_weights, returns)
        bt_combined = engine.run(combined_weights, returns, strategy_name="Combined")
        backtest_results["Combined"] = bt_combined

    all_metrics = {
        name: compute_metrics(bt.daily_returns)
        for name, bt in backtest_results.items()
    }

    return backtest_results, all_metrics, prices, returns


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    st.caption("UK Equity Research Framework")

    st.subheader("Universe")
    universe = st.selectbox(
        "Index",
        ["ftse_all", "ftse100", "ftse250"],
        format_func=lambda x: {"ftse_all": "FTSE All (~350)", "ftse100": "FTSE 100", "ftse250": "FTSE 250"}[x],
    )
    universe_size = st.slider("Max tickers", 50, 350, 200, step=50)

    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    start_date = col1.text_input("Start", "2015-01-01")
    end_date = col2.text_input("End", "2023-12-31")

    st.subheader("Strategies")
    run_momentum = st.checkbox("Momentum (12-1)", value=True)
    run_mean_rev = st.checkbox("Mean Reversion (5-day)", value=True)
    run_earnings = st.checkbox("Earnings Drift (gap)", value=False,
                               help="Requires OHLCV download — slower first run")

    strategies_selected = tuple([
        s for s, flag in [("momentum", run_momentum), ("mean_reversion", run_mean_rev),
                          ("earnings", run_earnings)] if flag
    ])

    st.subheader("Strategy Parameters")
    with st.expander("Momentum", expanded=False):
        mom_lookback = st.slider("Lookback (days)", 63, 504, 252, step=21)
        mom_skip = st.slider("Skip (days)", 0, 63, 21, step=5)
        mom_long_pct = st.slider("Decile size", 0.05, 0.30, 0.10, step=0.05,
                                  format="%.0f%%",
                                  help="Fraction of universe per long/short book")

    with st.expander("Mean Reversion", expanded=False):
        mr_lookback = st.slider("Lookback (days) ", 3, 21, 5, step=1)
        mr_long_pct = st.slider("Quintile size", 0.10, 0.40, 0.20, step=0.05,
                                 format="%.0f%%")

    with st.expander("Earnings Drift", expanded=False):
        gap_threshold = st.slider("Gap threshold", 0.02, 0.15, 0.05, step=0.01,
                                   format="%.0f%%")
        hold_days = st.slider("Hold days", 3, 30, 10, step=1)

    st.subheader("Risk & Costs")
    initial_capital = st.number_input("Initial capital (£)", value=1_000_000,
                                       step=100_000, format="%d")
    max_leverage = st.slider("Max leverage", 1.0, 3.0, 1.5, step=0.1)
    max_position_pct = st.slider("Max position size", 0.01, 0.10, 0.05, step=0.01,
                                  format="%.0f%%")
    target_vol = st.slider("Vol target (ann.)", 0.05, 0.30, 0.10, step=0.01,
                             format="%.0f%%")

    with st.expander("Transaction Costs", expanded=False):
        commission_bps = st.slider("Commission (bps/side)", 0, 30, 10)
        slippage_bps = st.slider("Slippage (bps/side)", 0, 30, 10)
        stamp_duty_pct = st.slider("Stamp duty (%)", 0.0, 1.0, 0.5, step=0.1,
                                    help="UK SDRT on long purchases") / 100

    combination = st.selectbox("Combination method",
                                ["equal_weight", "volatility_scale"],
                                format_func=lambda x: x.replace("_", " ").title())

    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📈 Alpha Engine")
st.caption("UK Systematic Trading Research Framework — LSE Equities — GBP")

if not strategies_selected:
    st.warning("Select at least one strategy in the sidebar.")
    st.stop()

if not run_btn and "backtest_results" not in st.session_state:
    st.info(
        "Configure your backtest in the sidebar and click **▶ Run Backtest** to start.\n\n"
        "Data is cached locally after the first download — subsequent runs are instant."
    )
    st.stop()

# Run or use cached results
if run_btn:
    with st.spinner("Running pipeline…"):
        backtest_results, all_metrics, prices, returns = run_backtest(
            universe, universe_size, start_date, end_date,
            strategies_selected,
            mom_lookback, mom_skip, mom_long_pct,
            mr_lookback, mr_long_pct,
            gap_threshold, hold_days,
            max_leverage, max_position_pct, target_vol,
            float(commission_bps), float(slippage_bps), stamp_duty_pct,
            float(initial_capital), combination,
        )
    st.session_state["backtest_results"] = backtest_results
    st.session_state["all_metrics"] = all_metrics
    st.session_state["prices"] = prices
    st.session_state["returns"] = returns
    st.success("Backtest complete.")

backtest_results = st.session_state["backtest_results"]
all_metrics = st.session_state["all_metrics"]
prices = st.session_state["prices"]
returns = st.session_state["returns"]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_summary, tab_equity, tab_drawdown, tab_monthly, tab_data = st.tabs([
    "📊 Summary", "📈 Equity Curves", "📉 Drawdowns", "🗓 Monthly Returns", "🔍 Data"
])


# ── Tab 1: Summary metrics ──────────────────────────────────────────────────
with tab_summary:
    st.subheader("Performance Summary (GBP)")

    strategy_names = list(backtest_results.keys())

    # KPI cards for primary strategy (or Combined if available)
    primary = "Combined" if "Combined" in backtest_results else strategy_names[0]
    m = all_metrics[primary]
    bt = backtest_results[primary]

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Annual Return", _fmt_pct(m["annual_return"]))
    kpi2.metric("Sharpe Ratio", _fmt_f2(m["sharpe_ratio"]))
    kpi3.metric("Max Drawdown", _fmt_pct(m["max_drawdown"]))
    kpi4.metric("Ann. Volatility", _fmt_pct(m["annual_volatility"]))
    kpi5.metric("Final Value",
                _fmt_gbp(bt.equity_curve.iloc[-1]),
                delta=_fmt_pct(bt.equity_curve.iloc[-1] / float(initial_capital) - 1))

    st.caption(f"Strategy shown: **{primary}** | Capital: £{initial_capital:,} | "
               f"Universe: {universe.upper()} | {start_date} → {end_date}")

    st.divider()

    # Full metrics table
    display_keys = [
        ("annual_return", "Annual Return"),
        ("annual_volatility", "Volatility (Ann.)"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("max_drawdown", "Max Drawdown"),
        ("calmar_ratio", "Calmar Ratio"),
        ("win_rate", "Win Rate"),
        ("profit_factor", "Profit Factor"),
        ("var_95", "VaR (95%)"),
        ("cvar_95", "CVaR (95%)"),
        ("n_days", "Trading Days"),
    ]

    pct_keys = {"annual_return", "annual_volatility", "max_drawdown", "win_rate",
                "var_95", "cvar_95"}

    rows = []
    for key, label in display_keys:
        row = {"Metric": label}
        for name in strategy_names:
            v = all_metrics[name].get(key, float("nan"))
            if key in pct_keys:
                row[name] = _fmt_pct(v)
            elif key == "n_days":
                row[name] = f"{int(v):,}"
            else:
                row[name] = _fmt_f2(v)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)

    # Cost breakdown
    st.subheader("Cost Breakdown")
    cost_rows = []
    for name, bt in backtest_results.items():
        ann_days = len(bt.daily_returns)
        cost_rows.append({
            "Strategy": name,
            "Total linear cost (ann.)": _fmt_pct(bt.linear_costs.sum() * 252 / max(ann_days, 1)),
            "Stamp duty (ann.)": _fmt_pct(bt.stamp_duty_costs.sum() * 252 / max(ann_days, 1)),
            "Avg daily turnover": _fmt_pct(bt.turnover.mean()),
            "Avg gross exposure": f"{bt.gross_exposure.mean():.2f}×",
        })
    st.dataframe(pd.DataFrame(cost_rows).set_index("Strategy"), use_container_width=True)


# ── Tab 2: Equity Curves ────────────────────────────────────────────────────
with tab_equity:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    st.subheader("Equity Curves — Normalised (base = 1.0)")

    fig, ax = plt.subplots(figsize=(13, 5))
    colours = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (name, bt) in enumerate(backtest_results.items()):
        normalised = bt.equity_curve / bt.equity_curve.iloc[0]
        final = normalised.iloc[-1]
        ax.plot(normalised.index, normalised.values,
                label=f"{name}  ({final:.2f}×)",
                color=colours[i % len(colours)], linewidth=1.8)

    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Portfolio Value (normalised)")
    ax.set_title("Strategy Equity Curves — UK Equities (GBP)", fontsize=12)
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Rolling 63-Day Sharpe Ratio")

    from analytics.metrics import compute_rolling_sharpe

    fig2, ax2 = plt.subplots(figsize=(13, 4))
    for i, (name, bt) in enumerate(backtest_results.items()):
        rs = compute_rolling_sharpe(bt.daily_returns, window=63)
        ax2.plot(rs.index, rs.values, label=name,
                 color=colours[i % len(colours)], linewidth=1.4)
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax2.axhline(1, color="green", linewidth=0.5, linestyle=":", alpha=0.6)
    ax2.set_ylabel("Rolling Sharpe (63-day)")
    ax2.set_title("Rolling Sharpe Ratio", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.tick_params(axis="x", rotation=20)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ── Tab 3: Drawdowns ────────────────────────────────────────────────────────
with tab_drawdown:
    st.subheader("Drawdown from Peak")

    fig, ax = plt.subplots(figsize=(13, 5))
    colours = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (name, bt) in enumerate(backtest_results.items()):
        dd_pct = bt.drawdowns * 100
        colour = colours[i % len(colours)]
        ax.fill_between(dd_pct.index, dd_pct.values, 0, alpha=0.2, color=colour)
        ax.plot(dd_pct.index, dd_pct.values, color=colour, linewidth=1.2,
                label=f"{name}  (max: {dd_pct.min():.1f}%)")

    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Portfolio Drawdowns — UK Equities", fontsize=12)
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Drawdown stats table
    dd_rows = []
    for name, bt in backtest_results.items():
        dd = bt.drawdowns
        dd_rows.append({
            "Strategy": name,
            "Max Drawdown": _fmt_pct(dd.min()),
            "Avg Drawdown": _fmt_pct(dd.mean()),
            "Days in Drawdown": f"{(dd < -0.01).sum():,}",
            "Current Drawdown": _fmt_pct(dd.iloc[-1]),
        })
    st.dataframe(pd.DataFrame(dd_rows).set_index("Strategy"), use_container_width=True)


# ── Tab 4: Monthly Returns ───────────────────────────────────────────────────
with tab_monthly:
    import seaborn as sns
    from analytics.metrics import compute_monthly_returns

    strategy_choice = st.selectbox("Strategy", list(backtest_results.keys()),
                                    key="monthly_strat")
    bt = backtest_results[strategy_choice]

    try:
        monthly = compute_monthly_returns(bt.daily_returns)
        month_labels = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        monthly.columns = [month_labels.get(c, str(c)) for c in monthly.columns]

        fig, ax = plt.subplots(figsize=(13, max(4, len(monthly) * 0.55)))
        sns.heatmap(
            monthly,
            ax=ax,
            cmap="RdYlGn",
            center=0,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 8},
            linewidths=0.4,
            cbar_kws={"label": "Monthly Return (%)"},
        )
        ax.set_title(f"Monthly Returns (%) — {strategy_choice}", fontsize=12)
        ax.set_ylabel("Year")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not render heatmap: {e}")

    # Annual summary
    st.subheader("Annual Returns")
    try:
        annual = (1 + bt.daily_returns).resample("YE").prod() - 1
        annual_df = annual.to_frame("Return")
        annual_df.index = annual_df.index.year
        annual_df.index.name = "Year"
        annual_df["Return"] = annual_df["Return"].map(_fmt_pct)
        st.dataframe(annual_df.T, use_container_width=True)
    except Exception:
        pass


# ── Tab 5: Data Explorer ────────────────────────────────────────────────────
with tab_data:
    st.subheader("Universe & Price Data")

    c1, c2, c3 = st.columns(3)
    c1.metric("Tickers in universe", len(prices.columns))
    c2.metric("Trading days", len(prices))
    c3.metric("Date range", f"{prices.index[0].date()} → {prices.index[-1].date()}")

    st.subheader("Sample Prices (last 10 rows, first 15 tickers)")
    st.dataframe(
        prices.iloc[-10:, :15].round(2),
        use_container_width=True,
    )

    st.subheader("Position Weights")
    strategy_choice_data = st.selectbox("Strategy ", list(backtest_results.keys()),
                                         key="data_strat")
    bt_data = backtest_results[strategy_choice_data]

    st.caption("Last 5 trading days — top 20 non-zero positions")
    last_weights = bt_data.positions.iloc[-5:]
    # Show only columns with recent non-zero weights
    active_cols = last_weights.columns[(last_weights.abs() > 0.001).any()][:20]
    if len(active_cols):
        st.dataframe(last_weights[active_cols].round(4), use_container_width=True)
    else:
        st.info("No active positions in the last 5 days.")

    # Download buttons
    st.subheader("Export")
    col_dl1, col_dl2 = st.columns(2)

    from analytics.metrics import compute_metrics
    metrics_df = pd.DataFrame(all_metrics).T
    col_dl1.download_button(
        "Download Metrics (CSV)",
        data=metrics_df.to_csv().encode(),
        file_name="performance_metrics.csv",
        mime="text/csv",
    )

    returns_export = pd.DataFrame({
        name: bt.daily_returns for name, bt in backtest_results.items()
    })
    col_dl2.download_button(
        "Download Daily Returns (CSV)",
        data=returns_export.to_csv().encode(),
        file_name="daily_returns.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Data: Yahoo Finance (LSE .L tickers) · Costs: 25 bps/side + 0.5% SDRT on longs · "
    "All performance in GBP · Past performance is not indicative of future results."
)
