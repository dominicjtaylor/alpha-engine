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

INDEX_DISPLAY_NAMES = {
    "ftse100": "FTSE 100",
    "ftse250": "FTSE 250",
    "ftse_all": "FTSE All-Share",
}


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
    fp_factor_names: tuple[str, ...] = (),
    fp_factor_long_pct: float = 0.10,
    force_refresh: bool = False,
):
    """Run the full pipeline. Cached by all input parameters."""
    from config import FrameworkConfig, StrategyConfig, RiskConfig
    from data.data_loader import DataLoader, clean_data, compute_returns
    from data.universe import load_ftse_universe, apply_liquidity_filters
    from strategies.momentum import CrossSectionalMomentum
    from strategies.mean_reversion import ShortTermMeanReversion
    from strategies.earnings_revision import EarningsRevisionDrift
    from strategies.factor_portfolio import FactorPortfolioStrategy
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
    prices_raw = loader.load_price_data(tickers, start, end, force_refresh=force_refresh)
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
    if "factor_portfolio" in strategies_selected and fp_factor_names:
        strategy_map["factor_portfolio"] = FactorPortfolioStrategy(
            strat_cfg,
            factor_names=list(fp_factor_names),
            long_pct=fp_factor_long_pct,
        )
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

    # Benchmark
    benchmark_returns = None
    try:
        from data.benchmark import load_benchmark_returns
        benchmark_returns = load_benchmark_returns(start_date=start, end_date=end)
    except Exception:
        pass

    all_metrics = {
        name: compute_metrics(bt.daily_returns, benchmark_returns=benchmark_returns)
        for name, bt in backtest_results.items()
    }

    return backtest_results, all_metrics, prices, returns, benchmark_returns


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Risk profile definitions
# ---------------------------------------------------------------------------

_RISK_PROFILES = {
    "Conservative": {"target_vol": 0.06, "max_leverage": 1.0, "max_position_pct": 0.03},
    "Balanced":     {"target_vol": 0.10, "max_leverage": 1.5, "max_position_pct": 0.05},
    "Aggressive":   {"target_vol": 0.15, "max_leverage": 2.0, "max_position_pct": 0.08},
}

# ---------------------------------------------------------------------------
# Sidebar — detect leaderboard pre-population BEFORE widgets render
# ---------------------------------------------------------------------------

_from_leaderboard = st.session_state.pop("factor_from_leaderboard", None)
_lb_factor_name = _from_leaderboard["factor_name"] if _from_leaderboard else None
_lb_factor_kwargs = _from_leaderboard["factor_kwargs"] if _from_leaderboard else {}

with st.sidebar:
    st.markdown('<p style="font-size:1.1rem;font-weight:700;margin:0 0 2px"><i class="fa-solid fa-gear ae-icon"></i>Configuration</p>', unsafe_allow_html=True)
    st.caption("UK Equity Research Framework")

    if _from_leaderboard:
        st.success(f"Pre-loaded from Leaderboard: **{_lb_factor_name}**")

    st.markdown('<p style="font-size:0.95rem;font-weight:600;margin:0.8rem 0 0.2rem"><i class="fa-solid fa-globe ae-icon"></i>Universe</p>', unsafe_allow_html=True)
    universe = st.selectbox(
        "Index",
        ["ftse_all", "ftse100", "ftse250"],
        format_func=lambda x: INDEX_DISPLAY_NAMES.get(x, x),
    )
    universe_size = st.slider("Max tickers", 50, 350, 200, step=50)

    st.markdown('<p style="font-size:0.95rem;font-weight:600;margin:0.8rem 0 0.2rem"><i class="fa-solid fa-calendar ae-icon"></i>Date Range</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    start_date = col1.text_input("Start", "2015-01-01")
    end_date = col2.text_input("End", "2023-12-31")

    st.markdown('<p style="font-size:0.95rem;font-weight:600;margin:0.8rem 0 0.2rem"><i class="fa-solid fa-layer-group ae-icon"></i>Strategies</p>', unsafe_allow_html=True)
    run_momentum = st.checkbox("Momentum (12-1)", value=True)
    run_mean_rev = st.checkbox("Mean Reversion (5-day)", value=True)
    run_earnings = st.checkbox("Earnings Drift (gap)", value=False,
                               help="Requires OHLCV download — slower first run")
    # Pre-enable Factor Portfolio if coming from leaderboard
    run_factor_portfolio = st.checkbox(
        "Factor Portfolio (multi-factor)",
        value=bool(_from_leaderboard),
    )

    strategies_selected = tuple([
        s for s, flag in [("momentum", run_momentum), ("mean_reversion", run_mean_rev),
                          ("earnings", run_earnings),
                          ("factor_portfolio", run_factor_portfolio)] if flag
    ])

    st.markdown('<p style="font-size:0.95rem;font-weight:600;margin:0.8rem 0 0.2rem"><i class="fa-solid fa-sliders ae-icon"></i>Strategy Parameters</p>', unsafe_allow_html=True)
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

    from factors import list_factors, factor_metadata as _factor_meta
    _available_factors = list_factors()
    _factor_descriptions = _factor_meta()

    with st.expander("Factor Portfolio", expanded=bool(_from_leaderboard)):
        # Pre-select leaderboard factor if available
        _fp_default = (
            [_lb_factor_name] if _lb_factor_name and _lb_factor_name in _available_factors
            else (_available_factors[:2] if len(_available_factors) >= 2 else _available_factors)
        )
        fp_selected = st.multiselect(
            "Factors to combine",
            _available_factors,
            default=_fp_default,
            format_func=lambda x: _factor_descriptions.get(x, {}).get("description", x),
        )
        fp_long_pct = st.slider("Long/short decile size ", 0.05, 0.30, 0.10, step=0.05,
                                 format="%.0f%%",
                                 help="Fraction of universe per long/short book")

    fp_factor_names = tuple(fp_selected) if run_factor_portfolio else ()
    fp_factor_long_pct = fp_long_pct if run_factor_portfolio else 0.10

    # -----------------------------------------------------------------------
    # Risk Profile
    # -----------------------------------------------------------------------
    st.markdown('<p style="font-size:0.95rem;font-weight:600;margin:0.8rem 0 0.2rem"><i class="fa-solid fa-shield-halved ae-icon"></i>Risk Profile</p>', unsafe_allow_html=True)
    selected_profile = st.radio(
        "Profile",
        list(_RISK_PROFILES.keys()),
        index=1,
        horizontal=True,
        help="Conservative: low vol, tight limits. Balanced: default. Aggressive: higher leverage/vol target.",
    )
    _profile = _RISK_PROFILES[selected_profile]

    initial_capital = st.number_input("Initial capital (£)", value=1_000_000,
                                       step=100_000, format="%d")

    with st.expander("Override risk limits", expanded=False):
        max_leverage = st.slider(
            "Max leverage",
            1.0, 3.0,
            float(_profile["max_leverage"]),
            step=0.1,
            key=f"lev_{selected_profile}",
        )
        max_position_pct = st.slider(
            "Max position size",
            0.01, 0.10,
            float(_profile["max_position_pct"]),
            step=0.01,
            format="%.0f%%",
            key=f"pos_{selected_profile}",
        )
        target_vol = st.slider(
            "Vol target (ann.)",
            0.05, 0.30,
            float(_profile["target_vol"]),
            step=0.01,
            format="%.0f%%",
            key=f"vol_{selected_profile}",
        )

    with st.expander("Transaction Costs", expanded=False):
        commission_bps = st.slider("Commission (bps/side)", 0, 30, 10)
        slippage_bps = st.slider("Slippage (bps/side)", 0, 30, 10)
        stamp_duty_pct = st.slider("Stamp duty (%)", 0.0, 1.0, 0.5, step=0.1,
                                    help="UK SDRT on long purchases") / 100

    combination = st.selectbox("Combination method",
                                ["equal_weight", "volatility_scale"],
                                format_func=lambda x: x.replace("_", " ").title())

    force_refresh = st.checkbox(
        "Force refresh data",
        value=False,
        help="Delete cached price data and re-download from Yahoo Finance",
    )

    run_btn = st.button("Run Backtest", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown('<h1 class="ae-title"><i class="fa-solid fa-chart-column ae-icon"></i>Backtest</h1>', unsafe_allow_html=True)
st.caption(
    "Simulate trading the strategy through history with realistic UK costs — LSE Equities — GBP · "
    "Benchmark: FTSE All-Share (passive buy-and-hold, no costs)"
)

# Leaderboard banner — show if factors are in the leaderboard
try:
    from data.factor_research_log import load_leaderboard as _load_lb
    _lb = _load_lb()
    if not _lb.empty:
        _sig = int((_lb["ic_tstat"].abs() >= 2.0).sum())
        st.info(
            f"**{len(_lb)} factors** saved to the leaderboard ({_sig} statistically significant). "
            f"[View Leaderboard →](Leaderboard)"
        )
except Exception:
    pass

with st.expander("What this page does — click to read", expanded=False):
    st.markdown("""
    ### From signal to strategy

    Factor Research tests whether a signal *predicts* returns.
    This page tests what happens when you **actually trade on it**.

    ```
    Factor scores (today)
    ↓
    Rank all stocks → buy top decile, short bottom decile
    ↓
    Hold for one rebalance period
    ↓
    Pay transaction costs (commission + stamp duty)
    ↓
    Measure portfolio return
    ↓
    Repeat daily / monthly for the full date range
    ```

    ---

    ### What the backtest includes

    | Component | Detail |
    |---|---|
    | **Execution lag** | Signals generated at close[t] execute at open[t+1] — no look-ahead |
    | **Transaction costs** | Commission + slippage (~25 bps/side) + 0.5% Stamp Duty on UK buys |
    | **Portfolio constraints** | Max position size, leverage cap, volatility targeting |
    | **Dollar-neutral** | Long book ≈ Short book — strategy profits from *relative* performance, not market direction |

    ---

    ### Why a signal with good IC can still fail

    A factor can show a strong Information Coefficient in Factor Research
    but still produce poor backtest results because:

    - **High turnover** — daily rebalancing triggers too many round-trips, costs exceed the edge
    - **Small spreads** — the factor predicts, but only by tiny amounts that costs eat away
    - **Crowding / impact** — in reality, large trades move prices against you

    If the factor is strong but backtest results are weak, try:
    increasing the rebalance period, reducing portfolio size, or combining with other factors.

    ---

    ### How to read the results

    | Metric | What it means | Target |
    |---|---|---|
    | **Annual Return** | Average yearly profit as % of capital | > 0% after costs |
    | **Sharpe Ratio** | Return per unit of risk (higher = better) | > 0.5 good, > 1.0 strong |
    | **Max Drawdown** | Worst peak-to-trough loss | Depends on risk tolerance |
    | **Alpha vs Benchmark** | Return above the passive benchmark | > 0% means added value |
    | **Calmar Ratio** | Annual return ÷ Max drawdown | > 0.5 suggests manageable risk |

    ---

    ### Typical workflow
    ```
    1. Select strategies and date range in the sidebar
    2. Adjust parameters (lookback, decile size, costs)
    3. Click ▶ Run Backtest
    4. Check Summary tab for risk-adjusted returns
    5. Check Equity Curves to see performance vs benchmark
    6. Check Monthly Returns for seasonal patterns
    ```
    """)

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
        backtest_results, all_metrics, prices, returns, benchmark_returns = run_backtest(
            universe, universe_size, start_date, end_date,
            strategies_selected,
            mom_lookback, mom_skip, mom_long_pct,
            mr_lookback, mr_long_pct,
            gap_threshold, hold_days,
            max_leverage, max_position_pct, target_vol,
            float(commission_bps), float(slippage_bps), stamp_duty_pct,
            float(initial_capital), combination,
            fp_factor_names, fp_factor_long_pct,
            force_refresh,
        )
    st.session_state["backtest_results"] = backtest_results
    st.session_state["all_metrics"] = all_metrics
    st.session_state["prices"] = prices
    st.session_state["returns"] = returns
    st.session_state["benchmark_returns"] = benchmark_returns
    st.success("Backtest complete.")

backtest_results = st.session_state["backtest_results"]
all_metrics = st.session_state["all_metrics"]
prices = st.session_state["prices"]
returns = st.session_state["returns"]
benchmark_returns = st.session_state.get("benchmark_returns", None)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_summary, tab_equity, tab_drawdown, tab_monthly, tab_data = st.tabs([
    "Summary", "Equity Curves", "Drawdowns", "Monthly Returns", "Data"
])


# ── Tab 1: Summary metrics ──────────────────────────────────────────────────
with tab_summary:
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-gauge-high ae-icon"></i>Performance Summary (GBP)</h3>', unsafe_allow_html=True)

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
               f"Universe: {INDEX_DISPLAY_NAMES.get(universe, universe.upper())} | {start_date} → {end_date}")

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
    has_bench = benchmark_returns is not None and any(
        "benchmark_cagr" in m for m in all_metrics.values()
    )
    if has_bench:
        display_keys += [
            ("benchmark_cagr", "Benchmark CAGR"),
            ("benchmark_sharpe", "Benchmark Sharpe"),
            ("benchmark_max_drawdown", "Benchmark Max DD"),
            ("alpha", "Alpha vs Benchmark"),
            ("tracking_error", "Tracking Error"),
            ("information_ratio", "Information Ratio"),
        ]

    pct_keys = {"annual_return", "annual_volatility", "max_drawdown", "win_rate",
                "var_95", "cvar_95", "benchmark_cagr", "benchmark_volatility",
                "benchmark_max_drawdown", "alpha", "tracking_error"}

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

    st.dataframe(pd.DataFrame(rows).set_index("Metric"), width="stretch")

    # Cost breakdown
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-coins ae-icon"></i>Cost Breakdown</h3>', unsafe_allow_html=True)
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
    st.dataframe(pd.DataFrame(cost_rows).set_index("Strategy"), width="stretch")


# ── Tab 2: Equity Curves ────────────────────────────────────────────────────
with tab_equity:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-chart-line ae-icon"></i>Equity Curves — Normalised (base = 1.0)</h3>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    colours = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (name, bt) in enumerate(backtest_results.items()):
        normalised = bt.equity_curve / bt.equity_curve.iloc[0]
        final = normalised.iloc[-1]
        ax.plot(normalised.index, normalised.values,
                label=f"{name}  ({final:.2f}×)",
                color=colours[i % len(colours)], linewidth=1.8)

    if benchmark_returns is not None:
        from data.benchmark import get_benchmark_equity_curve
        bench_equity = get_benchmark_equity_curve(benchmark_returns)
        bench_norm = bench_equity / bench_equity.iloc[0]
        ax.plot(bench_norm.index, bench_norm.values,
                label=f"Benchmark  ({bench_norm.iloc[-1]:.2f}×)",
                color="grey", linewidth=1.4, linestyle="--", alpha=0.8)

    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Portfolio Value (normalised)")
    ax.set_title("Strategy Equity Curves vs Benchmark — UK Equities (GBP)", fontsize=12)
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
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-wave-square ae-icon"></i>Rolling 63-Day Sharpe Ratio</h3>', unsafe_allow_html=True)

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

    st.divider()
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-shield-halved ae-icon"></i>Gross Exposure Over Time</h3>', unsafe_allow_html=True)
    st.caption("Sum of long + short positions as multiple of portfolio capital. Capped by max leverage setting.")

    fig3, ax3 = plt.subplots(figsize=(13, 3))
    for i, (name, bt) in enumerate(backtest_results.items()):
        ax3.plot(bt.gross_exposure.index, bt.gross_exposure.values,
                 label=f"{name}  (avg: {bt.gross_exposure.mean():.2f}×)",
                 color=colours[i % len(colours)], linewidth=1.4)
    ax3.axhline(max_leverage, color="orange", linewidth=0.8, linestyle="--", alpha=0.6,
                label=f"Leverage cap ({max_leverage:.1f}×)")
    ax3.set_ylabel("Gross Exposure (×)")
    ax3.legend(fontsize=9)
    ax3.tick_params(axis="x", rotation=20)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ── Tab 3: Drawdowns ────────────────────────────────────────────────────────
with tab_drawdown:
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-arrow-trend-down ae-icon"></i>Drawdown from Peak</h3>', unsafe_allow_html=True)

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
    st.dataframe(pd.DataFrame(dd_rows).set_index("Strategy"), width="stretch")


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
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-calendar ae-icon"></i>Annual Returns</h3>', unsafe_allow_html=True)
    try:
        annual = (1 + bt.daily_returns).resample("YE").prod() - 1
        annual_df = annual.to_frame("Return")
        annual_df.index = annual_df.index.year
        annual_df.index.name = "Year"
        annual_df["Return"] = annual_df["Return"].map(_fmt_pct)
        st.dataframe(annual_df.T, width="stretch")
    except Exception:
        pass


# ── Tab 5: Data Explorer ────────────────────────────────────────────────────
with tab_data:
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-database ae-icon"></i>Universe & Price Data</h3>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tickers in universe", len(prices.columns))
    c2.metric("Trading days", len(prices))
    c3.metric("Date range", f"{prices.index[0].date()} → {prices.index[-1].date()}")

    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-table ae-icon"></i>Sample Prices (last 10 rows, first 15 tickers)</h3>', unsafe_allow_html=True)
    st.dataframe(
        prices.iloc[-10:, :15].round(2),
        width="stretch",
    )

    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-briefcase ae-icon"></i>Position Weights</h3>', unsafe_allow_html=True)
    strategy_choice_data = st.selectbox("Strategy ", list(backtest_results.keys()),
                                         key="data_strat")
    bt_data = backtest_results[strategy_choice_data]

    st.caption("Last 5 trading days — top 20 non-zero positions")
    last_weights = bt_data.positions.iloc[-5:]
    # Show only columns with recent non-zero weights
    active_cols = last_weights.columns[(last_weights.abs() > 0.001).any()][:20]
    if len(active_cols):
        st.dataframe(last_weights[active_cols].round(4), width="stretch")
    else:
        st.info("No active positions in the last 5 days.")

    # Download buttons
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-file-arrow-down ae-icon"></i>Export</h3>', unsafe_allow_html=True)
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
