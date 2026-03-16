"""
Paper Trading page — Alpha Engine

Simulates live strategy execution with fake money.
Uses the same strategy and data pipeline as the backtester,
but runs against the most recent available market data.

Everything here is SIMULATED. No real money moves.
"""

from __future__ import annotations

import sys
import logging
from datetime import date, timedelta
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


logging.basicConfig(level=logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# PAPER MODE CSS — orange theme injected globally
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Orange accent on buttons */
.stButton > button[kind="primary"] {
    background-color: #e65c00 !important;
    border-color: #e65c00 !important;
}
/* Orange sidebar header */
section[data-testid="stSidebar"] {
    border-right: 4px solid #FF6B00;
}
/* Paper mode badge in metrics */
div[data-testid="metric-container"] {
    border-left: 3px solid #FF6B00;
    padding-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sticky PAPER MODE banner — impossible to miss
# ---------------------------------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #FF6B00, #FF8C00);
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 12px 24px;
    border-radius: 6px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 2px 8px rgba(255,107,0,0.35);
">
    <i class="fa-solid fa-file-pen" style="font-size:1.4rem;opacity:0.9"></i>
    <span>PAPER TRADING MODE &nbsp;·&nbsp; SIMULATED PORTFOLIO &nbsp;·&nbsp; NO REAL MONEY INVOLVED</span>
    <span style="margin-left:auto; font-size:0.85rem; opacity:0.85">Prices from Yahoo Finance &nbsp;|&nbsp; All values in GBP (£)</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    if np.isnan(v):
        return "—"
    return f"+{v:.1%}" if v > 0 else f"{v:.1%}"

def _fmt_gbp(v: float) -> str:
    return f"£{v:,.2f}" if not np.isnan(v) else "—"

def _fmt_gbp0(v: float) -> str:
    return f"£{v:,.0f}" if not np.isnan(v) else "—"

def _pnl_colour(v: float) -> str:
    return "green" if v >= 0 else "red"

def _paper_label(label: str) -> str:
    """Append PAPER badge to a metric label."""
    return f"{label}  *(paper)*"

# ---------------------------------------------------------------------------
# Cached data fetching — gets the most recent N days for signal computation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fetching latest LSE prices…", ttl=1800)
def fetch_latest_prices(universe: str, universe_size: int, lookback_days: int = 400):
    """
    Download the most recent ``lookback_days`` of price data.
    Returns (prices_df, returns_df, ohlcv_dict, latest_prices_series).
    ``lookback_days`` = 320 covers 12 months of momentum lookback + buffer.
    """
    from config import DataConfig
    from data.data_loader import DataLoader, clean_data, compute_returns
    from data.universe import load_ftse_universe, apply_liquidity_filters

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()

    data_cfg = DataConfig(universe=universe, universe_size=universe_size)
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

    # Latest closing prices (most recent row, drop NaN)
    latest_prices = prices.iloc[-1].dropna()
    return prices, returns, ohlcv, latest_prices


@st.cache_data(show_spinner="Computing strategy signals…", ttl=1800)
def compute_signals(
    universe: str,
    universe_size: int,
    strategy_name: str,
    mom_lookback: int,
    mom_skip: int,
    mom_long_pct: float,
    mr_lookback: int,
    mr_long_pct: float,
    gap_threshold: float,
    hold_days: int,
):
    """Compute today's target weights from the selected strategy."""
    from config import StrategyConfig
    from strategies.momentum import CrossSectionalMomentum
    from strategies.mean_reversion import ShortTermMeanReversion
    from strategies.earnings_revision import EarningsRevisionDrift

    prices, returns, ohlcv, latest_prices = fetch_latest_prices(universe, universe_size)

    cfg = StrategyConfig(
        momentum_lookback=mom_lookback,
        momentum_skip=mom_skip,
        momentum_long_pct=mom_long_pct,
        mean_reversion_lookback=mr_lookback,
        mean_reversion_long_pct=mr_long_pct,
        earnings_gap_threshold=gap_threshold,
        earnings_hold_days=hold_days,
    )

    strategy_map = {
        "momentum": CrossSectionalMomentum(cfg),
        "mean_reversion": ShortTermMeanReversion(cfg),
        "earnings": EarningsRevisionDrift(cfg),
    }
    kwargs = {"ohlcv": ohlcv} if strategy_name == "earnings" else {}
    result = strategy_map[strategy_name].run(prices, returns, **kwargs)

    # Today's target weights = last row of weights DataFrame
    today_weights = result.weights.iloc[-1].dropna()
    today_weights = today_weights[today_weights.abs() > 0.001]  # filter noise
    return today_weights, latest_prices, prices.index[-1]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="
        background:#FF6B00; color:white; font-weight:700;
        padding:8px 12px; border-radius:4px; text-align:center;
        letter-spacing:0.08em; margin-bottom:1rem;
    "><i class="fa-solid fa-file-pen" style="margin-right:6px"></i>PAPER TRADING</div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.95rem;font-weight:600;margin:0.6rem 0 0.2rem"><i class="fa-solid fa-briefcase ae-icon"></i>Portfolio Setup</p>', unsafe_allow_html=True)
    initial_capital = st.number_input(
        "Starting capital (£)", value=10_000, step=1_000, min_value=1_000, format="%d",
        help="Fake GBP. This is not real money."
    )

    st.markdown('<p style="font-size:0.95rem;font-weight:600;margin:0.6rem 0 0.2rem"><i class="fa-solid fa-layer-group ae-icon"></i>Strategy</p>', unsafe_allow_html=True)
    strategy = st.selectbox(
        "Signal source",
        ["momentum", "mean_reversion", "earnings"],
        format_func=lambda x: {
            "momentum": "Momentum (12-1)",
            "mean_reversion": "Mean Reversion (5-day)",
            "earnings": "Earnings Drift (gap)",
        }[x],
    )
    universe = st.selectbox(
        "Universe",
        ["ftse100", "ftse250", "ftse_all"],
        format_func=lambda x: INDEX_DISPLAY_NAMES.get(x, x),
    )
    universe_size = st.slider("Max tickers", 30, 200, 100, step=10)

    st.markdown('<p style="font-size:0.95rem;font-weight:600;margin:0.6rem 0 0.2rem"><i class="fa-solid fa-sliders ae-icon"></i>Signal Parameters</p>', unsafe_allow_html=True)
    with st.expander("Momentum", expanded=strategy == "momentum"):
        mom_lookback = st.slider("Lookback (days)", 63, 504, 252, step=21, key="pm_lb")
        mom_skip = st.slider("Skip (days)", 0, 63, 21, step=5, key="pm_skip")
        mom_long_pct = st.slider("Decile size", 0.05, 0.30, 0.10, step=0.05,
                                  format="%.0f%%", key="pm_pct")

    with st.expander("Mean Reversion", expanded=strategy == "mean_reversion"):
        mr_lookback = st.slider("Lookback (days)", 3, 21, 5, step=1, key="mr_lb")
        mr_long_pct = st.slider("Quintile size", 0.10, 0.40, 0.20, step=0.05,
                                 format="%.0f%%", key="mr_pct")

    with st.expander("Earnings Drift", expanded=strategy == "earnings"):
        gap_threshold = st.slider("Gap threshold", 0.02, 0.15, 0.05, step=0.01,
                                   format="%.0f%%", key="ep_gap")
        hold_days = st.slider("Hold days", 3, 30, 10, step=1, key="ep_hold")

    st.markdown('<p style="font-size:0.95rem;font-weight:600;margin:0.6rem 0 0.2rem"><i class="fa-solid fa-coins ae-icon"></i>Costs</p>', unsafe_allow_html=True)
    commission_bps = st.slider("Commission (bps)", 0, 30, 10, key="p_comm")
    slippage_bps = st.slider("Slippage (bps)", 0, 30, 10, key="p_slip")
    stamp_duty_pct = st.slider("Stamp duty (%)", 0.0, 1.0, 0.5, step=0.1,
                                key="p_sdrt") / 100

    st.divider()

    # Action buttons
    btn_refresh = st.button("Refresh Prices & Signals",
                             width="stretch",
                             help="Download latest prices and recompute signals")
    btn_rebalance = st.button("Execute Rebalance",
                               width="stretch", type="primary",
                               help="Apply today's signals to your paper portfolio")
    btn_reset = st.button("Reset Portfolio",
                           width="stretch",
                           help="Delete paper portfolio and start fresh")

    st.caption("All simulated. No real orders are placed.")

# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

from paper_trading.portfolio import PaperPortfolio

portfolio = PaperPortfolio()

# Handle reset
if btn_reset:
    portfolio.reset()
    st.session_state.pop("paper_signals", None)
    st.session_state.pop("paper_prices", None)
    st.rerun()

# ---------------------------------------------------------------------------
# Setup screen — shown when no portfolio exists
# ---------------------------------------------------------------------------

if not portfolio.is_active:
    st.markdown('<h1 class="ae-title"><i class="fa-solid fa-file-pen ae-icon"></i>Paper Trading</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-play ae-icon"></i>Start your paper portfolio</h3>', unsafe_allow_html=True)

    st.markdown("""
    Paper trading lets you test a strategy with **simulated money** before committing
    real capital. This is step 3 of the research workflow:

    ```
    1. Factor Research  → test if the signal predicts returns
    2. Backtest         → test if it makes money after costs
    3. Paper Trading    → run it live with simulated money  ← you are here
    ```

    This mode:
    - Uses **live prices** from Yahoo Finance (LSE tickers)
    - Applies the **same factor logic** as the backtester
    - Tracks your simulated equity, positions, and trade history
    - Models **UK transaction costs** including 0.5% stamp duty
    - **Never places real orders**
    """)

    col1, col2, col3 = st.columns(3)
    col1.info(f"**Starting capital:** {_fmt_gbp0(initial_capital)}")
    col2.info(f"**Strategy:** {strategy.replace('_', ' ').title()}")
    col3.info(f"**Universe:** {INDEX_DISPLAY_NAMES.get(universe, universe.upper())}")

    if st.button("Start Paper Portfolio", type="primary", width="content"):
        portfolio.create(
            initial_capital=float(initial_capital),
            strategy=strategy,
            universe=universe,
            universe_size=universe_size,
            commission_bps=float(commission_bps),
            slippage_bps=float(slippage_bps),
            stamp_duty_rate=stamp_duty_pct,
        )
        st.success("Paper portfolio created. Fetching latest signals…")
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Active portfolio — fetch signals and update prices
# ---------------------------------------------------------------------------

# Fetch latest signals if button pressed or first load
if btn_refresh or "paper_signals" not in st.session_state:
    with st.spinner("Fetching latest LSE prices and signals…"):
        try:
            signals, latest_prices, signal_date = compute_signals(
                universe=portfolio.config.get("universe", universe),
                universe_size=portfolio.config.get("universe_size", universe_size),
                strategy_name=portfolio.config.get("strategy", strategy),
                mom_lookback=mom_lookback,
                mom_skip=mom_skip,
                mom_long_pct=mom_long_pct,
                mr_lookback=mr_lookback,
                mr_long_pct=mr_long_pct,
                gap_threshold=gap_threshold,
                hold_days=hold_days,
            )
            st.session_state["paper_signals"] = signals
            st.session_state["paper_prices"] = latest_prices
            st.session_state["paper_signal_date"] = signal_date
        except Exception as e:
            st.error(f"Failed to fetch prices: {e}")
            st.stop()

signals: pd.Series = st.session_state.get("paper_signals", pd.Series(dtype=float))
latest_prices: pd.Series = st.session_state.get("paper_prices", pd.Series(dtype=float))
signal_date = st.session_state.get("paper_signal_date", "—")

# Execute rebalance
if btn_rebalance and len(signals) > 0 and len(latest_prices) > 0:
    executed_trades = portfolio.execute_rebalance(signals, latest_prices)
    st.success(f"Paper rebalance executed — {len(executed_trades)} trades")
    st.rerun()

# Get current portfolio state
positions_snap, current_equity = portfolio.refresh_prices(latest_prices)

initial_cap = portfolio.initial_capital
total_pnl = current_equity - initial_cap
total_pnl_pct = total_pnl / initial_cap if initial_cap > 0 else 0.0

# ---------------------------------------------------------------------------
# Dashboard header — always visible
# ---------------------------------------------------------------------------

st.markdown('<h1 class="ae-title"><i class="fa-solid fa-briefcase ae-icon"></i>Paper Portfolio</h1>', unsafe_allow_html=True)
st.caption(
    f"Strategy: **{portfolio.config.get('strategy', '—').replace('_',' ').title()}** &nbsp;|&nbsp; "
    f"Universe: **{INDEX_DISPLAY_NAMES.get(portfolio.config.get('universe', ''), portfolio.config.get('universe', '—').upper())}** &nbsp;|&nbsp; "
    f"Started: **{portfolio.config.get('created_date', portfolio._state.get('created_date','—'))}** &nbsp;|&nbsp; "
    f"Signal date: **{str(signal_date)[:10] if signal_date != '—' else '—'}**"
)

with st.expander("What this page does — click to read", expanded=False):
    st.markdown("""
    ### Live signal generation with simulated money

    Paper trading is the bridge between backtesting and real trading.
    It runs the **same factor logic** but against today's actual prices,
    so you can see what the strategy would do right now — without risking money.

    ```
    Download today's LSE prices
    ↓
    Compute factor scores for all stocks
    ↓
    Rank stocks → build long/short portfolio weights
    ↓
    Compare to current holdings → generate trade list
    ↓
    Execute simulated trades (deduct costs)
    ↓
    Track equity and positions over time
    ```

    ---

    ### How to use this page

    | Button | What it does |
    |---|---|
    | **Refresh Prices & Signals** | Download latest prices and recompute today's factor scores |
    | **Execute Rebalance** | Apply today's signals — buy/sell positions at current prices |
    | **Reset Portfolio** | Start fresh with new capital |

    **Typical daily workflow:**
    1. Click **Refresh Prices & Signals** to get the latest prices
    2. Review the **Pending Trades** section — check what the strategy wants to do today
    3. Click **Execute Rebalance** to apply the trades
    4. Monitor P&L and positions over time

    ---

    ### Important: everything here is simulated

    - No real orders are ever placed
    - Prices are from Yahoo Finance (15-min delayed for LSE)
    - Transaction costs are modelled but not actually paid
    - Use this to build conviction before trading with real capital
    """)


# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    _paper_label("Portfolio Value"),
    _fmt_gbp0(current_equity),
    delta=_fmt_gbp(total_pnl),
)
k2.metric(
    _paper_label("Total P&L"),
    _fmt_pct(total_pnl_pct),
    delta=_fmt_gbp(total_pnl),
    delta_color="normal",
)
k3.metric(
    _paper_label("Starting Capital"),
    _fmt_gbp0(initial_cap),
)
k4.metric(
    _paper_label("Open Positions"),
    len(positions_snap),
)
k5.metric(
    _paper_label("Total Trades"),
    len(portfolio.trade_log),
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_pos, tab_signals, tab_history, tab_trades = st.tabs([
    "Positions", "Today's Signals", "Equity History", "Trade Log"
])

# ── Tab 1: Current Positions ────────────────────────────────────────────────
with tab_pos:
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-briefcase ae-icon"></i>Open Positions <em style="font-size:0.75em;opacity:0.7">(paper)</em></h3>', unsafe_allow_html=True)

    if not positions_snap:
        st.info(
            "No open positions yet.  \n"
            "Click **Execute Rebalance** in the sidebar to enter today's signals."
        )
    else:
        pos_rows = []
        for p in sorted(positions_snap, key=lambda x: abs(x.market_value), reverse=True):
            pos_rows.append({
                "Ticker": p.ticker,
                "Side": p.side.upper(),
                "Shares": f"{p.shares:.2f}",
                "Entry Price": _fmt_gbp(p.entry_price),
                "Current Price": _fmt_gbp(p.current_price),
                "Market Value": _fmt_gbp0(p.market_value),
                "Unrealised P&L": _fmt_gbp(p.unrealised_pnl),
                "Return": _fmt_pct(p.unrealised_pct),
                "Entry Date": p.entry_date,
            })

        pos_df = pd.DataFrame(pos_rows)
        st.dataframe(pos_df, width="stretch", hide_index=True)

        # Allocation chart
        st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-chart-pie ae-icon"></i>Portfolio Allocation</h3>', unsafe_allow_html=True)
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        longs = [(p.ticker, p.market_value) for p in positions_snap if p.side == "long"]
        shorts = [(p.ticker, abs(p.market_value)) for p in positions_snap if p.side == "short"]

        fig, axes = plt.subplots(1, 2 if longs and shorts else 1, figsize=(12, 4))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        colors_long = plt.cm.Blues(np.linspace(0.4, 0.85, max(len(longs), 1)))
        colors_short = plt.cm.Reds(np.linspace(0.4, 0.85, max(len(shorts), 1)))

        if longs:
            tickers_l, vals_l = zip(*longs)
            axes[0].barh(tickers_l, vals_l, color=colors_long)
            axes[0].set_title("Long Positions (£)", fontsize=10)
            axes[0].xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"£{x:,.0f}")
            )

        if shorts and len(axes) > 1:
            tickers_s, vals_s = zip(*shorts)
            axes[1].barh(tickers_s, vals_s, color=colors_short)
            axes[1].set_title("Short Positions (£, abs)", fontsize=10)
            axes[1].xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"£{x:,.0f}")
            )

        # Paper mode watermark on chart
        for ax in axes:
            ax.text(0.99, 0.01, "PAPER", transform=ax.transAxes,
                    fontsize=20, color="orange", alpha=0.2,
                    ha="right", va="bottom", fontweight="bold")

        fig.suptitle("PAPER PORTFOLIO — Simulated Positions", fontsize=11,
                     color="#e65c00", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        cash = float(portfolio._state.get("cash", 0))
        st.caption(f"Cash: **{_fmt_gbp0(cash)}** &nbsp;|&nbsp; "
                   f"Market value: **{_fmt_gbp0(current_equity - cash)}** &nbsp;|&nbsp; "
                   f"Gross leverage: **{(current_equity - cash) / max(initial_cap, 1):.2f}×**")


# ── Tab 2: Today's Signals ──────────────────────────────────────────────────
with tab_signals:
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-crosshairs ae-icon"></i>Today\'s Target Weights — ' + portfolio.config.get("strategy","").replace("_"," ").title() + '</h3>', unsafe_allow_html=True)
    st.caption(
        f"Generated from prices as of **{str(signal_date)[:10]}**. "
        "These are the weights the strategy *wants* today. "
        "Click **Execute Rebalance** to apply them."
    )

    if len(signals) == 0:
        st.info("Click **Refresh Prices & Signals** in the sidebar to load today's signals.")
    else:
        # Split long / short
        longs = signals[signals > 0].sort_values(ascending=False)
        shorts = signals[signals < 0].sort_values()

        col_l, col_s = st.columns(2)
        with col_l:
            st.markdown('<p style="font-weight:600;color:#22c55e"><i class="fa-solid fa-arrow-up ae-icon"></i>Long positions</p>', unsafe_allow_html=True)
            if len(longs):
                long_df = longs.reset_index()
                long_df.columns = ["Ticker", "Target Weight"]
                long_df["Target Weight"] = long_df["Target Weight"].map(
                    lambda x: f"{x:.2%}"
                )
                st.dataframe(long_df, width="stretch", hide_index=True)
            else:
                st.caption("No long signals today.")

        with col_s:
            st.markdown('<p style="font-weight:600;color:#ef4444"><i class="fa-solid fa-arrow-down ae-icon"></i>Short positions</p>', unsafe_allow_html=True)
            if len(shorts):
                short_df = shorts.abs().reset_index()
                short_df.columns = ["Ticker", "Target Weight"]
                short_df["Target Weight"] = short_df["Target Weight"].map(
                    lambda x: f"{x:.2%}"
                )
                st.dataframe(short_df, width="stretch", hide_index=True)
            else:
                st.caption("No short signals today.")

        st.divider()
        st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-list-check ae-icon"></i>Pending Trades (preview)</h3>', unsafe_allow_html=True)
        st.caption("Trades required to move from current positions to today's target weights.")

        if len(latest_prices) > 0:
            pending = portfolio.preview_rebalance(signals, latest_prices)
            if pending:
                pend_rows = []
                for t in pending:
                    pend_rows.append({
                        "Ticker": t["ticker"],
                        "Action": t["action"],
                        "Shares": f"{t['shares']:.2f}",
                        "Price": _fmt_gbp(t["price"]),
                        "Notional": _fmt_gbp0(t["notional"]),
                        "Est. Cost (incl. SDRT)": _fmt_gbp(t["estimated_cost"]),
                    })
                pend_df = pd.DataFrame(pend_rows)

                # Colour action column
                def colour_action(val):
                    c = {"BUY": "#006400", "SELL": "#8B0000",
                         "SHORT": "#8B0000", "COVER": "#006400"}.get(val, "black")
                    return f"color: {c}; font-weight: bold"

                st.dataframe(
                    pend_df.style.applymap(colour_action, subset=["Action"]),
                    width="stretch",
                    hide_index=True,
                )
                total_cost = sum(t["estimated_cost"] for t in pending)
                total_notional = sum(t["notional"] for t in pending)
                st.caption(
                    f"Total trades: **{len(pending)}** &nbsp;|&nbsp; "
                    f"Total notional: **{_fmt_gbp0(total_notional)}** &nbsp;|&nbsp; "
                    f"Estimated total cost: **{_fmt_gbp(total_cost)}**"
                )
            else:
                st.success("Portfolio already matches today's signals — no trades needed.")
        else:
            st.info("Refresh prices to preview pending trades.")


# ── Tab 3: Equity History ────────────────────────────────────────────────────
with tab_history:
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-chart-line ae-icon"></i>Portfolio Equity History <em style="font-size:0.75em;opacity:0.7">(paper)</em></h3>', unsafe_allow_html=True)

    equity_series = portfolio.equity_history

    if len(equity_series) < 2:
        st.info(
            "Equity history builds up over time as you refresh and rebalance. "
            "Come back tomorrow!"
        )
    else:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(equity_series.index, equity_series.values,
                color="#FF6B00", linewidth=2.0, label="Paper Portfolio (£)")
        ax.axhline(initial_cap, color="grey", linewidth=0.8,
                   linestyle="--", label=f"Starting capital {_fmt_gbp0(initial_cap)}")
        ax.fill_between(equity_series.index, initial_cap, equity_series.values,
                        where=equity_series.values >= initial_cap,
                        alpha=0.15, color="#2ecc71", label="Above water")
        ax.fill_between(equity_series.index, initial_cap, equity_series.values,
                        where=equity_series.values < initial_cap,
                        alpha=0.15, color="#e74c3c", label="Below water")

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        ax.set_ylabel("Portfolio Value (£)")
        ax.set_title("PAPER Portfolio Equity — Simulated (GBP)", fontsize=11,
                     color="#e65c00", fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=20)

        # Watermark
        ax.text(0.5, 0.5, "PAPER TRADING", transform=ax.transAxes,
                fontsize=36, color="orange", alpha=0.08,
                ha="center", va="center", fontweight="bold", rotation=15)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary stats
        daily_returns = equity_series.pct_change().dropna()
        if len(daily_returns) >= 2:
            c1, c2, c3, c4 = st.columns(4)
            total_ret = equity_series.iloc[-1] / equity_series.iloc[0] - 1
            c1.metric("Total Return *(paper)*", _fmt_pct(total_ret))
            c2.metric("Peak Value *(paper)*", _fmt_gbp0(equity_series.max()))
            c3.metric("Lowest Value *(paper)*", _fmt_gbp0(equity_series.min()))
            dd = (equity_series / equity_series.cummax() - 1).min()
            c4.metric("Max Drawdown *(paper)*", _fmt_pct(dd))


# ── Tab 4: Trade Log ─────────────────────────────────────────────────────────
with tab_trades:
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-terminal ae-icon"></i>Trade Log <em style="font-size:0.75em;opacity:0.7">(paper)</em></h3>', unsafe_allow_html=True)
    st.caption("All trades are simulated. No real orders are placed.")

    trades = portfolio.trade_log
    if not trades:
        st.info("No trades yet. Execute a rebalance to see trades here.")
    else:
        trade_rows = []
        for t in reversed(trades):  # most recent first
            trade_rows.append({
                "Date": t["date"],
                "Ticker": t["ticker"],
                "Action": t["action"],
                "Shares": f"{t['shares']:.2f}",
                "Price": _fmt_gbp(t["price"]),
                "Notional": _fmt_gbp0(t["notional"]),
                "Commission": _fmt_gbp(t.get("commission", 0)),
                "Slippage": _fmt_gbp(t.get("slippage", 0)),
                "Stamp Duty": _fmt_gbp(t.get("stamp_duty", 0)),
                "Total Cost": _fmt_gbp(t["total_cost"]),
            })

        trade_df = pd.DataFrame(trade_rows)

        def colour_action(val):
            c = {"BUY": "#006400", "SELL": "#8B0000",
                 "SHORT": "#8B0000", "COVER": "#006400"}.get(val, "black")
            return f"color: {c}; font-weight: bold"

        st.dataframe(
            trade_df.style.applymap(colour_action, subset=["Action"]),
            width="stretch",
            hide_index=True,
        )

        total_commissions = sum(t.get("commission", 0) for t in trades)
        total_slippage = sum(t.get("slippage", 0) for t in trades)
        total_sdrt = sum(t.get("stamp_duty", 0) for t in trades)
        total_costs = sum(t["total_cost"] for t in trades)

        st.caption(
            f"**{len(trades)} total trades** &nbsp;|&nbsp; "
            f"Commission: {_fmt_gbp(total_commissions)} &nbsp;|&nbsp; "
            f"Slippage: {_fmt_gbp(total_slippage)} &nbsp;|&nbsp; "
            f"Stamp duty: {_fmt_gbp(total_sdrt)} &nbsp;|&nbsp; "
            f"**Total costs: {_fmt_gbp(total_costs)}**"
        )

        col_dl, _ = st.columns([1, 3])
        col_dl.download_button(
            "Download Trade Log (CSV)",
            data=trade_df.to_csv(index=False).encode(),
            file_name="paper_trade_log.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Bottom banner — reinforces paper mode on every scroll position
# ---------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style="
    background: #fff3e0;
    border: 2px solid #FF6B00;
    border-radius: 6px;
    padding: 10px 18px;
    color: #7f3800;
    font-size: 0.85rem;
    text-align: center;
">
    <i class="fa-solid fa-triangle-exclamation" style="margin-right:6px"></i> <strong>PAPER TRADING MODE</strong> &nbsp;·&nbsp;
    All positions, trades, and P&amp;L figures are <strong>entirely simulated</strong>.
    No real money is involved and no real orders are placed. &nbsp;·&nbsp;
    Past simulated performance is not indicative of future real results.
</div>
""", unsafe_allow_html=True)
