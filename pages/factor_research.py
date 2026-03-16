"""
Factor Research page — Alpha Engine

Evaluate alpha signals as quantitative factors: IC analysis, factor decay,
and quantile portfolio analysis.
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


logging.basicConfig(level=logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.1%}" if not np.isnan(v) else "—"
    except Exception:
        return "—"

def _fmt_f2(v) -> str:
    try:
        return f"{float(v):.2f}" if not np.isnan(v) else "—"
    except Exception:
        return "—"

def _fmt_f3(v) -> str:
    try:
        return f"{float(v):.3f}" if not np.isnan(v) else "—"
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Cached data + signal loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading price data…", ttl=3600)
def load_factor_data(
    universe: str,
    universe_size: int,
    start: str,
    end: str,
    force_refresh: bool = False,
):
    from config import DataConfig
    from data.data_loader import DataLoader, clean_data, compute_returns
    from data.universe import load_ftse_universe

    data_cfg = DataConfig(universe=universe, universe_size=universe_size, cache_dir="data/cache")
    tickers = load_ftse_universe(index=universe, top_n=universe_size)
    loader = DataLoader(cache_dir="data/cache", config=data_cfg)

    prices_raw = loader.load_price_data(tickers, start, end, force_refresh=force_refresh)
    ohlcv = loader.load_ohlcv(list(prices_raw.columns), start, end)
    prices = clean_data(prices_raw, min_history_days=data_cfg.min_history_days)
    returns = compute_returns(prices)
    return prices, returns, ohlcv


@st.cache_data(show_spinner="Computing factor signals…", ttl=3600)
def compute_factor_signals(
    factor_name: str,
    universe: str,
    universe_size: int,
    start: str,
    end: str,
    factor_kwargs: dict,
    force_refresh: bool = False,
):
    """Compute signals for a registered factor using the factor registry."""
    from factors import get_factor

    prices, returns, ohlcv = load_factor_data(
        universe, universe_size, start, end, force_refresh=force_refresh
    )
    factor = get_factor(factor_name, **factor_kwargs)
    signals = factor.compute(prices, returns)
    return prices, returns, signals


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.markdown(
    '<h1 class="ae-title"><i class="fa-solid fa-flask ae-icon"></i>Factor Research</h1>',
    unsafe_allow_html=True,
)

# Compact 3-column intro banner
_b1, _b2, _b3 = st.columns(3)
with _b1:
    st.markdown("""
    <div style="background:#1e2a3a;border-left:3px solid #3b82f6;padding:12px 14px;border-radius:4px;font-size:0.9rem;line-height:1.55">
    <strong><i class="fa-solid fa-brain ae-icon" style="color:#3b82f6"></i>What is a factor?</strong><br>
    Any signal that might predict which stocks will outperform — e.g. past 12-month return,
    recent reversal, earnings surprise. This page tests the signal statistically before you trade on it.
    </div>
    """, unsafe_allow_html=True)
with _b2:
    st.markdown("""
    <div style="background:#1e2a3a;border-left:3px solid #3b82f6;padding:12px 14px;border-radius:4px;font-size:0.9rem;line-height:1.55">
    <strong><i class="fa-solid fa-chart-line ae-icon" style="color:#3b82f6"></i>Key metric: IC</strong><br>
    Spearman correlation between factor scores and next-period returns.
    Ranges −1 to +1. Equity factors rarely exceed 0.10.<br>
    <strong>|t-stat| &gt; 2.0</strong> = significant at ~95% confidence.
    </div>
    """, unsafe_allow_html=True)
with _b3:
    st.markdown("""
    <div style="background:#1e2a3a;border-left:3px solid #3b82f6;padding:12px 14px;border-radius:4px;font-size:0.9rem;line-height:1.55">
    <strong><i class="fa-solid fa-route ae-icon" style="color:#3b82f6"></i>Typical workflow</strong><br>
    1. Pick factor + date range<br>
    2. Run Factor Analysis<br>
    3. Check IC Mean &amp; t-stat<br>
    4. Check Decay — how fast does it fade?<br>
    5. Save to Leaderboard if promising
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p style="font-size:1rem;font-weight:700;margin:0 0 4px"><i class="fa-solid fa-gear ae-icon"></i>Factor Settings</p>', unsafe_allow_html=True)

    universe = st.selectbox("Universe", ["ftse100", "ftse250", "ftse_all"], index=0,
        format_func=lambda x: INDEX_DISPLAY_NAMES.get(x, x))
    universe_size = st.slider("Universe size", 20, 350, 100, step=10)

    st.divider()

    from factors import list_factors, factor_metadata
    available_factors = list_factors()
    factor_meta = factor_metadata()

    factor_name = st.selectbox(
        "Factor",
        available_factors,
        format_func=lambda x: factor_meta.get(x, {}).get("description", x),
    )

    st.divider()

    start_date = st.date_input("Start date", value=pd.Timestamp("2018-01-01"))
    end_date = st.date_input("End date", value=pd.Timestamp("2023-12-31"))

    st.divider()
    st.subheader("Factor Parameters")

    factor_kwargs: dict = {}
    if factor_name == "momentum_12_1":
        factor_kwargs["lookback"] = st.slider("Lookback (days)", 63, 504, 252, step=21)
        factor_kwargs["skip"] = st.slider("Skip days (end)", 0, 63, 21, step=5)
    elif factor_name == "mean_reversion_5d":
        factor_kwargs["lookback"] = st.slider("Lookback (days)", 1, 21, 5, step=1)

    st.divider()
    st.subheader("IC Settings")
    ic_method = st.selectbox("IC method", ["spearman", "pearson"])
    ic_horizon = st.slider("IC horizon (days)", 1, 20, 1)
    rolling_ic_window = st.slider("Rolling IC window (days)", 21, 252, 63, step=21)

    st.divider()
    st.subheader("Quantile Settings")
    n_quantiles = st.slider("Number of quantiles", 3, 10, 5)

    force_refresh = st.checkbox(
        "Force refresh data",
        value=False,
        help="Delete cached price data and re-download from Yahoo Finance",
    )

    run_btn = st.button("Run Factor Analysis", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

if run_btn:
    with st.spinner("Loading data and computing signals…"):
        try:
            prices, returns, signals = compute_factor_signals(
                factor_name=factor_name,
                universe=universe,
                universe_size=universe_size,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                factor_kwargs=factor_kwargs,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            st.error(f"Failed to load data: {exc}")
            st.stop()
    # Store run config in session_state so the Save button works after analysis
    st.session_state["last_factor_run"] = {
        "factor_name": factor_name,
        "universe": universe,
        "universe_size": universe_size,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "factor_kwargs": factor_kwargs,
        "ic_horizon": ic_horizon,
        "n_quantiles": n_quantiles,
    }

    # Forward returns at chosen horizon
    fwd_returns = prices.pct_change(ic_horizon).shift(-ic_horizon)

    # ---------- IC Analysis ----------
    from factor_research.ic import compute_ic, compute_rolling_ic, compute_ic_summary
    from factor_research.decay import compute_factor_decay
    from factor_research.quantile import compute_quantile_portfolios

    with st.spinner("Computing IC…"):
        ic_summary = compute_ic_summary(signals, fwd_returns, method=ic_method)
        daily_ic = compute_ic(signals, fwd_returns, method=ic_method)
        rolling_ic = compute_rolling_ic(signals, fwd_returns, window=rolling_ic_window, method=ic_method)

    with st.spinner("Computing factor decay…"):
        decay_df = compute_factor_decay(signals, prices, horizons=(1, 5, 10, 20), method=ic_method)

    with st.spinner("Computing quantile portfolios…"):
        fwd1 = prices.pct_change(1).shift(-1)
        qr = compute_quantile_portfolios(signals, fwd1, n_quantiles=n_quantiles)

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------

    tab_ic, tab_decay, tab_quantile = st.tabs([
        "IC Analysis",
        "Factor Decay",
        "Quantile Portfolios",
    ])

    # -----------------------------------------------------------------------
    # Tab 1 — IC
    # -----------------------------------------------------------------------
    with tab_ic:
        st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-brain ae-icon"></i>IC Summary Statistics</h3>', unsafe_allow_html=True)

        metrics_col, hist_col = st.columns([3, 2])

        with metrics_col:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "IC Mean",
                _fmt_f3(ic_summary["ic_mean"]),
                help="Average daily correlation between factor scores and forward returns. "
                     "Positive = factor correctly ranks stocks. Target: > 0.02.",
            )
            col2.metric(
                "IC Std",
                _fmt_f3(ic_summary["ic_std"]),
                help="Volatility of the daily IC. Lower is better — a consistent signal "
                     "is more reliable than one that swings between positive and negative.",
            )
            col3.metric(
                "IC t-stat",
                _fmt_f2(ic_summary["ic_tstat"]),
                help="Statistical significance of the IC. |t| > 2.0 means the signal is "
                     "unlikely to be random noise (95% confidence). Aim for |t| > 2.",
            )
            col4.metric(
                "IC IR (ann.)",
                _fmt_f2(ic_summary["ic_ir"]),
                help="Information Ratio = IC Mean ÷ IC Std × √252. Measures signal quality "
                     "adjusted for consistency. > 0.5 is considered a good factor.",
            )

            col5, col6 = st.columns(2)
            col5.metric(
                "% Positive IC",
                _fmt_pct(ic_summary["ic_positive_pct"]),
                help="Fraction of trading days where the factor correctly ranked stocks "
                     "(positive IC). > 52% suggests a persistent edge.",
            )
            col6.metric(
                "Observations",
                f"{int(ic_summary['obs']):,}",
                help="Number of trading days used to compute the IC statistics. "
                     "More observations = more reliable estimates.",
            )

            # IC interpretation helper
            ic_mean_v = ic_summary["ic_mean"]
            ic_t = ic_summary["ic_tstat"]
            if not np.isnan(ic_t):
                if abs(ic_t) >= 2.0 and ic_mean_v > 0:
                    st.success(f"Statistically significant positive IC (t={ic_t:.2f}). The factor has predictive power at the {ic_horizon}-day horizon.")
                elif abs(ic_t) >= 2.0 and ic_mean_v < 0:
                    st.warning(f"Statistically significant **negative** IC (t={ic_t:.2f}). The factor is a contrarian signal — flip the sign for use as a long signal.")
                else:
                    st.info(f"IC t-statistic ({ic_t:.2f}) is below ±2.0. The factor has limited predictive power at the {ic_horizon}-day horizon.")

        with hist_col:
            st.markdown('<p style="font-weight:600;margin:0 0 4px"><i class="fa-solid fa-chart-simple ae-icon"></i>IC Distribution</p>', unsafe_allow_html=True)
            import matplotlib.pyplot as plt
            _fig_h, _ax_h = plt.subplots(figsize=(5, 3.5))
            _fig_h.patch.set_facecolor("#0e1117")
            _ax_h.set_facecolor("#0e1117")
            _ic_vals = daily_ic.dropna().values
            _ax_h.hist(_ic_vals, bins=40, color="#3498db", alpha=0.75, edgecolor="#222")
            _ax_h.axvline(0, color="white", linewidth=0.8, linestyle="--")
            _ax_h.axvline(ic_summary["ic_mean"], color="#f39c12", linewidth=1.5,
                          linestyle="--", label=f"Mean={ic_summary['ic_mean']:.3f}")
            _ax_h.set_xlabel("Daily IC", color="white", fontsize=9)
            _ax_h.set_ylabel("Frequency", color="white", fontsize=9)
            _ax_h.tick_params(colors="white", labelsize=8)
            for spine in ["bottom", "left"]:
                _ax_h.spines[spine].set_color("#444")
            _ax_h.spines["top"].set_visible(False)
            _ax_h.spines["right"].set_visible(False)
            _ax_h.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
            _fig_h.tight_layout()
            st.pyplot(_fig_h)
            plt.close(_fig_h)

        st.divider()

        # Daily IC chart
        st.markdown(f'<h3 class="ae-sub"><i class="fa-solid fa-chart-bar ae-icon"></i>Daily IC — {ic_horizon}-day Forward Return</h3>', unsafe_allow_html=True)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.patch.set_facecolor("#0e1117")
        for ax in axes:
            ax.set_facecolor("#0e1117")

        # Bar chart of daily IC
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in daily_ic.values]
        axes[0].bar(daily_ic.index, daily_ic.values, color=colors, alpha=0.6, width=1.5)
        axes[0].axhline(0, color="white", linewidth=0.5)
        axes[0].axhline(ic_mean_v, color="#f39c12", linewidth=1.5, linestyle="--", label=f"Mean IC={ic_mean_v:.3f}")
        axes[0].set_ylabel("Daily IC", color="white")
        axes[0].tick_params(colors="white")
        axes[0].spines["bottom"].set_color("#444")
        axes[0].spines["left"].set_color("#444")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

        # Rolling IC
        axes[1].plot(rolling_ic.index, rolling_ic.values, color="#3498db", linewidth=1.5)
        axes[1].axhline(0, color="white", linewidth=0.5)
        axes[1].fill_between(rolling_ic.index, 0, rolling_ic.values,
                             where=rolling_ic.values > 0, alpha=0.2, color="#2ecc71")
        axes[1].fill_between(rolling_ic.index, 0, rolling_ic.values,
                             where=rolling_ic.values < 0, alpha=0.2, color="#e74c3c")
        axes[1].set_ylabel(f"Rolling {rolling_ic_window}d IC", color="white")
        axes[1].tick_params(colors="white")
        axes[1].spines["bottom"].set_color("#444")
        axes[1].spines["left"].set_color("#444")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[1].tick_params(axis="x", colors="white")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Tab 2 — Factor Decay
    # -----------------------------------------------------------------------
    with tab_decay:
        st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-arrow-trend-down ae-icon"></i>Factor Decay Analysis</h3>', unsafe_allow_html=True)
        st.markdown("IC at multiple forward-return horizons. A slow-decaying factor suits monthly strategies; a fast-decaying factor suits daily.")

        # Table
        decay_display = decay_df.copy()
        decay_display.index.name = "Horizon (days)"
        decay_display.columns = ["IC Mean", "IC Std", "IC t-stat", "IC IR (ann.)", "% Positive", "Obs"]
        decay_display["IC Mean"] = decay_display["IC Mean"].map(lambda x: f"{x:.4f}")
        decay_display["IC Std"] = decay_display["IC Std"].map(lambda x: f"{x:.4f}")
        decay_display["IC t-stat"] = decay_display["IC t-stat"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—")
        decay_display["IC IR (ann.)"] = decay_display["IC IR (ann.)"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—")
        decay_display["% Positive"] = decay_display["% Positive"].map(lambda x: f"{x:.1%}" if np.isfinite(x) else "—")
        decay_display["Obs"] = decay_display["Obs"].map(lambda x: f"{int(x):,}" if np.isfinite(x) else "—")
        st.dataframe(decay_display, width="stretch")

        # Decay chart
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        horizons_list = decay_df.index.tolist()
        ic_means = decay_df["ic_mean"].values
        ic_stds = decay_df["ic_std"].values

        ax.errorbar(horizons_list, ic_means, yerr=ic_stds,
                    color="#3498db", marker="o", markersize=8, linewidth=2,
                    capsize=5, capthick=2, ecolor="#555",
                    label="IC Mean ± 1 Std")
        ax.axhline(0, color="white", linewidth=0.5, linestyle="--")
        ax.fill_between(horizons_list, 0, ic_means,
                        where=np.array(ic_means) > 0, alpha=0.15, color="#2ecc71")
        ax.fill_between(horizons_list, 0, ic_means,
                        where=np.array(ic_means) < 0, alpha=0.15, color="#e74c3c")

        ax.set_xlabel("Forward Horizon (days)", color="white")
        ax.set_ylabel("IC Mean", color="white")
        ax.set_xticks(horizons_list)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        ax.set_title("IC Decay by Forward Horizon", color="white", fontsize=13, pad=10)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Tab 3 — Quantile Portfolios
    # -----------------------------------------------------------------------
    with tab_quantile:
        st.markdown(f'<h3 class="ae-sub"><i class="fa-solid fa-layer-group ae-icon"></i>Quantile Portfolio Analysis ({n_quantiles} quantiles)</h3>', unsafe_allow_html=True)
        st.markdown("Q1 = lowest factor (short signal), Q5/QN = highest factor (long signal). A useful factor shows monotonically increasing returns.")

        # Summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "L/S Spread Annual Return",
            _fmt_pct(qr.spread_annual_return),
            help="Annualised return of buying top-ranked stocks and shorting bottom-ranked "
                 "stocks, before transaction costs. This is the raw alpha of the factor.",
        )
        col2.metric(
            "L/S Spread Sharpe",
            _fmt_f2(qr.spread_sharpe),
            help="Sharpe ratio of the long-short spread. Measures return per unit of risk. "
                 "> 0.5 is encouraging; > 1.0 is strong for a raw factor.",
        )
        col3.metric(
            "Monotonicity Score",
            _fmt_f2(qr.monotonicity_score),
            help="Does higher factor rank → higher return? Score of 1.0 = perfect ordering "
                 "(Q1 < Q2 < ... < Q5). Score near 0 or negative = noisy or inverted factor.",
        )

        # Per-quantile bar chart
        st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-chart-bar ae-icon"></i>Annual Return by Quantile</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        q_labels = qr.mean_returns.index.tolist()
        q_vals = qr.mean_returns.values
        bar_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in q_vals]

        bars = ax.bar(q_labels, q_vals, color=bar_colors, alpha=0.8, edgecolor="#333")
        ax.axhline(0, color="white", linewidth=0.5)
        for bar, val in zip(bars, q_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.1%}", ha="center", va="bottom", color="white", fontsize=9)

        ax.set_ylabel("Annualised Return", color="white")
        ax.set_xlabel("Quantile", color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Cumulative return chart
        st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-chart-line ae-icon"></i>Cumulative Returns by Quantile</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        palette = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_quantiles))
        for i, col in enumerate(qr.quantile_cum_returns.columns):
            ax.plot(qr.quantile_cum_returns.index, qr.quantile_cum_returns[col],
                    label=col, color=palette[i], linewidth=1.5)

        # L/S spread as dashed black
        ax.plot(qr.long_short_cum.index, qr.long_short_cum.values,
                label=f"L/S (Q{n_quantiles}-Q1)", color="white", linewidth=2, linestyle="--")
        ax.axhline(1.0, color="#555", linewidth=0.5)

        ax.set_ylabel("Cumulative Return (1 = start)", color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", colors="white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9, ncol=3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Per-quantile stats table
        st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-table ae-icon"></i>Per-Quantile Statistics</h3>', unsafe_allow_html=True)
        stats_df = pd.DataFrame({
            "Annual Return": qr.mean_returns.map(lambda x: f"{x:.1%}"),
            "Sharpe Ratio": qr.quantile_sharpes.map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—"),
            "Hit Rate": qr.hit_rates.map(lambda x: f"{x:.1%}"),
        })
        st.dataframe(stats_df, width="stretch")

        # Long-short spread chart
        st.markdown(f'<h3 class="ae-sub"><i class="fa-solid fa-arrow-right-arrow-left ae-icon"></i>Long-Short Spread (Q{n_quantiles} − Q1) Cumulative Return</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        ax.plot(qr.long_short_cum.index, qr.long_short_cum.values, color="#f39c12", linewidth=1.5)
        ax.axhline(1.0, color="#555", linewidth=0.5, linestyle="--")
        ax.fill_between(qr.long_short_cum.index, 1.0, qr.long_short_cum.values,
                        where=qr.long_short_cum.values >= 1.0, alpha=0.15, color="#2ecc71")
        ax.fill_between(qr.long_short_cum.index, 1.0, qr.long_short_cum.values,
                        where=qr.long_short_cum.values < 1.0, alpha=0.15, color="#e74c3c")

        ax.set_ylabel("Cum. Return", color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", colors="white")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Auto-save to leaderboard (overwrite only if IC IR improves)
    # -----------------------------------------------------------------------
    try:
        from data.factor_research_log import save_result
        _run = st.session_state.get("last_factor_run", {})
        _auto_record = {
            "factor_name": factor_name,
            "lookback": factor_kwargs.get("lookback", ""),
            "skip": factor_kwargs.get("skip", ""),
            "ic_horizon": ic_horizon,
            "universe": universe,
            "universe_size": universe_size,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "n_quantiles": n_quantiles,
            "ic_mean": float(ic_summary.get("ic_mean", float("nan"))),
            "ic_std": float(ic_summary.get("ic_std", float("nan"))),
            "ic_tstat": float(ic_summary.get("ic_tstat", float("nan"))),
            "ic_ir": float(ic_summary.get("ic_ir", float("nan"))),
            "pct_positive_ic": float(ic_summary.get("ic_positive_pct", float("nan"))),
            "observations": int(ic_summary.get("obs", 0)),
            "spread_annual_return": float(qr.spread_annual_return),
            "monotonicity_score": float(qr.monotonicity_score),
            "factor_kwargs": factor_kwargs,
        }
        _saved = save_result(_auto_record, update_if_better=True)
        if _saved:
            st.toast(f"Leaderboard updated: {factor_name}", icon=None)
    except Exception as _e:
        pass  # auto-save is best-effort; don't block the UI

else:
    st.info("Configure factor settings in the sidebar and click **▶ Run Factor Analysis** to begin.")

    st.markdown("""
    ### What does this page analyse?

    | Analysis | What it tells you |
    |---|---|
    | **IC (Information Coefficient)** | How well the factor predicts next-period returns (Spearman rank correlation) |
    | **IC t-statistic** | Statistical significance of the factor edge |
    | **Rolling IC** | Whether the factor is stable or regime-dependent |
    | **Factor Decay** | How quickly the signal fades — guides rebalancing frequency |
    | **Quantile Portfolios** | Does the factor create monotonic return separation across the distribution? |
    | **Monotonicity Score** | 1.0 = perfect monotone ordering; < 0 = inverted factor |
    | **L/S Spread** | Pure alpha of going long top-quantile and short bottom-quantile |

    ### Tips
    - **IC > 0.05** is considered meaningful in practice for cross-sectional equity factors
    - **|IC t-stat| > 2.0** = statistically significant at ~95% confidence
    - A factor with high IC at t+1 but near-zero at t+20 is a **mean-reversion** factor
    - A factor with persistent IC through t+20 is a **momentum** factor
    - Use the quantile chart to check for **monotonicity** — if Q3 > Q1 but Q5 < Q4, the factor is noisy
    """)
