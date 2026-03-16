"""
Factor Leaderboard — Alpha Engine

Ranked table of all saved factor analyses with robustness badges,
visual summaries, and one-click promotion to the Backtest page.
"""

from __future__ import annotations

import json
import sys
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



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal_badge(ic_tstat: float, ic_mean: float) -> str:
    """Return a robustness badge string based on t-stat and IC mean."""
    try:
        t = float(ic_tstat)
        m = float(ic_mean)
    except (TypeError, ValueError):
        return "Unknown"
    if abs(t) >= 2.5 and m > 0.02:
        return "Robust"
    if abs(t) >= 2.0 or (abs(t) >= 1.5 and m > 0.015):
        return "Promising"
    return "Weak"


def _factor_label(row) -> str:
    """
    Build a rich display label for a leaderboard row that uniquely identifies
    the factor configuration.

    Format: ``factor_name | lb=N | skip=N | h=Nd | IC IR=N.NN``
    """
    name = str(row.get("factor_name", ""))

    parts = [name]

    lb = row.get("lookback", "")
    try:
        lb_i = int(float(lb))
        if lb_i > 0:
            parts.append(f"lb={lb_i}")
    except (TypeError, ValueError):
        pass

    skip = row.get("skip", "")
    try:
        skip_i = int(float(skip))
        if skip_i > 0:
            parts.append(f"skip={skip_i}")
    except (TypeError, ValueError):
        pass

    h = row.get("ic_horizon", "")
    try:
        h_i = int(float(h))
        if h_i > 0:
            parts.append(f"h={h_i}d")
    except (TypeError, ValueError):
        pass

    try:
        ir = float(row.get("ic_ir", float("nan")))
        if np.isfinite(ir):
            parts.append(f"IC IR={ir:.2f}")
    except (TypeError, ValueError):
        pass

    return " | ".join(parts)


def _fmt_f3(v) -> str:
    try:
        return f"{float(v):.3f}" if np.isfinite(float(v)) else "—"
    except Exception:
        return "—"


def _fmt_f2(v) -> str:
    try:
        return f"{float(v):.2f}" if np.isfinite(float(v)) else "—"
    except Exception:
        return "—"


def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.1%}" if np.isfinite(float(v)) else "—"
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

st.markdown('<h1 class="ae-title"><i class="fa-solid fa-trophy ae-icon"></i>Factor Leaderboard</h1>', unsafe_allow_html=True)
st.markdown("All saved factor analyses, ranked by statistical robustness.")

try:
    from data.factor_research_log import load_leaderboard, clear_leaderboard
    df_raw = load_leaderboard()
except Exception as exc:
    st.error(f"Could not load leaderboard: {exc}")
    st.stop()

if df_raw.empty:
    st.info(
        "No factor results saved yet.\n\n"
        "Run a factor analysis on the **Factor Research** page and click "
        "Click **Save to Leaderboard** to populate this table."
    )
    st.stop()

df = df_raw.copy()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

total_factors = len(df)
sig_count = int((df["ic_tstat"].abs() >= 2.0).sum()) if "ic_tstat" in df.columns else 0
best_tstat_row = df.loc[df["ic_tstat"].abs().idxmax()] if "ic_tstat" in df.columns else None
best_tstat = best_tstat_row["ic_tstat"] if best_tstat_row is not None else float("nan")
best_factor_name = best_tstat_row["factor_name"] if best_tstat_row is not None else "—"

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Total Factors Saved",
    total_factors,
    help="Number of unique factor configurations saved to the leaderboard.",
)
k2.metric(
    "Statistically Significant",
    sig_count,
    help="Factors with |IC t-stat| ≥ 2.0 (roughly 95% confidence the signal is real).",
)
k3.metric(
    "Best t-stat",
    _fmt_f2(best_tstat),
    help=f"Highest absolute IC t-statistic across all saved factors. Factor: {best_factor_name}",
)
k4.metric(
    "Significant Rate",
    f"{sig_count / total_factors:.0%}" if total_factors else "—",
    help="Fraction of tested factors that reached statistical significance.",
)

st.divider()

# ---------------------------------------------------------------------------
# Sort controls
# ---------------------------------------------------------------------------

sort_col, filter_col = st.columns([2, 3])

with sort_col:
    sort_by = st.radio(
        "Sort by",
        ["IC IR", "IC t-stat", "IC Mean", "% Positive IC", "L/S Spread"],
        horizontal=True,
    )

_sort_map = {
    "IC IR": ("ic_ir", False, None),
    "IC t-stat": ("ic_tstat", True, lambda x: x.abs()),
    "IC Mean": ("ic_mean", False, None),
    "% Positive IC": ("pct_positive_ic", False, None),
    "L/S Spread": ("spread_annual_return", False, None),
}

sort_key, sort_ascending, sort_key_fn = _sort_map[sort_by]
if sort_key in df.columns:
    if sort_key_fn is not None:
        df = df.sort_values(sort_key, key=sort_key_fn, ascending=sort_ascending)
    else:
        df = df.sort_values(sort_key, ascending=sort_ascending)

with filter_col:
    factor_filter = st.multiselect(
        "Filter by factor name",
        options=sorted(df["factor_name"].unique()),
        default=[],
        help="Leave blank to show all factors.",
    )
    if factor_filter:
        df = df[df["factor_name"].isin(factor_filter)]

if df.empty:
    st.warning("No factors match the current filter.")
    st.stop()

# ---------------------------------------------------------------------------
# Charts — scatter + top-10 bar (side by side)
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-circle-dot ae-icon"></i>IC Mean vs |t-stat|</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors_scatter = []
    for _, row in df.iterrows():
        badge = _signal_badge(row.get("ic_tstat", 0), row.get("ic_mean", 0))
        if badge == "Robust":
            colors_scatter.append("#2ecc71")
        elif badge == "Promising":
            colors_scatter.append("#f39c12")
        else:
            colors_scatter.append("#e74c3c")

    ic_means = df["ic_mean"].fillna(0).values
    ic_tstats = df["ic_tstat"].abs().fillna(0).values

    ax.scatter(ic_means, ic_tstats, c=colors_scatter, s=80, alpha=0.85, edgecolors="#333", linewidth=0.5)
    ax.axvline(0, color="#555", linewidth=0.7, linestyle="--")
    ax.axhline(2.0, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.7, label="|t| = 2.0")

    # Label points
    for i, row in df.iterrows():
        ax.annotate(
            row.get("factor_name", "")[:14],
            (row.get("ic_mean", 0), abs(row.get("ic_tstat", 0))),
            fontsize=6, color="white", alpha=0.7,
            xytext=(3, 3), textcoords="offset points",
        )

    ax.set_xlabel("IC Mean", color="white")
    ax.set_ylabel("|IC t-stat|", color="white")
    ax.tick_params(colors="white")
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax.set_title("Factor Cloud", color="white", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-ranking-star ae-icon"></i>Top 10 by |IC t-stat|</h3>', unsafe_allow_html=True)
    top10 = df.nlargest(10, "ic_tstat", keep="first") if "ic_tstat" in df.columns else df.head(10)
    top10 = top10[::-1]  # flip for horizontal bar chart (highest at top)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")

    bar_colors = []
    for _, row in top10.iterrows():
        badge = _signal_badge(row.get("ic_tstat", 0), row.get("ic_mean", 0))
        if badge == "Robust":
            bar_colors.append("#2ecc71")
        elif badge == "Promising":
            bar_colors.append("#f39c12")
        else:
            bar_colors.append("#e74c3c")

    labels = top10["factor_name"].str[:18].tolist()
    values = top10["ic_tstat"].abs().tolist()

    bars = ax2.barh(labels, values, color=bar_colors, alpha=0.85, edgecolor="#333")
    ax2.axvline(2.0, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.7)
    ax2.set_xlabel("|IC t-stat|", color="white")
    ax2.tick_params(colors="white")
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_color("#444")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("Top 10 Factors", color="white", fontsize=11)

    for bar, val in zip(bars, values):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", color="white", fontsize=8)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ---------------------------------------------------------------------------
# Full table with badges
# ---------------------------------------------------------------------------

st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-table ae-icon"></i>All Saved Factors</h3>', unsafe_allow_html=True)

display_df = pd.DataFrame()
display_df["Rank"] = range(1, len(df) + 1)
display_df["Signal"] = df.apply(
    lambda r: _signal_badge(r.get("ic_tstat", 0), r.get("ic_mean", 0)), axis=1
)
display_df["Factor"] = df["factor_name"]

def _int_or_dash(v) -> str:
    try:
        i = int(float(v))
        return str(i) if i > 0 else "—"
    except (TypeError, ValueError):
        return "—"

display_df["Lookback"] = df["lookback"].map(_int_or_dash)
display_df["Skip"] = df["skip"].map(_int_or_dash)
INDEX_DISPLAY_NAMES = {"ftse100": "FTSE 100", "ftse250": "FTSE 250", "ftse_all": "FTSE All-Share"}
display_df["Universe"] = df.get("universe", pd.Series(["—"]*len(df), index=df.index)).map(lambda x: INDEX_DISPLAY_NAMES.get(x, x) if isinstance(x, str) else x)
display_df["IC Mean"] = df["ic_mean"].map(_fmt_f3)
display_df["IC Std"] = df["ic_std"].map(_fmt_f3)
display_df["t-stat"] = df["ic_tstat"].map(_fmt_f2)
display_df["IC IR"] = df["ic_ir"].map(_fmt_f2)
display_df["% Positive"] = df["pct_positive_ic"].map(_fmt_pct)
display_df["L/S Spread"] = df["spread_annual_return"].map(_fmt_pct)
display_df["Mono. Score"] = df["monotonicity_score"].map(_fmt_f2)
display_df["IC Horizon"] = df.get("ic_horizon", "—").astype(str) + "d"
display_df["Obs"] = df["observations"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
display_df["Tested"] = pd.to_datetime(df.get("timestamp", pd.NaT)).dt.strftime("%Y-%m-%d").fillna("—")

st.dataframe(display_df, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Promote to Backtest
# ---------------------------------------------------------------------------

st.divider()
st.markdown('<h3 class="ae-sub"><i class="fa-solid fa-rocket ae-icon"></i>Use in Backtest</h3>', unsafe_allow_html=True)
st.markdown(
    "Select a specific factor **configuration** and send it to the Backtest page. "
    "Each row represents a unique combination of factor type, lookback, and horizon."
)

# Build config_hash → row map.  df is already sorted by IC IR descending so the
# best-performing configuration appears first in the selector.
_promote_map: dict[str, pd.Series] = {}
for _, _prow in df.iterrows():
    _ph = str(_prow.get("config_hash", ""))
    if _ph:
        _promote_map[_ph] = _prow

if not _promote_map:
    st.info("No factor configurations with a valid config hash found. Re-run Factor Research.")
else:
    promo_col1, promo_col2 = st.columns([3, 1])

    with promo_col1:
        selected_hash = st.selectbox(
            "Factor configuration",
            list(_promote_map.keys()),
            format_func=lambda h: _factor_label(_promote_map[h]),
            key="promote_factor",
        )

    with promo_col2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        promote_btn = st.button("Use in Backtest", type="primary")

    if promote_btn and selected_hash:
        _promo_row = _promote_map[selected_hash]
        raw_kw = _promo_row.get("factor_kwargs", "{}")
        try:
            factor_kwargs = json.loads(raw_kw) if isinstance(raw_kw, str) else {}
        except Exception:
            factor_kwargs = {}

        st.session_state["factor_from_leaderboard"] = {
            "factor_name": str(_promo_row.get("factor_name", "")),
            "factor_kwargs": factor_kwargs,
            "config_hash": selected_hash,
            "label": _factor_label(_promo_row),
        }
        st.switch_page("pages/backtest.py")

# ---------------------------------------------------------------------------
# Danger zone
# ---------------------------------------------------------------------------

with st.expander("Danger Zone", expanded=False):
    st.markdown("This will permanently delete all saved factor results.")
    if st.button("Clear Leaderboard", type="secondary"):
        clear_leaderboard()
        st.success("Leaderboard cleared. Reload the page to see the empty state.")
        st.rerun()
