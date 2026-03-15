"""
Alpha Engine — Home

Entry point for the Streamlit multi-page app.
Navigation is available in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Alpha Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Alpha Engine")
st.subheader("UK Systematic Trading Research Framework")

st.markdown("""
A modular research platform for backtesting and simulating evidence-based
equity strategies on **London Stock Exchange (LSE)** securities.

All performance is reported in **GBP (£)**.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Backtest
    Run historical simulations on FTSE 100 / 250 equities.
    Compare strategies, tune parameters, and analyse performance
    with realistic UK transaction costs and stamp duty.

    → Use the **Backtest** page in the sidebar
    """)

with col2:
    st.markdown("""
    ### 📝 Paper Trading
    Simulate live strategy execution with fake money.
    Generate today's signals from the latest LSE prices,
    preview the rebalance, and track your paper P&L over time.

    → Use the **Paper Trading** page in the sidebar
    """)

st.divider()

st.markdown("""
### Strategies

| Strategy | Signal | Rebalance |
|---|---|---|
| **Momentum (12-1)** | 12-month return excl. last month | Monthly |
| **Mean Reversion (5-day)** | Contrarian 5-day return | Daily |
| **Earnings Drift** | Overnight gap > 5% (PEAD proxy) | Event-driven |

### UK Market Details
- Tickers use **`.L` suffix** (Yahoo Finance LSE convention)
- Trading calendar: **LSE** (excludes UK bank holidays)
- Transaction costs: **commission + slippage + 0.5% SDRT** on long purchases
- Default leverage cap: **1.5×**
""")
