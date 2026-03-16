"""
Alpha Engine — entry point.

Defines multi-page navigation order. All page content lives in pages/.
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

# Font Awesome + shared heading styles (loaded once, available to all pages)
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
.ae-title  { font-size:2rem;  font-weight:700; margin:0 0 0.2rem; line-height:1.25; }
.ae-header { font-size:1.45rem; font-weight:600; margin:1.1rem 0 0.3rem; }
.ae-sub    { font-size:1.1rem;  font-weight:600; margin:0.8rem 0 0.2rem; }
.ae-icon   { margin-right:8px; opacity:0.9; }
.ae-badge-green  { color:#22c55e; font-weight:700; }
.ae-badge-yellow { color:#f59e0b; font-weight:700; }
.ae-badge-red    { color:#ef4444; font-weight:700; }
</style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/factor_research.py", title="Factor Research", icon="🔬"),
    st.Page("pages/leaderboard.py", title="Leaderboard", icon="🏆"),
    st.Page("pages/backtest.py", title="Backtest", icon="📊"),
    st.Page("pages/paper_trading.py", title="Paper Trading", icon="📝"),
    st.Page("pages/help.py", title="Help", icon="❓"),
])
pg.run()
