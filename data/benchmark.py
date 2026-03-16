"""
Passive benchmark portfolio for the UK Systematic Trading Research Framework.

Approximates a Vanguard-style global allocation using US-listed ETF proxies
available on Yahoo Finance. Returns are USD-denominated total returns.

Note: The strategy portfolio is GBP-denominated (LSE equities). The benchmark
is USD-denominated. For relative metrics (Sharpe, alpha, IR), the currency
difference matters less than for absolute GBP return comparisons. Use the
relative metrics (alpha, tracking error, IR) as the primary comparison tool.

Annual rebalancing: on the first trading day of each calendar year, the
portfolio is reset to the target weights. Within each year, weights drift
with market returns (buy-and-hold). This is fully vectorized per year.

Benchmark allocation (Vanguard-style global):
    VTI  (US Total Market)           → 58.89%
    VGK  (Developed Europe)          → 13.22%
    BNDW (Global Aggregate Bonds)    → 10.00%
    VWO  (Emerging Markets)          →  9.17%
    EWJ  (Japan)                     →  5.12%
    VPL  (Asia Pacific ex-Japan)     →  3.60%
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Target weights (must sum to 1.0)
BENCHMARK_WEIGHTS: dict[str, float] = {
    "VTI": 0.5889,
    "VGK": 0.1322,
    "BNDW": 0.1000,
    "VWO": 0.0917,
    "EWJ": 0.0512,
    "VPL": 0.0360,
}

_CACHE_FILE = Path("data/cache/_benchmark.parquet")


def _download_etf_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices for all benchmark ETFs."""
    tickers = list(BENCHMARK_WEIGHTS.keys())
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logger.warning("Benchmark: missing ETF data for %s", missing)

    return prices.ffill(limit=5).dropna()


def _build_benchmark_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Compute daily benchmark returns with annual rebalancing.

    On the first trading day of each calendar year the portfolio is reset to
    the target weights. Within the year, weights drift with returns.
    The daily return is the dot product of the beginning-of-day weights
    (post-rebalance where applicable) and the ETF daily returns.

    This is vectorized within each annual period; the only loop is over years.
    """
    weights = pd.Series({t: w for t, w in BENCHMARK_WEIGHTS.items()
                         if t in prices.columns})
    weights = weights / weights.sum()  # renormalise for any missing ETFs

    etf_returns = prices.pct_change().dropna()
    parts: list[pd.Series] = []

    for year, yr_ret in etf_returns.groupby(etf_returns.index.year):
        # Vectorized dot product: each day's portfolio return
        common = weights.index.intersection(yr_ret.columns)
        w = weights[common]
        daily_ret = yr_ret[common].dot(w)
        parts.append(daily_ret)

    benchmark = pd.concat(parts).sort_index()
    benchmark.name = "Benchmark"
    return benchmark


def _is_cache_valid(start_date: str, end_date: str) -> bool:
    """Check whether the cached benchmark covers the requested range."""
    if not _CACHE_FILE.exists():
        return False
    try:
        df = pd.read_parquet(_CACHE_FILE)
        if df.empty:
            return False
        first = df.index.min()
        last = df.index.max()
        return (first <= pd.Timestamp(start_date) + pd.offsets.BDay(5) and
                last >= pd.Timestamp(end_date) - pd.offsets.BDay(5))
    except Exception:  # noqa: BLE001
        return False


def load_benchmark_returns(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
    """
    Load daily benchmark returns for the Vanguard-style passive portfolio.

    Parameters
    ----------
    start_date : str, optional
        Start date in ISO 8601 format. Defaults to '2010-01-01'.
    end_date : str, optional
        End date in ISO 8601 format. Defaults to today.

    Returns
    -------
    pd.Series
        Daily benchmark returns (DatetimeIndex). USD-denominated total return.
    """
    from datetime import date
    start = start_date or "2010-01-01"
    end = end_date or date.today().isoformat()

    if _is_cache_valid(start, end):
        df = pd.read_parquet(_CACHE_FILE)
        df.index = pd.to_datetime(df.index)
        series = df["returns"]
        return series[start:end]

    logger.info("Downloading benchmark ETF prices (%s → %s)", start, end)
    prices = _download_etf_prices(start, end)

    if prices.empty:
        logger.error("Benchmark: no ETF price data downloaded.")
        return pd.Series(dtype=float, name="Benchmark")

    benchmark = _build_benchmark_returns(prices)

    # Cache
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_frame("returns").to_parquet(_CACHE_FILE, compression="snappy")
    logger.info("Benchmark cached: %d days", len(benchmark))

    return benchmark[start:end]


def get_benchmark_equity_curve(benchmark_returns: pd.Series) -> pd.Series:
    """
    Convert daily benchmark returns to a normalised equity curve (base = 1.0).

    Parameters
    ----------
    benchmark_returns : pd.Series
        Daily benchmark returns.

    Returns
    -------
    pd.Series
        Cumulative equity curve starting at 1.0.
    """
    return (1 + benchmark_returns).cumprod()
