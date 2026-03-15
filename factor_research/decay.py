"""
Factor Decay Analysis.

Measures how quickly a factor's predictive power fades by computing IC
at multiple forward-return horizons (t+1, t+5, t+10, t+20 days).

A factor with persistent alpha should have declining but still-positive IC
at longer horizons.  A factor that decays quickly is suitable only for
short-holding-period strategies (e.g. mean reversion), while a slow-decaying
factor is more appropriate for monthly rebalancing (e.g. momentum).
"""

from __future__ import annotations

import pandas as pd

from factor_research.ic import compute_ic, compute_ic_summary


def compute_factor_decay(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute IC at multiple forward-return horizons to assess factor decay.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values — wide format (DatetimeIndex × ticker).
    prices : pd.DataFrame
        Adjusted close prices — wide format, same structure as factor.
    horizons : tuple[int, ...]
        Forward horizons in trading days.
        Default: (1, 5, 10, 20) — daily, weekly, bi-weekly, monthly.
    method : str
        Correlation method for IC: ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    pd.DataFrame
        Rows = horizons, columns = ``ic_mean``, ``ic_std``, ``ic_tstat``,
        ``ic_ir``, ``ic_positive_pct``, ``obs``.
        Index is the horizon in days.
    """
    rows: list[dict] = []

    for h in horizons:
        fwd_returns = prices.pct_change(h).shift(-h)
        summary = compute_ic_summary(factor, fwd_returns, method=method)
        summary["horizon"] = h
        rows.append(summary)

    df = pd.DataFrame(rows).set_index("horizon")
    df.index.name = "horizon_days"
    return df


def compute_decay_series(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    method: str = "spearman",
) -> dict[int, pd.Series]:
    """
    Return the full daily IC series at each horizon.

    Useful for plotting IC time-series at different forecast horizons
    to visualise stability and regime variation.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (wide format).
    prices : pd.DataFrame
        Adjusted close prices (wide format).
    horizons : tuple[int, ...]
        Forward horizons in trading days.
    method : str
        Correlation method for IC.

    Returns
    -------
    dict[int, pd.Series]
        ``{horizon: daily_ic_series}`` for each horizon.
    """
    result: dict[int, pd.Series] = {}
    for h in horizons:
        fwd_returns = prices.pct_change(h).shift(-h)
        result[h] = compute_ic(factor, fwd_returns, method=method)
    return result
