"""
Information Coefficient (IC) analysis for alpha factors.

IC measures the rank correlation between a factor's values on day t and
the realised return over the next n days.  Rank (Spearman) correlation is
preferred over Pearson because it is robust to outliers and non-normality.

All computations operate on **wide DataFrames** (DatetimeIndex × ticker).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    Compute the daily Information Coefficient between a factor and forward returns.

    For each date we rank both the factor values and the forward returns
    cross-sectionally, then compute their correlation.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values — wide format (DatetimeIndex × ticker).
    forward_returns : pd.DataFrame
        Forward returns over a horizon — wide format (DatetimeIndex × ticker).
        Typically ``prices.pct_change().shift(-horizon)`` for horizon-day IC.
    method : str
        Correlation method: ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    pd.Series
        Daily IC values indexed by date.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    # Align on common dates and tickers
    common_dates = factor.index.intersection(forward_returns.index)
    common_tickers = factor.columns.intersection(forward_returns.columns)
    f = factor.loc[common_dates, common_tickers]
    r = forward_returns.loc[common_dates, common_tickers]

    ic_values: list[float] = []
    dates: list = []

    for date in common_dates:
        f_row = f.loc[date].dropna()
        r_row = r.loc[date].dropna()
        shared = f_row.index.intersection(r_row.index)
        if len(shared) < 5:
            continue

        f_vals = f_row[shared].values
        r_vals = r_row[shared].values

        if method == "spearman":
            corr, _ = stats.spearmanr(f_vals, r_vals)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = np.corrcoef(f_vals, r_vals)[0, 1]

        if np.isfinite(corr):
            ic_values.append(float(corr))
            dates.append(date)

    return pd.Series(ic_values, index=pd.DatetimeIndex(dates), name="IC")


def compute_rolling_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    window: int = 63,
    method: str = "spearman",
) -> pd.Series:
    """
    Compute rolling mean IC over a trailing window.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (wide format).
    forward_returns : pd.DataFrame
        Forward returns (wide format).
    window : int
        Rolling window in trading days (default 63 ≈ 1 quarter).
    method : str
        Correlation method — ``"spearman"`` or ``"pearson"``.

    Returns
    -------
    pd.Series
        Rolling IC indexed by date.
    """
    daily_ic = compute_ic(factor, forward_returns, method=method)
    return daily_ic.rolling(window, min_periods=max(window // 4, 5)).mean().rename("Rolling IC")


def compute_ic_summary(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> dict[str, float]:
    """
    Compute summary IC statistics: mean, std, t-statistic, and annualised ICIR.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (wide format).
    forward_returns : pd.DataFrame
        Forward returns (wide format).
    method : str
        Correlation method — ``"spearman"`` or ``"pearson"``.

    Returns
    -------
    dict[str, float]
        Keys: ``ic_mean``, ``ic_std``, ``ic_tstat``, ``ic_ir``,
        ``ic_positive_pct``, ``obs``.
    """
    daily_ic = compute_ic(factor, forward_returns, method=method)

    if daily_ic.empty:
        return {k: float("nan") for k in ("ic_mean", "ic_std", "ic_tstat", "ic_ir", "ic_positive_pct", "obs")}

    n = len(daily_ic)
    ic_mean = float(daily_ic.mean())
    ic_std = float(daily_ic.std(ddof=1))

    tstat = (ic_mean / ic_std * np.sqrt(n)) if ic_std > 0 else float("nan")
    icir = (ic_mean / ic_std * np.sqrt(252)) if ic_std > 0 else float("nan")  # annualised ICIR
    positive_pct = float((daily_ic > 0).mean())

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_tstat": tstat,
        "ic_ir": icir,
        "ic_positive_pct": positive_pct,
        "obs": float(n),
    }
