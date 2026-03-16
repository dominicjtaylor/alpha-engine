"""
Information Coefficient (IC) analysis for alpha factors.

IC measures the rank correlation between a factor's values on day t and
the realised return over the next n days.  Rank (Spearman) correlation is
preferred over Pearson because it is robust to outliers and non-normality.

All computations operate on **wide DataFrames** (DatetimeIndex × ticker).
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# Minimum cross-sectional standard deviation to consider a date valid.
# Below this threshold the factor (or return) is effectively constant and
# spearmanr raises ConstantInputWarning.
_MIN_CS_STD = 1e-8


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

    n_total = len(common_dates)
    n_skipped_short = 0      # too few assets
    n_skipped_constant = 0   # constant factor or return cross-section

    for date in common_dates:
        f_row = f.loc[date].dropna()
        r_row = r.loc[date].dropna()
        shared = f_row.index.intersection(r_row.index)

        if len(shared) < 5:
            n_skipped_short += 1
            continue

        f_vals = f_row[shared].values
        r_vals = r_row[shared].values

        # Skip dates where the factor or returns are constant — spearmanr raises
        # ConstantInputWarning and returns NaN for these, which pollutes IC stats.
        if np.std(f_vals) < _MIN_CS_STD or np.std(r_vals) < _MIN_CS_STD:
            n_skipped_constant += 1
            continue

        if method == "spearman":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = stats.spearmanr(f_vals, r_vals)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = np.corrcoef(f_vals, r_vals)[0, 1]

        if np.isfinite(corr):
            ic_values.append(float(corr))
            dates.append(date)

    # Diagnostics
    n_valid = len(ic_values)
    log.debug(
        "IC [%s]: %d/%d dates valid, %d skipped (too few assets), "
        "%d skipped (constant cross-section)",
        method, n_valid, n_total, n_skipped_short, n_skipped_constant,
    )
    if n_total > 0 and n_skipped_constant / n_total > 0.10:
        log.warning(
            "%.1f%% of dates skipped due to constant factor/return values "
            "(%d/%d dates). Check factor generation for aggregation bugs, "
            "look-ahead bias, or data quality issues.",
            100.0 * n_skipped_constant / n_total,
            n_skipped_constant, n_total,
        )

    return pd.Series(ic_values, index=pd.DatetimeIndex(dates), name="IC")


def check_factor_dispersion(
    factor: pd.DataFrame,
    min_cs_std: float = _MIN_CS_STD,
) -> dict:
    """
    Audit cross-sectional and time-series variation in a factor DataFrame.

    Called after factor generation to surface constant-signal issues before
    IC / quantile analysis begins.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (index=date, columns=tickers).
    min_cs_std : float
        Minimum cross-sectional std to consider a date valid.

    Returns
    -------
    dict
        Keys: n_dates, n_assets_mean, mean_cs_std, pct_constant_dates,
        n_constant_dates, passed (bool).
        Logs a WARNING if ``pct_constant_dates > 0.10`` or ``mean_cs_std`` is
        near zero.
    """
    cs_std = factor.std(axis=1)  # std across tickers per date
    n_constant = int((cs_std < min_cs_std).sum())
    n_dates = len(cs_std.dropna())
    mean_cs_std = float(cs_std.mean())
    pct_constant = n_constant / max(n_dates, 1)

    n_assets_mean = float(factor.notna().sum(axis=1).mean())

    result = {
        "n_dates": n_dates,
        "n_assets_mean": n_assets_mean,
        "mean_cs_std": mean_cs_std,
        "n_constant_dates": n_constant,
        "pct_constant_dates": pct_constant,
        "passed": pct_constant <= 0.10 and mean_cs_std >= min_cs_std,
    }

    log.debug(
        "Factor dispersion: %d dates, %.1f assets/date, mean CS std=%.4f, "
        "%d constant dates (%.1f%%)",
        n_dates, n_assets_mean, mean_cs_std, n_constant, 100 * pct_constant,
    )
    if not result["passed"]:
        log.warning(
            "Factor has poor cross-sectional dispersion: mean CS std=%.2e, "
            "%d/%d dates constant (%.1f%%). "
            "IC and quantile results will be unreliable.",
            mean_cs_std, n_constant, n_dates, 100 * pct_constant,
        )

    return result


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
