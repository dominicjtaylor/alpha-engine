"""
Quantile Portfolio Analysis.

Sorts the cross-section of stocks into N quantile buckets based on a factor
value, then tracks the average return of each bucket over the holding period.
A useful factor should exhibit return monotonicity: Q1 (lowest factor) < Q2
< ... < QN (highest factor).  The long-short spread (QN minus Q1) captures
the factor's edge.

Results are returned as a ``QuantileResult`` dataclass that bundles all
computed series for easy downstream consumption (charts, metrics, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MIN_CS_STD = 1e-8


@dataclass
class QuantileResult:
    """Results of a quantile portfolio analysis."""

    n_quantiles: int
    quantile_returns: pd.DataFrame    # DatetimeIndex × quantile label (Q1, Q2, …)
    quantile_cum_returns: pd.DataFrame
    long_short_returns: pd.Series     # QN minus Q1 daily return
    long_short_cum: pd.Series
    mean_returns: pd.Series           # annualised mean return per quantile
    hit_rates: pd.Series              # fraction of positive return days per quantile
    quantile_sharpes: pd.Series       # annualised Sharpe per quantile
    monotonicity_score: float         # Spearman rank corr of quantile labels vs mean returns
    spread_annual_return: float
    spread_sharpe: float
    metadata: dict = field(default_factory=dict)


def compute_quantile_portfolios(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
    long_quantile: int | None = None,
    short_quantile: int | None = None,
) -> QuantileResult:
    """
    Sort the cross-section into ``n_quantiles`` buckets each day and track
    the equal-weighted return of each bucket over the next period.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values — wide format (DatetimeIndex × ticker).
        Higher values are assumed to be more positive signals (e.g. momentum).
    forward_returns : pd.DataFrame
        Single-period forward returns — wide format.
        Typically ``prices.pct_change().shift(-1)``.
    n_quantiles : int
        Number of quantile buckets (default 5 = quintiles).
    long_quantile : int | None
        Which quantile to use as the "long" leg for L/S spread.
        Defaults to ``n_quantiles`` (highest factor = long).
    short_quantile : int | None
        Which quantile to use as the "short" leg.
        Defaults to ``1`` (lowest factor = short).

    Returns
    -------
    QuantileResult
        Dataclass with quantile return series, cumulative returns,
        long-short spread, and summary statistics.
    """
    if long_quantile is None:
        long_quantile = n_quantiles
    if short_quantile is None:
        short_quantile = 1

    # Align
    common_dates = factor.index.intersection(forward_returns.index)
    common_tickers = factor.columns.intersection(forward_returns.columns)
    f = factor.loc[common_dates, common_tickers]
    r = forward_returns.loc[common_dates, common_tickers]

    labels = [f"Q{i}" for i in range(1, n_quantiles + 1)]
    quantile_ret_list: list[pd.Series] = []
    n_skipped_constant = 0

    for date in common_dates:
        f_row = f.loc[date].dropna()
        r_row = r.loc[date].dropna()
        shared = f_row.index.intersection(r_row.index)

        if len(shared) < n_quantiles * 2:
            quantile_ret_list.append(pd.Series(np.nan, index=labels, name=date))
            continue

        # Skip dates where the factor is constant — ranking is undefined and
        # every stock would collapse into the same quantile.
        if f_row[shared].std() < _MIN_CS_STD:
            n_skipped_constant += 1
            quantile_ret_list.append(pd.Series(np.nan, index=labels, name=date))
            continue

        ranked = f_row[shared].rank(pct=True)
        # Assign to quantile 1..n_quantiles
        quantile_assignments = np.ceil(ranked * n_quantiles).clip(1, n_quantiles).astype(int)

        q_returns: dict[str, float] = {}
        for q in range(1, n_quantiles + 1):
            mask = quantile_assignments == q
            tickers_in_q = quantile_assignments[mask].index
            if len(tickers_in_q) == 0:
                q_returns[f"Q{q}"] = np.nan
            else:
                q_returns[f"Q{q}"] = float(r_row[shared][tickers_in_q].mean())

        quantile_ret_list.append(pd.Series(q_returns, name=date))

    log.debug(
        "Quantile portfolios: %d/%d dates valid, %d skipped (constant cross-section)",
        len(common_dates) - n_skipped_constant, len(common_dates), n_skipped_constant,
    )
    if len(common_dates) > 0 and n_skipped_constant / len(common_dates) > 0.10:
        log.warning(
            "%.1f%% of dates skipped in quantile analysis due to constant factor values. "
            "Factor may lack cross-sectional dispersion.",
            100.0 * n_skipped_constant / len(common_dates),
        )

    quantile_returns = pd.DataFrame(quantile_ret_list).dropna(how="all")
    quantile_returns.index = pd.DatetimeIndex(quantile_returns.index)

    # Cumulative returns
    quantile_cum_returns = (1 + quantile_returns.fillna(0)).cumprod()

    # Long-short spread
    long_col = f"Q{long_quantile}"
    short_col = f"Q{short_quantile}"
    long_short_returns = (quantile_returns[long_col] - quantile_returns[short_col]).dropna()
    long_short_cum = (1 + long_short_returns).cumprod()

    # Summary statistics (annualised)
    mean_returns = quantile_returns.mean() * 252
    hit_rates = (quantile_returns > 0).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        vol = quantile_returns.std() * np.sqrt(252)
        quantile_sharpes = mean_returns / vol.replace(0, np.nan)

    # Monotonicity: Spearman rank correlation of quantile label (1..N) vs mean return
    from scipy.stats import spearmanr
    q_labels_numeric = np.arange(1, n_quantiles + 1)
    q_mean_vals = mean_returns.values
    if np.any(np.isnan(q_mean_vals)):
        monotonicity = float("nan")
    else:
        mono_corr, _ = spearmanr(q_labels_numeric, q_mean_vals)
        monotonicity = float(mono_corr)

    # L/S spread stats
    ls_annual = float(long_short_returns.mean() * 252) if len(long_short_returns) > 0 else float("nan")
    ls_vol = float(long_short_returns.std() * np.sqrt(252)) if len(long_short_returns) > 1 else float("nan")
    ls_sharpe = ls_annual / ls_vol if ls_vol and ls_vol > 0 else float("nan")

    return QuantileResult(
        n_quantiles=n_quantiles,
        quantile_returns=quantile_returns,
        quantile_cum_returns=quantile_cum_returns,
        long_short_returns=long_short_returns,
        long_short_cum=long_short_cum,
        mean_returns=mean_returns,
        hit_rates=hit_rates,
        quantile_sharpes=quantile_sharpes,
        monotonicity_score=monotonicity,
        spread_annual_return=ls_annual,
        spread_sharpe=ls_sharpe,
        metadata={
            "n_dates": len(common_dates),
            "n_tickers": len(common_tickers),
            "long_quantile": long_quantile,
            "short_quantile": short_quantile,
        },
    )
