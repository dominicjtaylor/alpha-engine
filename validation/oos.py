"""
Out-of-Sample (OOS) Testing and Walk-Forward Validation.

Provides two rigorous testing frameworks to prevent overfitting:

1. **Single OOS test** — the simplest check: fit/tune on a training period,
   then measure performance on a held-out test period with no look-ahead.

2. **Walk-forward validation** — repeatedly rolls a training window forward
   and measures OOS performance in successive non-overlapping test periods.
   This tests whether the strategy edge is consistent over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from analytics.metrics import compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single OOS Test
# ---------------------------------------------------------------------------

@dataclass
class OOSResult:
    """Results from a single out-of-sample test."""

    train_returns: pd.Series
    test_returns: pd.Series
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    degradation: dict[str, float] = field(default_factory=dict)


def run_oos_test(
    weights_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    backtest_engine=None,
    risk_free_rate: float = 0.04,
) -> OOSResult:
    """
    Run a single out-of-sample test.

    The ``weights_fn`` callable is invoked on the **training** slice to
    produce weights, then those weights are evaluated on the **test** slice.

    Parameters
    ----------
    weights_fn : Callable[[prices, returns], weights_df]
        A function that takes price and return DataFrames and returns a
        portfolio weights DataFrame (DatetimeIndex × ticker).
        Typically a lambda wrapping a strategy's ``run()`` method.
    prices : pd.DataFrame
        Full adjusted close prices (wide format).
    returns : pd.DataFrame
        Full daily returns (wide format).
    train_start, train_end : str
        ISO date strings (YYYY-MM-DD) for the training period.
    test_start, test_end : str
        ISO date strings (YYYY-MM-DD) for the test period.
    backtest_engine : BacktestEngine | None
        If provided, uses this engine to compute net returns (with costs).
        If None, computes gross returns directly.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation.

    Returns
    -------
    OOSResult
        Training metrics, test metrics, and degradation ratios.
    """
    # Slice data
    train_prices = prices.loc[train_start:train_end]
    train_returns_df = returns.loc[train_start:train_end]
    test_prices = prices.loc[test_start:test_end]
    test_returns_df = returns.loc[test_start:test_end]

    # Generate weights using training data
    train_weights = weights_fn(train_prices, train_returns_df)
    # The strategy only outputs weights for the training period —
    # run the strategy again on full data to get test-period weights
    full_weights = weights_fn(prices.loc[train_start:test_end], returns.loc[train_start:test_end])
    test_weights = full_weights.loc[test_start:test_end]

    def _get_returns(w: pd.DataFrame, r: pd.DataFrame) -> pd.Series:
        if backtest_engine is not None:
            result = backtest_engine.run(w, r, "oos_eval")
            return result.daily_returns
        # Gross returns — simple dot product with execution lag
        w_exec = w.shift(1).reindex(r.index, method=None).fillna(0.0)
        common = w_exec.columns.intersection(r.columns)
        return (w_exec[common] * r[common]).sum(axis=1)

    # Compute returns for each period
    train_rets = _get_returns(train_weights, train_returns_df)
    test_rets = _get_returns(test_weights, test_returns_df)

    train_metrics = compute_metrics(train_rets, risk_free_rate=risk_free_rate)
    test_metrics = compute_metrics(test_rets, risk_free_rate=risk_free_rate)

    # Degradation: ratio of test metric to train metric
    # A ratio close to 1.0 is ideal; < 0 means sign flip (bad)
    degradation: dict[str, float] = {}
    for key in ("sharpe_ratio", "annual_return", "annual_volatility"):
        t_val = train_metrics.get(key, float("nan"))
        o_val = test_metrics.get(key, float("nan"))
        if t_val and abs(t_val) > 1e-9:
            degradation[key] = o_val / t_val
        else:
            degradation[key] = float("nan")

    logger.info(
        "OOS test complete | train Sharpe=%.2f | test Sharpe=%.2f | degradation=%.2f",
        train_metrics.get("sharpe_ratio", float("nan")),
        test_metrics.get("sharpe_ratio", float("nan")),
        degradation.get("sharpe_ratio", float("nan")),
    )

    return OOSResult(
        train_returns=train_rets,
        test_returns=test_rets,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        degradation=degradation,
    )


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Aggregated results from a walk-forward validation run."""

    folds: list[OOSResult]
    combined_oos_returns: pd.Series
    combined_oos_equity: pd.Series
    oos_metrics: dict[str, float]
    in_sample_metrics_by_fold: list[dict[str, float]]
    oos_metrics_by_fold: list[dict[str, float]]
    sharpe_consistency: float   # fraction of folds with positive OOS Sharpe
    metadata: dict = field(default_factory=dict)


def run_walk_forward(
    weights_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    train_months: int = 36,
    test_months: int = 12,
    expanding: bool = True,
    min_train_months: int = 24,
    backtest_engine=None,
    initial_capital: float = 1_000_000.0,
    risk_free_rate: float = 0.04,
) -> WalkForwardResult:
    """
    Run a walk-forward validation, rolling the training window forward
    and measuring OOS performance in successive non-overlapping test periods.

    Parameters
    ----------
    weights_fn : Callable[[prices, returns], weights_df]
        Strategy function — same interface as in ``run_oos_test``.
    prices : pd.DataFrame
        Full adjusted close prices (wide format).
    returns : pd.DataFrame
        Full daily returns (wide format).
    train_months : int
        Training window size in calendar months.
        Ignored if ``expanding=True`` (all history used).
    test_months : int
        OOS test period length in calendar months.
    expanding : bool
        If True, training window expands from a fixed start date.
        If False, uses a rolling window of ``train_months``.
    min_train_months : int
        Minimum training period required to run a fold.
    backtest_engine : BacktestEngine | None
        If provided, uses this engine for net return calculation.
    initial_capital : float
        Starting capital for equity curve calculation (GBP).
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation.

    Returns
    -------
    WalkForwardResult
    """
    dates = prices.index
    if len(dates) == 0:
        raise ValueError("prices DataFrame is empty")

    start_date = dates[0]
    end_date = dates[-1]

    # Build fold boundaries
    folds_raw: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    test_start = start_date + pd.DateOffset(months=train_months)
    while test_start <= end_date:
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end_date)

        if expanding:
            train_s = start_date
        else:
            train_s = test_start - pd.DateOffset(months=train_months)

        train_e = test_start - pd.Timedelta(days=1)

        train_len_months = (train_e.year - train_s.year) * 12 + (train_e.month - train_s.month)
        if train_len_months < min_train_months:
            test_start += pd.DateOffset(months=test_months)
            continue

        if test_start > end_date:
            break

        folds_raw.append((train_s, train_e, test_start, test_end))
        test_start += pd.DateOffset(months=test_months)

    if not folds_raw:
        raise ValueError(
            f"No valid walk-forward folds. Dataset ({start_date.date()} to {end_date.date()}) "
            f"is too short for train_months={train_months}, test_months={test_months}."
        )

    oos_results: list[OOSResult] = []
    oos_return_series: list[pd.Series] = []

    for i, (ts, te, oos_s, oos_e) in enumerate(folds_raw):
        logger.info(
            "Walk-forward fold %d/%d: train=%s→%s, test=%s→%s",
            i + 1, len(folds_raw),
            ts.date(), te.date(), oos_s.date(), oos_e.date(),
        )
        try:
            fold_result = run_oos_test(
                weights_fn=weights_fn,
                prices=prices,
                returns=returns,
                train_start=ts.strftime("%Y-%m-%d"),
                train_end=te.strftime("%Y-%m-%d"),
                test_start=oos_s.strftime("%Y-%m-%d"),
                test_end=oos_e.strftime("%Y-%m-%d"),
                backtest_engine=backtest_engine,
                risk_free_rate=risk_free_rate,
            )
            oos_results.append(fold_result)
            oos_return_series.append(fold_result.test_returns)
        except Exception as exc:
            logger.warning("Walk-forward fold %d failed: %s", i + 1, exc)

    if not oos_return_series:
        raise RuntimeError("All walk-forward folds failed.")

    combined_oos = pd.concat(oos_return_series).sort_index()
    combined_equity = (1 + combined_oos).cumprod() * initial_capital

    oos_metrics = compute_metrics(combined_oos, risk_free_rate=risk_free_rate)
    is_metrics_list = [r.train_metrics for r in oos_results]
    oos_metrics_list = [r.test_metrics for r in oos_results]

    positive_sharpe_folds = sum(
        1 for m in oos_metrics_list
        if m.get("sharpe_ratio", float("nan")) > 0
    )
    sharpe_consistency = positive_sharpe_folds / len(oos_results) if oos_results else 0.0

    logger.info(
        "Walk-forward complete: %d folds | combined OOS Sharpe=%.2f | consistency=%.0f%%",
        len(oos_results),
        oos_metrics.get("sharpe_ratio", float("nan")),
        sharpe_consistency * 100,
    )

    return WalkForwardResult(
        folds=oos_results,
        combined_oos_returns=combined_oos,
        combined_oos_equity=combined_equity,
        oos_metrics=oos_metrics,
        in_sample_metrics_by_fold=is_metrics_list,
        oos_metrics_by_fold=oos_metrics_list,
        sharpe_consistency=sharpe_consistency,
        metadata={
            "n_folds": len(oos_results),
            "train_months": train_months,
            "test_months": test_months,
            "expanding": expanding,
        },
    )
