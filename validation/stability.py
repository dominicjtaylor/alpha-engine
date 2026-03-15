"""
Parameter Stability Analysis.

Tests whether a strategy's performance is sensitive to small changes in its
parameters. A robust strategy should show relatively consistent performance
across a neighbourhood of parameter values. A strategy that only works for
one specific parameter combination is likely overfit.

The output is a ``StabilityResult`` containing a grid of Sharpe ratios (and
other metrics) indexed by the varied parameters, suitable for heatmap plots.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from analytics.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """Output from a parameter stability run."""

    param_grid: dict[str, list]
    results_df: pd.DataFrame          # MultiIndex rows = param combos, cols = metrics
    sharpe_grid: pd.DataFrame         # Pivoted Sharpe (2D for heatmap when 2 params varied)
    best_params: dict[str, Any]
    best_sharpe: float
    sharpe_std: float                  # Std dev of Sharpe across param grid
    sharpe_cv: float                   # Coefficient of variation (std/mean)
    stability_score: float             # 1 - CV, clamped [0,1]; higher = more stable
    metadata: dict = field(default_factory=dict)


def run_parameter_stability(
    weights_fn_factory: Callable[[dict], Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    param_grid: dict[str, list],
    backtest_engine=None,
    risk_free_rate: float = 0.04,
    n_jobs: int = 1,
) -> StabilityResult:
    """
    Sweep a parameter grid and measure metric stability across combinations.

    Parameters
    ----------
    weights_fn_factory : Callable[dict, weights_fn]
        A factory that accepts a parameter dict and returns a ``weights_fn``
        callable of the form ``fn(prices, returns) -> weights_df``.
        Example::

            def factory(params):
                strat = CrossSectionalMomentum(config_with(**params))
                return lambda p, r: strat.run(p, r).weights

    prices : pd.DataFrame
        Full adjusted close prices.
    returns : pd.DataFrame
        Full daily returns.
    param_grid : dict[str, list]
        Parameter names → list of values to sweep.
        Example: ``{"lookback": [126, 189, 252], "long_pct": [0.10, 0.15, 0.20]}``
    backtest_engine : BacktestEngine | None
        If provided, uses this engine for net return calculation (with costs).
        If None, computes gross returns.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation.
    n_jobs : int
        Number of parallel workers (1 = serial).

    Returns
    -------
    StabilityResult
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))

    logger.info(
        "Parameter stability sweep: %d combinations of %s",
        len(combos), param_names,
    )

    rows: list[dict] = []

    for combo in combos:
        params = dict(zip(param_names, combo))
        try:
            weights_fn = weights_fn_factory(params)
            weights = weights_fn(prices, returns)

            if backtest_engine is not None:
                result = backtest_engine.run(weights, returns, "stability_sweep")
                rets = result.daily_returns
            else:
                w_exec = weights.shift(1).reindex(returns.index, method=None).fillna(0.0)
                common = w_exec.columns.intersection(returns.columns)
                rets = (w_exec[common] * returns[common]).sum(axis=1)

            metrics = compute_metrics(rets, risk_free_rate=risk_free_rate)
            row = {**params, **metrics}
        except Exception as exc:
            logger.warning("Stability sweep failed for params %s: %s", params, exc)
            row = {**params, "sharpe_ratio": float("nan"), "annual_return": float("nan")}

        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Build pivot for 2-param sweeps (for heatmap visualisation)
    if len(param_names) == 2:
        sharpe_grid = results_df.pivot(
            index=param_names[0],
            columns=param_names[1],
            values="sharpe_ratio",
        )
    elif len(param_names) == 1:
        sharpe_grid = results_df.set_index(param_names[0])[["sharpe_ratio"]]
    else:
        # For 3+ params, pivot the first two
        sharpe_grid = results_df.pivot_table(
            index=param_names[0],
            columns=param_names[1],
            values="sharpe_ratio",
            aggfunc="mean",
        )

    sharpe_vals = results_df["sharpe_ratio"].dropna().values
    best_idx = results_df["sharpe_ratio"].idxmax() if len(sharpe_vals) > 0 else None

    if best_idx is not None:
        best_row = results_df.loc[best_idx]
        best_params = {k: best_row[k] for k in param_names}
        best_sharpe = float(best_row["sharpe_ratio"])
    else:
        best_params = {}
        best_sharpe = float("nan")

    sharpe_std = float(np.nanstd(sharpe_vals)) if len(sharpe_vals) > 0 else float("nan")
    sharpe_mean = float(np.nanmean(sharpe_vals)) if len(sharpe_vals) > 0 else float("nan")
    sharpe_cv = abs(sharpe_std / sharpe_mean) if sharpe_mean and abs(sharpe_mean) > 1e-9 else float("nan")
    stability_score = max(0.0, min(1.0, 1.0 - sharpe_cv)) if np.isfinite(sharpe_cv) else float("nan")

    logger.info(
        "Stability sweep complete: best Sharpe=%.2f | Sharpe std=%.2f | stability=%.2f",
        best_sharpe, sharpe_std, stability_score if np.isfinite(stability_score) else float("nan"),
    )

    return StabilityResult(
        param_grid=param_grid,
        results_df=results_df,
        sharpe_grid=sharpe_grid,
        best_params=best_params,
        best_sharpe=best_sharpe,
        sharpe_std=sharpe_std,
        sharpe_cv=sharpe_cv if np.isfinite(sharpe_cv) else float("nan"),
        stability_score=stability_score if np.isfinite(stability_score) else float("nan"),
        metadata={
            "n_combinations": len(combos),
            "param_names": param_names,
        },
    )
