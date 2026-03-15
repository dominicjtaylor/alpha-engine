"""
Parameter sweep (grid search) for systematic strategy optimisation.

Performs an exhaustive grid search over strategy configuration parameters.
Each parameter combination is independently backtested. Results are ranked by
Sharpe ratio.

Uses Python's multiprocessing for parallel execution across CPU cores.

Warning: Grid search on historical data risks overfitting. Always validate
the best parameters on out-of-sample data using WalkForwardAnalysis.
"""

from __future__ import annotations

import copy
import itertools
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Type

import pandas as pd

from analytics.metrics import compute_metrics
from backtester.engine import BacktestEngine, BacktestResult
from config import StrategyConfig
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    """
    Result for a single parameter combination.

    Attributes
    ----------
    params : dict[str, Any]
        Parameter values used for this run.
    backtest_result : BacktestResult
        Full backtest output.
    metrics : dict[str, float]
        Performance metrics dictionary.
    """

    params: dict[str, Any]
    backtest_result: BacktestResult
    metrics: dict[str, float]


def _run_single_sweep(
    args: tuple,
) -> SweepResult | None:
    """
    Worker function for parallel parameter sweep.

    Designed to be called by multiprocessing.Pool.map().

    Parameters
    ----------
    args : tuple
        (params_dict, strategy_class, base_config, prices, returns, initial_capital)

    Returns
    -------
    SweepResult or None if the run failed.
    """
    params, strategy_class, base_config, prices, returns, initial_capital, risk_config = args
    try:
        # Create modified config by copying base and updating with sweep params
        config = copy.deepcopy(base_config)
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)

        strategy = strategy_class(config)
        result = strategy.run(prices, returns)

        from backtester.engine import BacktestEngine
        from config import RiskConfig
        engine = BacktestEngine(config=risk_config, initial_capital=initial_capital)
        bt_result = engine.run(result.weights, returns, strategy_name=strategy.name)

        from analytics.metrics import compute_metrics
        metrics = compute_metrics(bt_result.daily_returns)

        return SweepResult(params=params, backtest_result=bt_result, metrics=metrics)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Sweep run failed for params %s: %s", params, exc)
        return None


class ParameterSweep:
    """
    Grid search over strategy parameters.

    Constructs a Cartesian product of all parameter value combinations,
    runs each as a full backtest, and ranks results by Sharpe ratio.

    Parameters
    ----------
    engine : BacktestEngine
        Configured backtest engine (used for cost params and capital).
    strategy_class : type[BaseStrategy]
        Strategy class to instantiate for each combination.
    base_config : StrategyConfig
        Base configuration from which parameter variants are derived.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        strategy_class: Type[BaseStrategy],
        base_config: StrategyConfig,
    ) -> None:
        self.engine = engine
        self.strategy_class = strategy_class
        self.base_config = base_config

    def run_grid(
        self,
        param_grid: dict[str, list],
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        n_jobs: int = -1,
        **strategy_kwargs,
    ) -> list[SweepResult]:
        """
        Execute grid search across all parameter combinations.

        Parameters
        ----------
        param_grid : dict[str, list]
            Dictionary mapping parameter names to lists of values to try.
            Example: {"momentum_lookback": [126, 252], "momentum_long_pct": [0.10, 0.15]}
        prices : pd.DataFrame
            Price data for the full evaluation period.
        returns : pd.DataFrame
            Return data for the full evaluation period.
        n_jobs : int
            Number of parallel workers. -1 = use all available CPU cores.
            1 = sequential (useful for debugging).
        **strategy_kwargs
            Additional kwargs forwarded to strategy.run() (e.g., ohlcv).

        Returns
        -------
        list[SweepResult]
            Results for all valid parameter combinations, sorted by Sharpe ratio
            in descending order.
        """
        # Build Cartesian product of parameter values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = [
            dict(zip(param_names, combo))
            for combo in itertools.product(*param_values)
        ]

        n_combos = len(combinations)
        logger.info(
            "Parameter sweep: %d combinations for %s",
            n_combos, self.strategy_class.__name__,
        )

        if n_jobs == 1 or n_combos == 1:
            # Sequential execution (avoids multiprocessing overhead)
            raw_results = []
            for params in combinations:
                args = (
                    params,
                    self.strategy_class,
                    self.base_config,
                    prices,
                    returns,
                    self.engine.initial_capital,
                    self.engine.config,
                )
                res = _run_single_sweep(args)
                if res is not None:
                    raw_results.append(res)
        else:
            # Parallel execution
            if n_jobs == -1:
                n_jobs = mp.cpu_count()

            args_list = [
                (
                    params,
                    self.strategy_class,
                    self.base_config,
                    prices,
                    returns,
                    self.engine.initial_capital,
                    self.engine.config,
                )
                for params in combinations
            ]

            with mp.Pool(processes=n_jobs) as pool:
                raw_results_nullable = pool.map(_run_single_sweep, args_list)

            raw_results = [r for r in raw_results_nullable if r is not None]

        n_failed = n_combos - len(raw_results)
        if n_failed > 0:
            logger.warning("%d / %d sweep runs failed.", n_failed, n_combos)

        # Sort by Sharpe ratio descending
        results = sorted(
            raw_results,
            key=lambda r: r.metrics.get("sharpe_ratio", float("-inf")),
            reverse=True,
        )

        logger.info(
            "Parameter sweep complete. Best Sharpe: %.2f (params: %s)",
            results[0].metrics.get("sharpe_ratio", float("nan")) if results else float("nan"),
            results[0].params if results else {},
        )
        return results

    def summarize(self, results: list[SweepResult]) -> pd.DataFrame:
        """
        Build a summary DataFrame of all sweep results.

        Parameters
        ----------
        results : list[SweepResult]
            Output of :meth:`run_grid`.

        Returns
        -------
        pd.DataFrame
            Columns: all parameter names + all metric names.
            Sorted by Sharpe ratio descending.
        """
        rows = []
        for r in results:
            row = dict(r.params)
            row.update(r.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        if "sharpe_ratio" in df.columns:
            df = df.sort_values("sharpe_ratio", ascending=False)
        return df.reset_index(drop=True)
