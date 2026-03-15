"""
Walk-forward analysis for out-of-sample strategy validation.

Walk-forward testing splits the historical data into in-sample (IS) training
periods and out-of-sample (OOS) evaluation periods. This gives a more realistic
picture of strategy performance than a single full-period backtest, because
the OOS results were never used during signal development.

Two modes are supported:
- **Expanding window**: IS period grows forward from a fixed start date.
  More data is used for each successive fold. Realistic for live deployment.
- **Rolling window**: IS window has fixed length and rolls forward.
  Tests whether strategy parameters are stable over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from backtester.engine import BacktestEngine, BacktestResult
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """
    Dates for a single walk-forward fold.

    Attributes
    ----------
    fold_number : int
        Sequential fold index (0-based).
    is_start : str
        In-sample period start date (ISO 8601).
    is_end : str
        In-sample period end date (ISO 8601).
    oos_start : str
        Out-of-sample period start date (ISO 8601).
    oos_end : str
        Out-of-sample period end date (ISO 8601).
    """

    fold_number: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str


@dataclass
class WalkForwardResult:
    """
    Aggregated results from a complete walk-forward analysis.

    Attributes
    ----------
    strategy_name : str
        Name of the strategy tested.
    folds : list[WalkForwardFold]
        Fold date definitions.
    oos_results : list[BacktestResult]
        BacktestResult for each OOS fold.
    combined_oos_returns : pd.Series
        Concatenated OOS daily returns across all folds.
    combined_oos_equity : pd.Series
        Equity curve computed from combined_oos_returns.
    is_months : int
        In-sample period length in months.
    oos_months : int
        Out-of-sample period length in months.
    expanding : bool
        True if expanding window was used.
    """

    strategy_name: str
    folds: list[WalkForwardFold]
    oos_results: list[BacktestResult]
    combined_oos_returns: pd.Series
    combined_oos_equity: pd.Series
    is_months: int
    oos_months: int
    expanding: bool
    metadata: dict = field(default_factory=dict)


class WalkForwardAnalysis:
    """
    Walk-forward validation for systematic trading strategies.

    Parameters
    ----------
    engine : BacktestEngine
        Configured backtest engine.
    is_months : int
        In-sample period length in months. Default 36 (3 years).
    oos_months : int
        Out-of-sample evaluation period in months. Default 12 (1 year).
    expanding : bool
        If True, use expanding IS window (IS start fixed, grows forward).
        If False, use rolling IS window (fixed IS length).
    """

    def __init__(
        self,
        engine: BacktestEngine,
        is_months: int = 36,
        oos_months: int = 12,
        expanding: bool = True,
    ) -> None:
        self.engine = engine
        self.is_months = is_months
        self.oos_months = oos_months
        self.expanding = expanding

    def run(
        self,
        strategy: BaseStrategy,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **strategy_kwargs,
    ) -> WalkForwardResult:
        """
        Execute walk-forward analysis.

        For each fold:
        1. Slice IS prices/returns and run strategy to generate weights.
        2. Run backtest engine on OOS weights using OOS returns.
        3. Store OOS BacktestResult.

        The strategy is stateless, so there is no information leakage
        between IS fitting and OOS evaluation.

        Parameters
        ----------
        strategy : BaseStrategy
            Strategy to evaluate (stateless — called fresh per fold).
        prices : pd.DataFrame
            Full-period adjusted close prices.
        returns : pd.DataFrame
            Full-period daily returns.
        **strategy_kwargs
            Additional arguments forwarded to strategy.run() (e.g., ohlcv dict).

        Returns
        -------
        WalkForwardResult
            Aggregated OOS results and combined equity curve.
        """
        folds = self._build_folds(prices.index)
        if not folds:
            raise ValueError(
                "Not enough data for walk-forward analysis. "
                f"Need at least {self.is_months + self.oos_months} months."
            )

        logger.info(
            "Walk-forward: %d folds | IS=%d months | OOS=%d months | mode=%s",
            len(folds), self.is_months, self.oos_months,
            "expanding" if self.expanding else "rolling",
        )

        oos_results: list[BacktestResult] = []
        oos_return_series: list[pd.Series] = []

        for fold in folds:
            logger.info(
                "Fold %d: IS [%s → %s] | OOS [%s → %s]",
                fold.fold_number, fold.is_start, fold.is_end,
                fold.oos_start, fold.oos_end,
            )

            # Slice IS data for signal generation
            is_prices = prices.loc[fold.is_start:fold.is_end]
            is_returns = returns.loc[fold.is_start:fold.is_end]

            # Run strategy on IS period to generate weights
            # We use the full IS weights but only evaluate OOS
            is_result = strategy.run(is_prices, is_returns, **strategy_kwargs)

            # Run strategy on IS+OOS to get weights covering OOS period
            # (strategy needs lookback history before OOS start)
            full_prices = prices.loc[fold.is_start:fold.oos_end]
            full_returns = returns.loc[fold.is_start:fold.oos_end]
            full_result = strategy.run(full_prices, full_returns, **strategy_kwargs)

            # Slice weights and returns to OOS window only
            oos_weights = full_result.weights.loc[fold.oos_start:fold.oos_end]
            oos_returns = returns.loc[fold.oos_start:fold.oos_end]

            if oos_weights.empty or oos_returns.empty:
                logger.warning("Fold %d: empty OOS data, skipping.", fold.fold_number)
                continue

            oos_bt = self.engine.run(
                weights=oos_weights,
                returns=oos_returns,
                strategy_name=f"{strategy.name}_fold{fold.fold_number}_OOS",
            )
            oos_results.append(oos_bt)
            oos_return_series.append(oos_bt.daily_returns)

        if not oos_return_series:
            raise RuntimeError("Walk-forward produced no valid OOS results.")

        combined_returns = pd.concat(oos_return_series).sort_index()
        combined_equity = (1 + combined_returns).cumprod() * self.engine.initial_capital

        logger.info(
            "Walk-forward complete. Combined OOS: %.1f%% total return | "
            "%d trading days",
            (combined_equity.iloc[-1] / self.engine.initial_capital - 1) * 100,
            len(combined_returns),
        )

        return WalkForwardResult(
            strategy_name=strategy.name,
            folds=folds,
            oos_results=oos_results,
            combined_oos_returns=combined_returns,
            combined_oos_equity=combined_equity,
            is_months=self.is_months,
            oos_months=self.oos_months,
            expanding=self.expanding,
            metadata={"n_folds": len(folds), "n_oos_days": len(combined_returns)},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_folds(self, date_index: pd.DatetimeIndex) -> list[WalkForwardFold]:
        """
        Construct fold date boundaries from the full date index.

        Parameters
        ----------
        date_index : pd.DatetimeIndex
            All available trading dates.

        Returns
        -------
        list[WalkForwardFold]
        """
        folds: list[WalkForwardFold] = []
        start = date_index.min()
        end = date_index.max()

        is_delta = pd.DateOffset(months=self.is_months)
        oos_delta = pd.DateOffset(months=self.oos_months)

        fold_num = 0
        is_start = start

        while True:
            is_end = is_start + is_delta
            oos_start = is_end + pd.Timedelta(days=1)
            oos_end = oos_start + oos_delta

            if oos_end > end:
                break

            # Find actual trading days closest to these dates
            is_start_actual = self._nearest_date(date_index, is_start)
            is_end_actual = self._nearest_date(date_index, is_end)
            oos_start_actual = self._nearest_date(date_index, oos_start)
            oos_end_actual = self._nearest_date(date_index, oos_end)

            folds.append(WalkForwardFold(
                fold_number=fold_num,
                is_start=str(is_start_actual.date()),
                is_end=str(is_end_actual.date()),
                oos_start=str(oos_start_actual.date()),
                oos_end=str(oos_end_actual.date()),
            ))

            fold_num += 1
            if self.expanding:
                # IS start stays fixed, advance by OOS window
                oos_start = oos_end + pd.Timedelta(days=1)
                is_end = oos_start - pd.Timedelta(days=1)
                is_start = start  # IS always starts from the beginning
            else:
                # Rolling: advance both IS and OOS by OOS window
                is_start = is_start + oos_delta

        return folds

    @staticmethod
    def _nearest_date(
        date_index: pd.DatetimeIndex,
        target: pd.Timestamp,
    ) -> pd.Timestamp:
        """Return the trading date closest to (and not after) the target date."""
        available = date_index[date_index <= target]
        if available.empty:
            return date_index.min()
        return available.max()
