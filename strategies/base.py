"""
Abstract base class and shared utilities for all trading strategies.

All strategies in this framework follow the same interface contract:
1. ``generate_signals()`` — compute raw alpha scores from price data
2. ``construct_portfolio()`` — convert scores to signed portfolio weights
3. ``run()`` — orchestrate the above two steps and return a ``StrategyResult``

Strategies are **stateless** beyond their configuration parameters.
No mutable state is stored between calls, which is critical for walk-forward
testing where the same strategy object is called with different date slices.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import StrategyConfig


@dataclass
class StrategyResult:
    """
    Standardised output returned by every strategy.

    Attributes
    ----------
    signals : pd.DataFrame
        Raw alpha scores. Wide format (index=date, columns=tickers).
        Higher value = stronger long signal. NaN = insufficient data.
    weights : pd.DataFrame
        Signed portfolio weights. Wide format, same shape as signals.
        Positive = long, negative = short.
        Each row (date) is independently normalised; rows sum to ~0
        for long-short dollar-neutral portfolios.
    name : str
        Human-readable strategy name.
    metadata : dict, optional
        Intermediate DataFrames and diagnostics for debugging and analysis.
    """

    signals: pd.DataFrame
    weights: pd.DataFrame
    name: str = ""
    metadata: Optional[dict] = field(default=None)


class BaseStrategy(ABC):
    """
    Abstract base class for all alpha strategies.

    Subclasses must implement :meth:`generate_signals` and
    :meth:`construct_portfolio`. The :meth:`run` method orchestrates
    these and should not be overridden.

    Parameters
    ----------
    name : str
        Human-readable strategy name used in logging and reporting.
    config : StrategyConfig
        Strategy parameter configuration.
    """

    def __init__(self, name: str, config: StrategyConfig) -> None:
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute raw cross-sectional alpha scores.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (index=date, columns=tickers), GBP.
        returns : pd.DataFrame
            Daily simple returns, same shape as prices.
        **kwargs
            Additional data (e.g., ``ohlcv`` dict for EarningsRevisionDrift).

        Returns
        -------
        pd.DataFrame
            Signal scores, same shape as input.
            Higher value = stronger long signal.
            NaN = no signal (ticker excluded from portfolio on that date).
        """

    @abstractmethod
    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert raw signal scores to signed portfolio weights.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of :meth:`generate_signals`.
        prices : pd.DataFrame
            Adjusted close prices (for context/normalisation).

        Returns
        -------
        pd.DataFrame
            Signed portfolio weights, same shape as signals.
            Positive = long position, negative = short position.
            Each row is normalised independently.
        """

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> StrategyResult:
        """
        Orchestrate signal generation and portfolio construction.

        This method should not be overridden by subclasses.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices.
        returns : pd.DataFrame
            Daily returns.
        **kwargs
            Forwarded to :meth:`generate_signals`.

        Returns
        -------
        StrategyResult
        """
        self.logger.info("Running strategy '%s' on %d dates × %d tickers",
                         self.name, *prices.shape)

        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.construct_portfolio(signals, prices)
        self._validate_weights(weights)

        self.logger.info(
            "Strategy '%s' complete. Non-zero positions on %.1f%% of date/ticker cells.",
            self.name,
            100.0 * (weights != 0).sum().sum() / max(weights.size, 1),
        )
        return StrategyResult(signals=signals, weights=weights, name=self.name)

    # ------------------------------------------------------------------
    # Protected utilities available to all subclasses
    # ------------------------------------------------------------------

    def _rank_cross_section(
        self,
        df: pd.DataFrame,
        ascending: bool = True,
        pct: bool = True,
    ) -> pd.DataFrame:
        """
        Compute cross-sectional percentile ranks row-by-row.

        Parameters
        ----------
        df : pd.DataFrame
            Values to rank (index=date, columns=tickers).
        ascending : bool
            If True, smallest value gets rank 0.0; largest gets 1.0.
        pct : bool
            If True, returns percentile ranks in [0, 1].
            If False, returns integer ranks.

        Returns
        -------
        pd.DataFrame
            Ranked values. NaN tickers are excluded from the rank pool
            and their rank remains NaN.
        """
        return df.rank(axis=1, pct=pct, ascending=ascending, na_option="keep")

    def _equal_weight_book(
        self,
        mask: pd.DataFrame,
        sign: float = 1.0,
    ) -> pd.DataFrame:
        """
        Assign equal weights to positions identified by a boolean mask.

        Parameters
        ----------
        mask : pd.DataFrame
            Boolean DataFrame indicating positions to include.
        sign : float
            +1.0 for long book, -1.0 for short book.

        Returns
        -------
        pd.DataFrame
            Weights DataFrame. Each row sums to ``sign`` × 1.0 where
            there are positions, 0.0 where there are no positions.
        """
        n_positions = mask.sum(axis=1).replace(0, np.nan)
        weights = mask.astype(float).div(n_positions, axis=0) * sign
        return weights.fillna(0.0)

    def _validate_weights(self, weights: pd.DataFrame) -> None:
        """
        Sanity-check portfolio weights and log warnings.

        Checks for:
        - Infinite values (programming error)
        - Gross exposure > 4× (unusually high leverage warning)
        - Any single weight > 50% of portfolio

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights to validate.
        """
        if np.isinf(weights.values).any():
            raise ValueError(f"[{self.name}] Portfolio weights contain infinite values.")

        gross_exposure = weights.abs().sum(axis=1)
        max_gross = gross_exposure.max()
        if max_gross > 4.0:
            self.logger.warning(
                "[%s] Maximum gross exposure %.2f× exceeds 4×. Check signal scaling.",
                self.name, max_gross,
            )

        max_single_weight = weights.abs().max().max()
        if max_single_weight > 0.5:
            self.logger.warning(
                "[%s] Maximum single position weight %.1f%% exceeds 50%%.",
                self.name, max_single_weight * 100,
            )
