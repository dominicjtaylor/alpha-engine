"""
Cross-Sectional Momentum Strategy.

Signal: 12-1 momentum — cumulative return from t-252 to t-21, excluding the
most recent month to avoid short-term reversal contamination.

Portfolio: Rank stocks cross-sectionally. Long the top decile, short the
bottom decile. Equal-weighted within each book. Dollar-neutral (rows sum ~0).

Reference: Jegadeesh & Titman (1993), Fama & French (1996).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import StrategyConfig
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class CrossSectionalMomentum(BaseStrategy):
    """
    12-1 Cross-Sectional Momentum strategy for UK equities.

    Computes the cumulative return from ``t - lookback`` to ``t - skip``
    (skipping the most recent month to avoid reversal), ranks stocks
    cross-sectionally, and constructs a dollar-neutral long-short portfolio.

    Parameters
    ----------
    config : StrategyConfig
        Strategy parameters. Key fields:
        - ``momentum_lookback`` (default 252): total lookback in trading days.
        - ``momentum_skip`` (default 21): recent days to exclude.
        - ``momentum_long_pct`` (default 0.10): fraction of universe in each book.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__("CrossSectionalMomentum", config)
        self.lookback: int = config.momentum_lookback
        self.skip: int = config.momentum_skip
        self.long_pct: float = config.momentum_long_pct

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute 12-1 momentum scores.

        Signal = prices.shift(skip) / prices.shift(lookback) - 1

        This gives the cumulative return from t-252 to t-21 (using defaults),
        computed entirely from lagged prices so there is no look-ahead bias.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (index=date, columns=tickers).
        returns : pd.DataFrame
            Daily returns (not used directly, but accepted for interface consistency).

        Returns
        -------
        pd.DataFrame
            Momentum scores. First ``lookback`` rows are NaN.
        """
        signals = prices.shift(self.skip) / prices.shift(self.lookback) - 1
        logger.debug(
            "Momentum signals computed. Non-NaN cells: %d / %d",
            signals.notna().sum().sum(), signals.size,
        )
        return signals

    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert momentum scores to equal-weighted long-short portfolio weights.

        Algorithm (per date row):
        1. Compute cross-sectional percentile rank of momentum scores.
        2. Long mask: rank >= (1 - long_pct) — top decile by default.
        3. Short mask: rank <= long_pct — bottom decile by default.
        4. Equal-weight within each book: weight = ±1 / n_positions.
        5. Combine long (positive) and short (negative) books.

        The result is approximately dollar-neutral: long and short books
        each carry ~1/n weight, so net exposure is ~0.

        Parameters
        ----------
        signals : pd.DataFrame
            Momentum scores from :meth:`generate_signals`.
        prices : pd.DataFrame
            Prices (not used, kept for interface consistency).

        Returns
        -------
        pd.DataFrame
            Signed portfolio weights. Long positions > 0, short < 0.
        """
        # Percentile ranks: 0 = lowest momentum, 1 = highest momentum
        ranks = self._rank_cross_section(signals, ascending=True, pct=True)

        # Long: top long_pct of universe
        long_mask = ranks >= (1.0 - self.long_pct)
        long_mask = long_mask & signals.notna()

        # Short: bottom long_pct of universe
        short_mask = ranks <= self.long_pct
        short_mask = short_mask & signals.notna()

        # Equal-weight within each book
        long_weights = self._equal_weight_book(long_mask, sign=1.0)
        short_weights = self._equal_weight_book(short_mask, sign=-1.0)

        weights = long_weights + short_weights

        n_long = long_mask.sum(axis=1).mean()
        n_short = short_mask.sum(axis=1).mean()
        logger.debug(
            "Momentum portfolio: avg %.1f long / %.1f short positions per day",
            n_long, n_short,
        )
        return weights
