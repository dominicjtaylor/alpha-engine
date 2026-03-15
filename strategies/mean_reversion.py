"""
Short-Term Mean Reversion Strategy.

Signal: 5-day cumulative return. Stocks with large negative 5-day returns
tend to mean-revert (bounce), and vice versa.

reversion_signal = -return(t-5 to t)

Portfolio: Long the worst recent performers (bottom quintile by 5-day return),
short the best recent performers (top quintile). Daily rebalance.

Reference: Lehmann (1990), Lo & MacKinlay (1990).
"""

from __future__ import annotations

import logging

import pandas as pd

from config import StrategyConfig
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class ShortTermMeanReversion(BaseStrategy):
    """
    5-Day Short-Term Mean Reversion strategy for UK equities.

    The signal is the raw 5-day cumulative return. The portfolio construction
    inverts the signal polarity: recent losers become longs, recent winners
    become shorts.

    The strategy rebalances daily, making it more sensitive to transaction
    costs than momentum. A wider long_pct (0.20 by default) is used to
    increase diversification and reduce per-position turnover.

    Parameters
    ----------
    config : StrategyConfig
        Strategy parameters. Key fields:
        - ``mean_reversion_lookback`` (default 5): lookback window.
        - ``mean_reversion_long_pct`` (default 0.20): fraction per book.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__("ShortTermMeanReversion", config)
        self.lookback: int = config.mean_reversion_lookback
        self.long_pct: float = config.mean_reversion_long_pct

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute 5-day cumulative return as the raw signal.

        Signal = prices[t] / prices[t - lookback] - 1

        A **negative** signal (recent loser) implies a long position after
        signal inversion in :meth:`construct_portfolio`.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (index=date, columns=tickers).
        returns : pd.DataFrame
            Daily returns (accepted for interface consistency, not used directly).

        Returns
        -------
        pd.DataFrame
            5-day cumulative return scores. First ``lookback`` rows are NaN.
        """
        signals = prices / prices.shift(self.lookback) - 1
        logger.debug(
            "MeanReversion signals computed. Lookback=%d. Non-NaN: %d / %d",
            self.lookback, signals.notna().sum().sum(), signals.size,
        )
        return signals

    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert 5-day return scores to equal-weighted contrarian weights.

        Signal polarity is **inverted**: low 5-day return → long position,
        high 5-day return → short position.

        Algorithm (per date row):
        1. Rank tickers by 5-day return (ascending: lowest return = rank 0).
        2. Short mask: rank >= (1 - long_pct) — recent winners.
        3. Long mask: rank <= long_pct — recent losers.
        4. Equal-weight within each book.

        Parameters
        ----------
        signals : pd.DataFrame
            5-day return scores from :meth:`generate_signals`.
        prices : pd.DataFrame
            Prices (not used directly).

        Returns
        -------
        pd.DataFrame
            Signed portfolio weights. Recent losers are long (+), winners short (-).
        """
        # Percentile ranks: 0 = biggest loser, 1 = biggest winner
        ranks = self._rank_cross_section(signals, ascending=True, pct=True)

        # Long mask: recent losers (low rank = low recent return)
        long_mask = (ranks <= self.long_pct) & signals.notna()

        # Short mask: recent winners (high rank = high recent return)
        short_mask = (ranks >= (1.0 - self.long_pct)) & signals.notna()

        long_weights = self._equal_weight_book(long_mask, sign=1.0)
        short_weights = self._equal_weight_book(short_mask, sign=-1.0)

        weights = long_weights + short_weights

        n_long = long_mask.sum(axis=1).mean()
        n_short = short_mask.sum(axis=1).mean()
        logger.debug(
            "MeanReversion portfolio: avg %.1f long / %.1f short positions per day",
            n_long, n_short,
        )
        return weights
