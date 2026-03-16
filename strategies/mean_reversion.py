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
        Compute mean-reversion scores via the factor registry.

        Delegates to :class:`factors.mean_reversion_5d.MeanReversion5dFactor`.
        The factor returns the **negated** N-day return so that the polarity
        contract is honoured: high score = recent loser = expected bounce.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (index=date, columns=tickers).
        returns : pd.DataFrame
            Daily returns (forwarded to factor; not used by this factor).

        Returns
        -------
        pd.DataFrame
            Mean-reversion scores. First ``lookback`` rows are NaN.
            High score = recent loser = long candidate.
        """
        from factors import get_factor
        factor = get_factor("mean_reversion_5d", lookback=self.lookback)
        signals = factor.compute(prices, returns)
        logger.debug(
            "MeanReversion signals computed via registry. Lookback=%d. Non-NaN: %d / %d",
            self.lookback, signals.notna().sum().sum(), signals.size,
        )
        return signals

    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert mean-reversion scores to equal-weighted contrarian weights.

        Because the factor already negated the raw return (high score = recent
        loser), portfolio construction is now identical to momentum: long top
        scores, short bottom scores.

        Algorithm (per date row):
        1. Rank tickers by score (ascending: recent winner = low score).
        2. Long mask: rank >= (1 - long_pct) — recent losers (high score).
        3. Short mask: rank <= long_pct — recent winners (low score).
        4. Equal-weight within each book.

        Parameters
        ----------
        signals : pd.DataFrame
            Mean-reversion scores from :meth:`generate_signals`.
        prices : pd.DataFrame
            Prices (not used directly).

        Returns
        -------
        pd.DataFrame
            Signed portfolio weights. Recent losers are long (+), winners short (-).
        """
        # Percentile ranks: 0 = recent winner (low score), 1 = recent loser (high score)
        ranks = self._rank_cross_section(signals, ascending=True, pct=True)

        # Long mask: high scores = recent losers
        long_mask = (ranks >= (1.0 - self.long_pct)) & signals.notna()

        # Short mask: low scores = recent winners
        short_mask = (ranks <= self.long_pct) & signals.notna()

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
