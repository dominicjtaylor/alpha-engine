"""
Short-Term Mean Reversion Factor.

Signal: negated N-day cumulative return.
Default: negated 5-day return.

Polarity: high score = large recent loss = expected positive future return
(contrarian bounce). Negation is applied internally so the polarity contract
is satisfied — callers see high score = buy like all other factors.

Reference: Lehmann (1990), Lo & MacKinlay (1990).
"""

from __future__ import annotations

import pandas as pd

from factors.base_factor import BaseFactor
from factors.factor_registry import register


@register
class MeanReversion5dFactor(BaseFactor):
    """
    Short-term mean reversion factor.

    Signal = -(prices / prices.shift(lookback) - 1)

    The negation ensures polarity consistency: a stock that has fallen
    sharply over the lookback receives a high (positive) score, signalling
    a likely bounce.  IC vs 1-day forward returns is expected to be positive.

    Parameters
    ----------
    lookback : int
        Lookback window in trading days (default 5 = 1 week).
    """

    name = "mean_reversion_5d"
    description = "Short-term mean reversion: negated N-day return (contrarian)"

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute short-term mean-reversion scores.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (DatetimeIndex × tickers), GBP.
        returns : pd.DataFrame, optional
            Not used; accepted for interface consistency.

        Returns
        -------
        pd.DataFrame
            Mean-reversion scores. First ``lookback`` rows are NaN.
            Higher score = recent loser = expected positive future return.
        """
        raw_return = prices / prices.shift(self.lookback) - 1
        return -raw_return  # invert so high score = buy
