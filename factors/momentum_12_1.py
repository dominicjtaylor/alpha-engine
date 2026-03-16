"""
12-1 Cross-Sectional Momentum Factor.

Signal: cumulative return from t-lookback to t-skip.
Default: return from t-252 to t-21 (past 12 months excluding last month).

Polarity: high score = strong recent momentum = expected positive future return.
IC is positive — this is a classic momentum factor.

Reference: Jegadeesh & Titman (1993), Fama & French (1996).
"""

from __future__ import annotations

import pandas as pd

from factors.base_factor import BaseFactor
from factors.factor_registry import register


@register
class Momentum121Factor(BaseFactor):
    """
    12-1 cross-sectional momentum factor.

    Signal = prices.shift(skip) / prices.shift(lookback) - 1

    The ``skip`` window excludes the most recent month to avoid short-term
    reversal contamination that would dilute the momentum signal.

    Parameters
    ----------
    lookback : int
        Total lookback in trading days (default 252 ≈ 12 months).
    skip : int
        Recent days to exclude (default 21 ≈ 1 month).
    """

    name = "momentum_12_1"
    description = "12-1 cross-sectional momentum: past 12 months excl. last month"

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute 12-1 momentum scores.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (DatetimeIndex × tickers), GBP.
        returns : pd.DataFrame, optional
            Not used; accepted for interface consistency.

        Returns
        -------
        pd.DataFrame
            Momentum scores. First ``lookback`` rows are NaN.
            Higher score = stronger momentum = expected positive future return.
        """
        return prices.shift(self.skip) / prices.shift(self.lookback) - 1
