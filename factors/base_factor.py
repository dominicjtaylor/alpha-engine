"""
Abstract base class for all alpha factors.

A factor transforms raw price/returns data into a cross-sectional signal
DataFrame where higher values indicate a stronger expected positive return.

Convention: HIGH score = stronger long signal (positive IC expected).
Factors must honour this polarity contract so that the IC pipeline,
quantile analysis, and strategy portfolio construction are all consistent.

All factors must be:
- Stateless (no mutable state between calls)
- Vectorized (no Python loops over dates)
- Free of look-ahead bias (signals at date t use only data available at t)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFactor(ABC):
    """
    Abstract base for all registered alpha factors.

    Subclasses must set class-level ``name`` and ``description`` attributes
    and implement :meth:`compute`.

    Polarity contract
    -----------------
    ``compute()`` must return a signal where **high value = expected positive
    return**.  Factors that are inherently contrarian (e.g. short-term mean
    reversion) should negate internally so callers see a consistent interface.
    """

    #: Unique registry key — override in every subclass.
    name: str = ""

    #: Human-readable one-line description.
    description: str = ""

    @abstractmethod
    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute cross-sectional factor signals.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices — wide format (DatetimeIndex × tickers), GBP.
        returns : pd.DataFrame, optional
            Daily simple returns (same shape). Provided for factors that need
            returns directly; ignored if not needed.
        **kwargs
            Additional data passed through (e.g. ``ohlcv`` dict).

        Returns
        -------
        pd.DataFrame
            Signal scores, same shape as ``prices``.
            Higher value = stronger long signal.
            NaN = no signal for that ticker/date.
        """
