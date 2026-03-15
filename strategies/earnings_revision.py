"""
Earnings Revision Drift Strategy (PEAD approximation).

Approximates Post-Earnings Announcement Drift using large overnight price gaps
as a proxy for earnings surprises. When a stock opens significantly higher or
lower than the previous close, the market may be reacting to an earnings or
news announcement. The drift effect suggests these moves continue for several
days after the initial reaction.

Implementation logic:
1. Identify large overnight gaps: |Open[t] / Close[t-1] - 1| > threshold (default 5%)
2. Positive gap → go long for hold_days trading days
3. Negative gap → go short for hold_days trading days
4. Overlapping signals (new signal during active position) override the previous

Reference: Ball & Brown (1968), Bernard & Thomas (1989).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import StrategyConfig
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class EarningsRevisionDrift(BaseStrategy):
    """
    Post-Earnings Announcement Drift via overnight gap detection.

    This is an event-driven strategy. On days with a large overnight price gap
    (proxy for an earnings announcement or significant news), a position is
    entered in the direction of the gap and held for a fixed number of
    trading days.

    The ``weights`` output is therefore sparse — most entries are non-zero only
    during active hold windows. The strategy can be active in many tickers
    simultaneously.

    Parameters
    ----------
    config : StrategyConfig
        Key fields:
        - ``earnings_gap_threshold`` (default 0.05): minimum |gap| to trigger.
        - ``earnings_hold_days`` (default 10): days to hold after signal.

    Notes
    -----
    This strategy requires OHLCV data (Open and Close prices) passed via the
    ``ohlcv`` keyword argument to :meth:`generate_signals`. If ``ohlcv`` is
    not provided, the strategy falls back to using adjusted close prices as
    a proxy (less accurate).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__("EarningsRevisionDrift", config)
        self.gap_threshold: float = config.earnings_gap_threshold
        self.hold_days: int = config.earnings_hold_days

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        ohlcv: Optional[dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Identify overnight gap events and assign directional signal scores.

        The signal value is the overnight gap magnitude (positive = long signal,
        negative = short signal). Only gaps exceeding ``gap_threshold`` receive
        a non-NaN signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (index=date, columns=tickers).
        returns : pd.DataFrame
            Daily returns.
        ohlcv : dict[str, pd.DataFrame], optional
            Mapping of ticker → OHLCV DataFrame with columns including
            'Open' and 'Close'. Passed from DataLoader.load_ohlcv().
            If None, uses close-to-close returns as a less accurate proxy.

        Returns
        -------
        pd.DataFrame
            Sparse signal DataFrame. Most entries are NaN (no event).
            Non-NaN entries contain the gap magnitude (+ or -).
        """
        signals = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

        if ohlcv is not None:
            signals = self._compute_gap_signals_from_ohlcv(ohlcv, prices.index, prices.columns)
        else:
            # Fallback: use daily returns as a proxy (less accurate)
            logger.warning(
                "EarningsRevisionDrift: no OHLCV data provided. "
                "Using daily returns as proxy for overnight gaps."
            )
            signals = self._compute_gap_signals_from_returns(returns)

        n_events = signals.notna().sum().sum()
        logger.info(
            "EarningsRevisionDrift: %d gap events detected (threshold=%.1f%%)",
            n_events, self.gap_threshold * 100,
        )
        return signals

    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Forward-fill gap signals over the holding period.

        For each gap event at (date t, ticker i):
        - Set weight = sign(gap) for the window [t, t + hold_days - 1]
        - If a new signal for the same ticker occurs during an active hold,
          the new signal overrides (its direction may differ)

        After filling all hold windows, each row is normalised by the number
        of active positions on that date, so gross exposure per book stays ~1.

        Implementation uses a **sparse event loop** over signal events
        (typically < 5% of date/ticker combinations) rather than a vectorised
        rolling operation, which cannot correctly track individual hold windows.

        Parameters
        ----------
        signals : pd.DataFrame
            Sparse signal DataFrame from :meth:`generate_signals`.
        prices : pd.DataFrame
            Not used directly; kept for interface consistency.

        Returns
        -------
        pd.DataFrame
            Portfolio weights with forward-filled positions.
        """
        dates = signals.index
        tickers = signals.columns
        n_dates = len(dates)
        n_tickers = len(tickers)

        # Date → integer index lookup for O(1) access
        date_to_idx = {d: i for i, d in enumerate(dates)}
        ticker_to_idx = {t: j for j, t in enumerate(tickers)}

        # Pre-allocate raw weight array
        raw_weights = np.zeros((n_dates, n_tickers), dtype=np.float64)

        # Iterate over sparse signal events only
        event_rows, event_cols = np.where(signals.notna().values)
        for row_idx, col_idx in zip(event_rows, event_cols):
            gap_value = signals.iloc[row_idx, col_idx]
            direction = np.sign(gap_value)
            end_idx = min(row_idx + self.hold_days, n_dates)
            raw_weights[row_idx:end_idx, col_idx] = direction

        # Normalise: divide signed weights by daily active long + short counts
        # so each "book" carries gross exposure ~1
        weights_df = pd.DataFrame(raw_weights, index=dates, columns=tickers)

        # Daily count of long positions and short positions
        daily_long_count = (weights_df > 0).sum(axis=1).replace(0, np.nan)
        daily_short_count = (weights_df < 0).sum(axis=1).replace(0, np.nan)

        # Scale longs to sum to 1, shorts to sum to -1 per day
        long_weights = weights_df.where(weights_df > 0, 0.0).div(daily_long_count, axis=0)
        short_weights = weights_df.where(weights_df < 0, 0.0).div(daily_short_count, axis=0)

        combined = long_weights.fillna(0.0) + short_weights.fillna(0.0)

        logger.debug(
            "EarningsRevisionDrift portfolio: mean %.1f active positions/day",
            (combined != 0).sum(axis=1).mean(),
        )
        return combined

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_gap_signals_from_ohlcv(
        self,
        ohlcv: dict[str, pd.DataFrame],
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> pd.DataFrame:
        """
        Compute overnight gap signals from OHLCV data.

        overnight_gap[t] = Open[t] / Close[t-1] - 1

        Parameters
        ----------
        ohlcv : dict[str, pd.DataFrame]
            Ticker → DataFrame with 'Open' and 'Close' columns.
        index : pd.DatetimeIndex
            Target date index.
        columns : pd.Index
            Target ticker columns.

        Returns
        -------
        pd.DataFrame
            Sparse signal DataFrame aligned to ``index`` and ``columns``.
        """
        signals = pd.DataFrame(np.nan, index=index, columns=columns)

        for ticker in columns:
            if ticker not in ohlcv:
                continue
            df = ohlcv[ticker]
            if "Open" not in df.columns or "Close" not in df.columns:
                continue

            try:
                open_prices = df["Open"].reindex(index)
                close_prices = df["Close"].reindex(index)
                prev_close = close_prices.shift(1)

                overnight_gap = open_prices / prev_close - 1
                # Only keep gaps exceeding the threshold
                mask = overnight_gap.abs() >= self.gap_threshold
                signals[ticker] = overnight_gap.where(mask)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Gap signal error for %s: %s", ticker, exc)

        return signals

    def _compute_gap_signals_from_returns(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fallback: use large daily returns as a proxy for overnight gaps.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily simple returns.

        Returns
        -------
        pd.DataFrame
            Sparse signal DataFrame where |return| > threshold.
        """
        mask = returns.abs() >= self.gap_threshold
        return returns.where(mask)
