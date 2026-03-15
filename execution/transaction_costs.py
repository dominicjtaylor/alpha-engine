"""
UK transaction cost model for the Systematic Trading Research Framework.

Models three components of trading costs specific to UK equity markets:
1. Broker commission — per-side, expressed in basis points.
2. Market impact / slippage — per-side, wider than US markets for LSE.
3. UK Stamp Duty Reserve Tax (SDRT) — 0.5% on gross long purchases only.

Stamp duty applies only to new long purchases of UK-listed equities.
It does not apply to short sales, position closures, or short-selling.

All costs are returned as a fraction of total portfolio value (not per-share).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class UKCostParams:
    """
    Transaction cost parameters for UK equity trading.

    Attributes
    ----------
    commission_bps : float
        Broker commission per side in basis points (1 bp = 0.01%).
        Typical range for retail/institutional: 5–15 bps.
    slippage_bps : float
        Market impact and slippage per side in basis points.
        LSE securities tend to have wider spreads than NYSE equivalents.
    spread_bps : float
        Half bid-ask spread cost per side in basis points.
    stamp_duty_rate : float
        UK Stamp Duty Reserve Tax rate. Currently 0.005 (0.5%).
        Applied only to long purchases (not short sales).
    """

    commission_bps: float = 10.0
    slippage_bps: float = 10.0
    spread_bps: float = 5.0
    stamp_duty_rate: float = 0.005  # 0.5% SDRT


class UKTransactionCostModel:
    """
    Models realistic UK equity trading costs including stamp duty.

    The total cost on a given trading day has two components:

    1. **Linear trading cost** applied to all turnover (both long and short sides):
       ``cost = turnover × (commission + slippage + spread) / 10000``

    2. **Stamp duty** applied only to *new long purchases*:
       A new long purchase occurs when a position weight increases into or
       within positive territory. The stamp duty cost is:
       ``stamp_cost = gross_long_purchases × stamp_duty_rate``

    Both components are expressed as fractions of portfolio value.

    Parameters
    ----------
    params : UKCostParams
        Cost parameters.
    """

    def __init__(self, params: UKCostParams) -> None:
        self.params = params
        self._total_linear_bps = (
            params.commission_bps + params.slippage_bps + params.spread_bps
        )
        logger.info(
            "UKTransactionCostModel initialised: linear=%.1f bps/side, stamp=%.1f%%",
            self._total_linear_bps, params.stamp_duty_rate * 100,
        )

    def compute_cost(
        self,
        weights: pd.DataFrame,
        prev_weights: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Compute daily transaction costs from weight changes.

        Parameters
        ----------
        weights : pd.DataFrame
            Target portfolio weights for today (post-execution).
        prev_weights : pd.DataFrame
            Actual portfolio weights from the previous day.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            - ``linear_costs``: daily linear trading cost as fraction of portfolio.
            - ``stamp_duty_costs``: daily stamp duty cost as fraction of portfolio.
            Both series share the same DatetimeIndex as ``weights``.
        """
        weight_changes = weights - prev_weights

        # --- Linear cost on all turnover ---
        # Turnover = sum of absolute weight changes / 2
        # (each round-trip is counted once: buy + sell)
        turnover = weight_changes.abs().sum(axis=1) / 2.0
        linear_costs = turnover * self._total_linear_bps / 10_000.0

        # --- Stamp duty on gross long purchases ---
        # A long purchase occurs when weight increases and the new weight is positive
        long_purchases = (weight_changes.where(weights > 0, 0.0)
                          .clip(lower=0.0)
                          .sum(axis=1))
        stamp_duty_costs = long_purchases * self.params.stamp_duty_rate

        return linear_costs, stamp_duty_costs

    def compute_total_cost(
        self,
        weights: pd.DataFrame,
        prev_weights: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute total daily transaction cost (linear + stamp duty).

        Parameters
        ----------
        weights : pd.DataFrame
            Target portfolio weights.
        prev_weights : pd.DataFrame
            Previous day's portfolio weights.

        Returns
        -------
        pd.Series
            Total daily cost as fraction of portfolio value.
        """
        linear, stamp = self.compute_cost(weights, prev_weights)
        total = linear + stamp
        logger.debug(
            "Costs: avg linear=%.2f bps/day, avg stamp=%.2f bps/day",
            linear.mean() * 10_000, stamp.mean() * 10_000,
        )
        return total

    def estimate_annual_cost(
        self,
        avg_daily_turnover: float,
        avg_daily_long_purchases: float,
        trading_days: int = 252,
    ) -> dict[str, float]:
        """
        Estimate annualised transaction costs given average daily turnover.

        Useful for pre-backtest cost budgeting.

        Parameters
        ----------
        avg_daily_turnover : float
            Average daily portfolio turnover as fraction of NAV.
        avg_daily_long_purchases : float
            Average daily gross long purchases as fraction of NAV.
        trading_days : int
            Number of trading days per year (252 for LSE).

        Returns
        -------
        dict[str, float]
            Keys: 'annual_linear_cost', 'annual_stamp_duty', 'annual_total_cost'.
            Values are annualised fractions of NAV (e.g., 0.01 = 1% per year).
        """
        annual_linear = avg_daily_turnover * self._total_linear_bps / 10_000 * trading_days
        annual_stamp = avg_daily_long_purchases * self.params.stamp_duty_rate * trading_days
        return {
            "annual_linear_cost": annual_linear,
            "annual_stamp_duty": annual_stamp,
            "annual_total_cost": annual_linear + annual_stamp,
        }
