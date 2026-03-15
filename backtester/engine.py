"""
Vectorized backtesting engine for the UK Systematic Trading Research Framework.

Simulates realistic trading conditions:
- EOD signal → next-day execution (weights.shift(1) execution lag)
- UK-specific transaction costs including stamp duty
- Position size limits and leverage constraints
- Daily rebalancing
- Capital tracking in GBP

The engine is fully vectorized — no Python loop over dates — enabling backtests
on 350 stocks over 15 years to complete in seconds.

All outputs (equity curve, P&L, costs) are denominated in GBP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import RiskConfig
from execution.transaction_costs import UKCostParams, UKTransactionCostModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """
    Complete output from a single backtest run.

    All monetary series are in GBP unless noted.

    Attributes
    ----------
    strategy_name : str
        Name of the strategy that produced these results.
    daily_returns : pd.Series
        Net daily portfolio returns as a fraction (DatetimeIndex).
        Includes transaction costs and stamp duty.
    equity_curve : pd.Series
        Cumulative portfolio value in GBP, starting from initial_capital.
    drawdowns : pd.Series
        Drawdown from peak (always <= 0). E.g., -0.10 = 10% drawdown.
    positions : pd.DataFrame
        Daily portfolio weights after execution lag and position limits.
    turnover : pd.Series
        Daily portfolio turnover as fraction of NAV.
    gross_exposure : pd.Series
        Daily sum of absolute position weights (1.0 = fully invested).
    gross_returns : pd.Series
        Daily returns before transaction costs.
    linear_costs : pd.Series
        Daily linear trading cost (commission + slippage + spread).
    stamp_duty_costs : pd.Series
        Daily stamp duty cost (UK SDRT on long purchases).
    trade_costs : pd.Series
        Total daily cost = linear_costs + stamp_duty_costs.
    metadata : dict
        Strategy name, date range, config snapshot, and summary statistics.
    """

    strategy_name: str
    daily_returns: pd.Series
    equity_curve: pd.Series
    drawdowns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    gross_exposure: pd.Series
    gross_returns: pd.Series
    linear_costs: pd.Series
    stamp_duty_costs: pd.Series
    trade_costs: pd.Series
    metadata: dict = field(default_factory=dict)


class BacktestEngine:
    """
    Vectorized backtesting engine for UK equities.

    Key design principles
    ---------------------
    1. **No look-ahead bias**: weights computed at close[t] are executed at
       open[t+1], implemented as ``weights.shift(1)``.
    2. **Vectorized**: all operations use pandas/numpy broadcasting.
    3. **UK costs**: uses :class:`UKTransactionCostModel` including SDRT.
    4. **Position limits**: single-position weights clipped before execution.

    Parameters
    ----------
    config : RiskConfig
        Risk and cost configuration.
    initial_capital : float
        Starting portfolio capital in GBP.
    """

    def __init__(
        self,
        config: RiskConfig,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        self.config = config
        self.initial_capital = initial_capital

        cost_params = UKCostParams(
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
            spread_bps=config.spread_bps,
            stamp_duty_rate=config.stamp_duty_rate,
        )
        self.cost_model = UKTransactionCostModel(cost_params)
        logger.info(
            "BacktestEngine initialised. Capital: £%,.0f | Max position: %.1f%% | "
            "Max leverage: %.1fx",
            initial_capital,
            config.max_position_pct * 100,
            config.max_leverage,
        )

    def run(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        strategy_name: str = "unnamed",
    ) -> BacktestResult:
        """
        Run a vectorized backtest.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights from a strategy (index=date, columns=tickers).
            Rows need not sum to 1; position limits are applied by the engine.
        returns : pd.DataFrame
            Daily simple returns, same shape as weights.
        strategy_name : str
            Human-readable name included in the BacktestResult.

        Returns
        -------
        BacktestResult
            Complete backtest output including equity curve and cost breakdown.
        """
        logger.info("Running backtest for '%s'", strategy_name)

        # 1. Align on common dates and tickers
        common_dates = weights.index.intersection(returns.index)
        common_tickers = weights.columns.intersection(returns.columns)
        weights = weights.loc[common_dates, common_tickers]
        returns = returns.loc[common_dates, common_tickers]

        # 2. Apply single-position size limits
        weights = self._apply_position_limits(weights)

        # 3. Apply gross leverage limit
        weights = self._apply_leverage_limit(weights)

        # 4. Execution lag: EOD signal → next-day fill
        #    weights_exec[t] = weights computed at close[t-1]
        weights_exec = weights.shift(1).fillna(0.0)
        prev_weights = weights_exec.shift(1).fillna(0.0)

        # 5. Gross portfolio return: sum of (weight × asset return) per day
        gross_returns = (weights_exec * returns).sum(axis=1)

        # 6. Transaction costs
        linear_costs, stamp_costs = self.cost_model.compute_cost(weights_exec, prev_weights)

        # 7. Net daily return
        net_returns = gross_returns - linear_costs - stamp_costs

        # 8. Derived series
        equity_curve = self._compute_equity_curve(net_returns)
        drawdowns = self._compute_drawdowns(equity_curve)
        turnover = self._compute_turnover(weights_exec)
        gross_exposure = weights_exec.abs().sum(axis=1)

        metadata = {
            "strategy_name": strategy_name,
            "start_date": str(common_dates.min().date()),
            "end_date": str(common_dates.max().date()),
            "n_dates": len(common_dates),
            "n_tickers": len(common_tickers),
            "initial_capital_gbp": self.initial_capital,
            "avg_daily_turnover": float(turnover.mean()),
            "avg_gross_exposure": float(gross_exposure.mean()),
            "total_linear_cost_pct": float(linear_costs.sum() * 100),
            "total_stamp_duty_pct": float(stamp_costs.sum() * 100),
        }

        logger.info(
            "Backtest '%s' complete. Net return: %.1f%% | Max DD: %.1f%% | "
            "Avg turnover: %.1f%%/day",
            strategy_name,
            (equity_curve.iloc[-1] / self.initial_capital - 1) * 100,
            drawdowns.min() * 100,
            turnover.mean() * 100,
        )

        return BacktestResult(
            strategy_name=strategy_name,
            daily_returns=net_returns,
            equity_curve=equity_curve,
            drawdowns=drawdowns,
            positions=weights_exec,
            turnover=turnover,
            gross_exposure=gross_exposure,
            gross_returns=gross_returns,
            linear_costs=linear_costs,
            stamp_duty_costs=stamp_costs,
            trade_costs=linear_costs + stamp_costs,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_position_limits(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Clip individual position weights to [-max_position_pct, +max_position_pct]."""
        return weights.clip(
            lower=-self.config.max_position_pct,
            upper=self.config.max_position_pct,
        )

    def _apply_leverage_limit(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Scale down rows where gross exposure exceeds max_leverage.

        Scaling preserves relative weights within each row.
        """
        gross = weights.abs().sum(axis=1)
        # Where gross > max_leverage, scale uniformly
        scale = (self.config.max_leverage / gross).clip(upper=1.0)
        return weights.mul(scale, axis=0)

    def _compute_equity_curve(self, daily_returns: pd.Series) -> pd.Series:
        """
        Compute cumulative equity curve in GBP.

        Parameters
        ----------
        daily_returns : pd.Series
            Net daily returns.

        Returns
        -------
        pd.Series
            Portfolio value in GBP, starting at initial_capital.
        """
        return (1.0 + daily_returns).cumprod() * self.initial_capital

    def _compute_drawdowns(self, equity_curve: pd.Series) -> pd.Series:
        """
        Compute drawdown from rolling peak.

        Returns
        -------
        pd.Series
            Drawdown series (always <= 0). E.g., -0.15 = 15% below peak.
        """
        rolling_max = equity_curve.cummax()
        return equity_curve / rolling_max - 1.0

    def _compute_turnover(self, weights_exec: pd.DataFrame) -> pd.Series:
        """
        Compute daily portfolio turnover.

        Turnover = sum of absolute weight changes / 2.
        Division by 2 because each rebalancing trade involves both a sale
        and a purchase (counted once as a round trip).

        Parameters
        ----------
        weights_exec : pd.DataFrame
            Execution weights (already shifted by 1 day).

        Returns
        -------
        pd.Series
            Daily turnover as fraction of portfolio NAV.
        """
        return weights_exec.diff().abs().sum(axis=1) / 2.0
