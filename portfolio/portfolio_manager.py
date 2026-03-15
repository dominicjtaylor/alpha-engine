"""
Multi-strategy portfolio combination and risk control.

The PortfolioManager is responsible for:
1. Combining weights from multiple strategies into a single portfolio.
2. Applying portfolio-level risk controls (position limits, leverage,
   volatility targeting).
3. Optionally adjusting rebalance frequency.

All calculations are in GBP. Risk controls use a shift(1) convention to
avoid look-ahead bias: today's risk scaling is based on yesterday's
estimated volatility/exposure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config import RiskConfig
from strategies.base import StrategyResult

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


class CombinationMethod(Enum):
    """Strategy combination methods."""

    EQUAL_WEIGHT = "equal_weight"
    """Average the weights across all strategies with equal allocation."""

    VOLATILITY_SCALE = "volatility_scale"
    """Scale each strategy's allocation inversely proportional to its
    realised volatility, normalised to a common target volatility."""

    SHARPE_WEIGHT = "sharpe_weight"
    """Weight strategies by their rolling Sharpe ratio (requires pre-backtest)."""


@dataclass
class PortfolioConfig:
    """
    Configuration for portfolio combination and risk controls.

    Attributes
    ----------
    combination_method : CombinationMethod
        How to combine strategy weights.
    vol_lookback : int
        Rolling window (trading days) for realised volatility estimation.
    target_vol_per_strategy : float
        Annualised volatility target applied to each strategy before combining
        (used in VOLATILITY_SCALE mode).
    rebalance_frequency : str
        Pandas offset alias: 'D' (daily), 'W' (weekly), 'M' (monthly).
    """

    combination_method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT
    vol_lookback: int = 63
    target_vol_per_strategy: float = 0.10
    rebalance_frequency: str = "D"


class PortfolioManager:
    """
    Combines multiple strategy weight DataFrames and applies risk controls.

    Responsibility boundary: this class handles signal combination and
    risk scaling only. Transaction cost modelling stays in BacktestEngine.

    All strategy weight DataFrames must share the same [n_dates × n_tickers]
    grid, enabling direct DataFrame arithmetic.

    Parameters
    ----------
    config : PortfolioConfig
        Portfolio combination configuration.
    risk_config : RiskConfig
        Risk limit configuration (max leverage, position pct, vol target).
    """

    def __init__(
        self,
        config: PortfolioConfig,
        risk_config: RiskConfig,
    ) -> None:
        self.config = config
        self.risk_config = risk_config
        logger.info(
            "PortfolioManager initialised: method=%s | max leverage=%.1fx",
            config.combination_method.value, risk_config.max_leverage,
        )

    def combine_strategies(
        self,
        strategy_results: dict[str, StrategyResult],
    ) -> pd.DataFrame:
        """
        Combine multiple strategy weight DataFrames into a single portfolio.

        Parameters
        ----------
        strategy_results : dict[str, StrategyResult]
            Mapping of strategy name → StrategyResult from strategy.run().
            Each StrategyResult.weights must have the same columns (tickers).

        Returns
        -------
        pd.DataFrame
            Combined portfolio weights (index=date, columns=tickers).
        """
        if not strategy_results:
            raise ValueError("No strategy results provided to combine.")

        weights_dict = {name: result.weights for name, result in strategy_results.items()}
        method = self.config.combination_method

        if method == CombinationMethod.EQUAL_WEIGHT:
            combined = self._combine_equal_weight(weights_dict)
        elif method == CombinationMethod.VOLATILITY_SCALE:
            combined = self._combine_vol_scale(weights_dict)
        elif method == CombinationMethod.SHARPE_WEIGHT:
            combined = self._combine_sharpe_weight(weights_dict)
        else:
            raise ValueError(f"Unknown combination method: {method}")

        logger.info(
            "Combined %d strategies via '%s'. Shape: %s",
            len(strategy_results), method.value, combined.shape,
        )
        return combined

    def apply_risk_controls(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply portfolio-level risk controls sequentially.

        Controls applied in order:
        1. Clip individual position sizes to max_position_pct.
        2. Portfolio-level volatility targeting: scale to hit target_volatility.
        3. Clip gross exposure to max_gross_exposure (max leverage).

        All scaling uses lagged (shift(1)) estimates to avoid look-ahead bias.

        Parameters
        ----------
        weights : pd.DataFrame
            Combined portfolio weights.
        returns : pd.DataFrame
            Daily asset returns (used to estimate portfolio vol for scaling).

        Returns
        -------
        pd.DataFrame
            Risk-controlled portfolio weights.
        """
        # 1. Position size limit
        weights = weights.clip(
            lower=-self.risk_config.max_position_pct,
            upper=self.risk_config.max_position_pct,
        )

        # 2. Volatility targeting
        weights = self._apply_vol_targeting(weights, returns)

        # 3. Leverage cap
        weights = self._apply_leverage_cap(weights)

        return weights

    def apply_rebalance_frequency(
        self,
        weights: pd.DataFrame,
        frequency: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Forward-fill weights on non-rebalance days.

        For example, with frequency='M' (monthly rebalancing), weights are
        only updated on the last trading day of each month. Between rebalance
        dates, yesterday's weights are carried forward unchanged.

        Parameters
        ----------
        weights : pd.DataFrame
            Daily portfolio weights.
        frequency : str, optional
            Pandas offset alias ('D', 'W', 'M'). If None, uses config value.
            'D' = no-op (daily rebalance, no forward-fill needed).

        Returns
        -------
        pd.DataFrame
            Weights with non-rebalance days filled from previous rebalance date.
        """
        freq = frequency or self.config.rebalance_frequency
        if freq == "D":
            return weights

        # Identify rebalance dates: last trading day of each period
        rebalance_dates = weights.resample(freq).last().index
        rebalance_mask = weights.index.isin(rebalance_dates)

        # Keep weights only on rebalance dates; forward-fill to non-rebalance days
        weights_rebalanced = weights.where(rebalance_mask)
        weights_rebalanced = weights_rebalanced.ffill()

        n_rebalance = rebalance_mask.sum()
        logger.info(
            "Rebalance frequency '%s': %d rebalance dates over %d trading days",
            freq, n_rebalance, len(weights),
        )
        return weights_rebalanced

    # ------------------------------------------------------------------
    # Private: combination methods
    # ------------------------------------------------------------------

    def _combine_equal_weight(
        self,
        weights_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Average strategy weights with equal allocation."""
        # Align all DataFrames to the union of dates and tickers
        all_weights = list(weights_dict.values())
        combined = pd.concat(all_weights, axis=0).groupby(level=0).mean()

        # Alternative: explicit reindex + mean across strategies
        common_index = all_weights[0].index
        common_cols = all_weights[0].columns
        for w in all_weights[1:]:
            common_index = common_index.intersection(w.index)
            common_cols = common_cols.intersection(w.columns)

        aligned = [w.reindex(index=common_index, columns=common_cols)
                   for w in all_weights]
        combined = sum(aligned) / len(aligned)
        return combined.fillna(0.0)

    def _combine_vol_scale(
        self,
        weights_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Scale each strategy's weights by inverse realised volatility.

        Implementation:
        1. Compute proxy daily returns for each strategy:
           strategy_return[t] = (weights[t-1] * asset_returns[t]).sum()
           We approximate this using the weight magnitude as a proxy
           (actual returns not available at this stage).
        2. Compute rolling std of strategy proxy returns.
        3. Scale = target_vol / (rolling_std * sqrt(252)).
        4. Combine scaled weights with equal allocation.

        Note: The proxy return used here is the sum of absolute weights
        (a rough vol indicator). For full accuracy, run each strategy
        through BacktestEngine first and use BacktestResult.daily_returns.
        """
        target_vol = self.config.target_vol_per_strategy
        lookback = self.config.vol_lookback
        scaled = []

        for name, weights in weights_dict.items():
            # Proxy vol: use rolling std of gross weight changes
            proxy_signal = weights.abs().sum(axis=1)
            proxy_vol = proxy_signal.rolling(lookback).std()
            proxy_vol = proxy_vol.replace(0, np.nan)
            # Don't scale until we have enough history
            scale = (target_vol / proxy_vol).clip(upper=3.0).shift(1).fillna(1.0)
            scaled_w = weights.mul(scale, axis=0)
            scaled.append(scaled_w)

        # Equal-weight the scaled strategies
        common_index = scaled[0].index
        common_cols = scaled[0].columns
        for w in scaled[1:]:
            common_index = common_index.intersection(w.index)
            common_cols = common_cols.intersection(w.columns)

        aligned = [w.reindex(index=common_index, columns=common_cols)
                   for w in scaled]
        combined = sum(aligned) / len(aligned)
        return combined.fillna(0.0)

    def _combine_sharpe_weight(
        self,
        weights_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Weight strategies by rolling Sharpe ratio.

        Computes rolling gross return proxies per strategy and allocates
        more capital to higher-Sharpe strategies. Falls back to equal
        weighting when Sharpe data is insufficient.
        """
        lookback = self.config.vol_lookback

        # Compute rolling Sharpe proxy per strategy (using weight magnitude)
        sharpe_dict: dict[str, pd.Series] = {}
        for name, weights in weights_dict.items():
            proxy = weights.abs().sum(axis=1)
            roll_mean = proxy.rolling(lookback).mean()
            roll_std = proxy.rolling(lookback).std().replace(0, np.nan)
            sharpe_proxy = (roll_mean / roll_std).shift(1).fillna(1.0).clip(lower=0)
            sharpe_dict[name] = sharpe_proxy

        sharpe_df = pd.DataFrame(sharpe_dict)
        sharpe_total = sharpe_df.sum(axis=1).replace(0, np.nan)
        allocation = sharpe_df.div(sharpe_total, axis=0).fillna(1.0 / len(weights_dict))

        # Combine with Sharpe-weighted allocation
        all_weights = list(weights_dict.values())
        common_index = all_weights[0].index
        common_cols = all_weights[0].columns
        for w in all_weights[1:]:
            common_index = common_index.intersection(w.index)
            common_cols = common_cols.intersection(w.columns)

        combined = pd.DataFrame(0.0, index=common_index, columns=common_cols)
        for i, (name, weights) in enumerate(weights_dict.items()):
            aligned = weights.reindex(index=common_index, columns=common_cols).fillna(0.0)
            alloc = allocation[name].reindex(common_index).fillna(1.0 / len(weights_dict))
            combined += aligned.mul(alloc, axis=0)

        return combined

    # ------------------------------------------------------------------
    # Private: risk controls
    # ------------------------------------------------------------------

    def _apply_vol_targeting(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scale portfolio weights to target annualised volatility.

        Uses lagged portfolio volatility estimate to avoid look-ahead bias.
        The scaling factor is capped at max_leverage to prevent extreme leverage.

        Parameters
        ----------
        weights : pd.DataFrame
        returns : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        lookback = self.risk_config.vol_lookback
        target_vol = self.risk_config.target_volatility

        # Estimate portfolio daily returns from lagged weights
        aligned_returns = returns.reindex(columns=weights.columns).fillna(0.0)
        portfolio_returns = (weights.shift(1) * aligned_returns).sum(axis=1)

        # Rolling annualised volatility
        rolling_vol = portfolio_returns.rolling(lookback).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling_vol = rolling_vol.replace(0, np.nan)

        # Scale factor: target / realised. Shift by 1 to use yesterday's estimate.
        scale = (target_vol / rolling_vol).shift(1).clip(
            upper=self.risk_config.max_leverage
        ).fillna(1.0)

        return weights.mul(scale, axis=0)

    def _apply_leverage_cap(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Uniformly scale down any rows where gross exposure exceeds max_leverage.

        Parameters
        ----------
        weights : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        gross = weights.abs().sum(axis=1)
        scale = (self.risk_config.max_leverage / gross).clip(upper=1.0)
        return weights.mul(scale, axis=0)
