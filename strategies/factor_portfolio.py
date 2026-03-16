"""
Multi-Factor Composite Portfolio Strategy.

Combines multiple registered factors into a single composite score using
configurable per-factor weights. Each factor signal is cross-sectionally
z-scored before combining, ensuring no single factor dominates due to
scale differences.

Example usage:
    strat = FactorPortfolioStrategy(
        config=StrategyConfig(),
        factor_names=["momentum_12_1", "mean_reversion_5d"],
        factor_weights=[0.7, 0.3],
        long_pct=0.10,
    )
    result = strat.run(prices, returns)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import StrategyConfig
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class FactorPortfolioStrategy(BaseStrategy):
    """
    Composite long-short portfolio driven by a weighted blend of factors.

    Each factor from the registry is computed and cross-sectionally z-scored.
    The z-scores are combined as a weighted average to form a composite signal.
    Stocks are then ranked by composite score and assigned to long/short books.

    Parameters
    ----------
    config : StrategyConfig
        Strategy parameters (used for fallback defaults).
    factor_names : list[str]
        Names of factors to combine (must be registered in the factor registry).
    factor_weights : list[float], optional
        Per-factor weights for the composite. Must sum to 1.0 (normalised
        internally if not). Defaults to equal weighting.
    factor_params : dict[str, dict], optional
        Extra keyword arguments forwarded to each factor constructor.
        Keys are factor names; values are dicts of kwargs.
        E.g. ``{"momentum_12_1": {"lookback": 126, "skip": 21}}``.
    long_pct : float
        Fraction of universe to place in each of the long and short books.
        Default 0.10 (top/bottom decile).
    """

    def __init__(
        self,
        config: StrategyConfig,
        factor_names: list[str],
        factor_weights: list[float] | None = None,
        factor_params: dict[str, dict] | None = None,
        long_pct: float = 0.10,
    ) -> None:
        if not factor_names:
            raise ValueError("factor_names must be a non-empty list.")

        name = "FactorPortfolio[" + ",".join(factor_names) + "]"
        super().__init__(name, config)

        self.factor_names: list[str] = factor_names
        self.factor_params: dict[str, dict] = factor_params or {}
        self.long_pct: float = long_pct

        # Normalise weights to sum to 1
        n = len(factor_names)
        if factor_weights is None:
            self.factor_weights = [1.0 / n] * n
        else:
            if len(factor_weights) != n:
                raise ValueError(
                    f"factor_weights length ({len(factor_weights)}) must match "
                    f"factor_names length ({n})."
                )
            total = sum(factor_weights)
            if total <= 0:
                raise ValueError("factor_weights must sum to a positive value.")
            self.factor_weights = [w / total for w in factor_weights]

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute per-factor signals, z-score each, and blend into a composite.

        Each factor is retrieved from the registry, computed on ``prices``
        and ``returns``, then cross-sectionally z-scored (mean 0, std 1 per
        date). The z-scores are combined as a weighted average.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (index=date, columns=tickers).
        returns : pd.DataFrame
            Daily returns (forwarded to each factor).

        Returns
        -------
        pd.DataFrame
            Composite factor scores. High score = long candidate.
            NaN where all constituent factors have NaN.
        """
        from factors import get_factor

        factor_signals: list[pd.DataFrame] = []
        for name, weight in zip(self.factor_names, self.factor_weights):
            params = self.factor_params.get(name, {})
            factor = get_factor(name, **params)
            raw = factor.compute(prices, returns)

            # Cross-sectional z-score: (x - mean) / std per date row
            row_mean = raw.mean(axis=1)
            row_std = raw.std(axis=1).replace(0, np.nan)
            zscored = raw.sub(row_mean, axis=0).div(row_std, axis=0)

            factor_signals.append(zscored * weight)
            logger.debug(
                "Factor '%s' (weight=%.2f): non-NaN cells %d / %d",
                name, weight, raw.notna().sum().sum(), raw.size,
            )

        # Sum weighted z-scores; NaN only where ALL factors are NaN
        composite = factor_signals[0].copy()
        for fs in factor_signals[1:]:
            # align on columns; addna-aware
            composite = composite.add(fs, fill_value=0)

        # Where all factors were NaN (fill_value=0 masks this), restore NaN
        all_nan = pd.concat(
            [sig.isna() for sig in factor_signals], axis=1, keys=range(len(factor_signals))
        ).groupby(level=0, axis=1).all() if False else None

        # Simpler: count how many non-NaN contributions each cell had
        valid_count = sum(
            sig.notna().astype(float) for sig in factor_signals
        )
        composite[valid_count == 0] = np.nan

        logger.debug(
            "Composite signal: %d factors combined. Non-NaN cells: %d / %d",
            len(self.factor_names), composite.notna().sum().sum(), composite.size,
        )
        return composite

    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert composite scores to equal-weighted long-short portfolio weights.

        Algorithm (per date row):
        1. Cross-sectional percentile rank of composite scores.
        2. Long mask: top ``long_pct`` by composite score.
        3. Short mask: bottom ``long_pct`` by composite score.
        4. Equal-weight within each book.

        Parameters
        ----------
        signals : pd.DataFrame
            Composite factor scores from :meth:`generate_signals`.
        prices : pd.DataFrame
            Prices (unused; kept for interface consistency).

        Returns
        -------
        pd.DataFrame
            Signed portfolio weights. Long > 0, short < 0.
        """
        ranks = self._rank_cross_section(signals, ascending=True, pct=True)

        long_mask = (ranks >= (1.0 - self.long_pct)) & signals.notna()
        short_mask = (ranks <= self.long_pct) & signals.notna()

        long_weights = self._equal_weight_book(long_mask, sign=1.0)
        short_weights = self._equal_weight_book(short_mask, sign=-1.0)

        weights = long_weights + short_weights

        n_long = long_mask.sum(axis=1).mean()
        n_short = short_mask.sum(axis=1).mean()
        logger.debug(
            "FactorPortfolio portfolio: avg %.1f long / %.1f short positions per day",
            n_long, n_short,
        )
        return weights
