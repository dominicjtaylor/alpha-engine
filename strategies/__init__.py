"""Strategy implementations for the UK Systematic Trading Research Framework."""

from strategies.base import BaseStrategy, StrategyResult
from strategies.earnings_revision import EarningsRevisionDrift
from strategies.mean_reversion import ShortTermMeanReversion
from strategies.momentum import CrossSectionalMomentum

__all__ = [
    "BaseStrategy",
    "StrategyResult",
    "CrossSectionalMomentum",
    "ShortTermMeanReversion",
    "EarningsRevisionDrift",
]
