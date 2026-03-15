"""
Factor Research Toolkit.

Evaluate signals as alpha factors via information coefficient (IC),
factor decay, and quantile portfolio analysis.
"""

from factor_research.ic import (
    compute_ic,
    compute_rolling_ic,
    compute_ic_summary,
)
from factor_research.decay import compute_factor_decay
from factor_research.quantile import (
    compute_quantile_portfolios,
    QuantileResult,
)

__all__ = [
    "compute_ic",
    "compute_rolling_ic",
    "compute_ic_summary",
    "compute_factor_decay",
    "compute_quantile_portfolios",
    "QuantileResult",
]
