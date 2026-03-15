"""Analytics package for the UK Systematic Trading Research Framework."""

from analytics.metrics import (
    annual_return,
    annual_volatility,
    calmar_ratio,
    compute_metrics,
    compute_monthly_returns,
    compute_rolling_sharpe,
    compute_turnover_stats,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from analytics.visualizations import PerformanceVisualizer

__all__ = [
    "compute_metrics",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "compute_rolling_sharpe",
    "compute_monthly_returns",
    "compute_turnover_stats",
    "PerformanceVisualizer",
]
