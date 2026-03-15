"""Data pipeline package for the UK Systematic Trading Research Framework."""

from data.data_loader import DataLoader, clean_data, compute_returns
from data.universe import get_ftse_universe, load_ftse_universe, apply_liquidity_filters

__all__ = [
    "DataLoader",
    "clean_data",
    "compute_returns",
    "get_ftse_universe",
    "load_ftse_universe",
    "apply_liquidity_filters",
]
