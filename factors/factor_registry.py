"""
Factor Registry.

Central registry mapping factor names to factor classes.  Factors are
registered via the ``@register`` decorator in their own module.  The
``factors/__init__.py`` imports all factor modules to trigger registration,
so callers only need ``from factors import get_factor``.

Usage
-----
# Define and register a new factor
from factors.factor_registry import register
from factors.base_factor import BaseFactor

@register
class MyFactor(BaseFactor):
    name = "my_factor"
    description = "My custom factor"
    def compute(self, prices, returns=None, **kwargs):
        return prices.pct_change(5)   # placeholder

# Consume from anywhere
from factors import get_factor, list_factors
factor = get_factor("momentum_12_1", lookback=252, skip=21)
signals = factor.compute(prices, returns)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from factors.base_factor import BaseFactor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal registry — populated by @register decorator
# ---------------------------------------------------------------------------

FACTOR_REGISTRY: dict[str, type[BaseFactor]] = {}


def register(cls: type[BaseFactor]) -> type[BaseFactor]:
    """
    Class decorator that adds a factor to the global registry.

    Parameters
    ----------
    cls : type[BaseFactor]
        Factor class with a ``name`` class attribute set.

    Returns
    -------
    type[BaseFactor]
        The same class, unchanged (decorator passes through).

    Raises
    ------
    ValueError
        If ``cls.name`` is empty or already registered by a different class.
    """
    if not cls.name:
        raise ValueError(f"Factor class {cls.__name__} must set a non-empty 'name' attribute.")
    if cls.name in FACTOR_REGISTRY and FACTOR_REGISTRY[cls.name] is not cls:
        logger.warning("Factor '%s' already registered — overwriting with %s.",
                       cls.name, cls.__name__)
    FACTOR_REGISTRY[cls.name] = cls
    logger.debug("Registered factor: '%s' (%s)", cls.name, cls.__name__)
    return cls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_factor(name: str, **params: Any) -> BaseFactor:
    """
    Instantiate a registered factor by name.

    Parameters
    ----------
    name : str
        Registry key — see ``list_factors()``.
    **params
        Constructor keyword arguments forwarded to the factor class.

    Returns
    -------
    BaseFactor
        A fresh factor instance (new instance on every call).

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in FACTOR_REGISTRY:
        raise KeyError(
            f"Factor '{name}' not found in registry. "
            f"Available: {list_factors()}"
        )
    return FACTOR_REGISTRY[name](**params)


def list_factors() -> list[str]:
    """Return a sorted list of all registered factor names."""
    return sorted(FACTOR_REGISTRY.keys())


def factor_metadata() -> dict[str, dict[str, str]]:
    """
    Return name and description for every registered factor.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{name: {"name": ..., "description": ..., "class": ...}}``
    """
    return {
        name: {
            "name": name,
            "description": cls.description,
            "class": cls.__name__,
        }
        for name, cls in FACTOR_REGISTRY.items()
    }


def compute_factor_matrix(
    names: list[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    params_by_name: dict[str, dict[str, Any]] | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Compute signals for multiple factors and return as a name → DataFrame dict.

    Parameters
    ----------
    names : list[str]
        Factor registry keys to compute.
    prices : pd.DataFrame
        Adjusted close prices (DatetimeIndex × tickers).
    returns : pd.DataFrame, optional
        Daily returns (same shape as prices).
    params_by_name : dict, optional
        ``{factor_name: {param: value, ...}}`` — constructor kwargs per factor.
    **kwargs
        Forwarded to every factor's ``compute()`` call.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{factor_name: signal_DataFrame}``
    """
    params_by_name = params_by_name or {}
    result: dict[str, pd.DataFrame] = {}
    for name in names:
        factor = get_factor(name, **params_by_name.get(name, {}))
        result[name] = factor.compute(prices, returns, **kwargs)
        logger.debug("compute_factor_matrix: computed '%s' (%d×%d)",
                     name, *result[name].shape)
    return result
