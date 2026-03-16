"""
Factor layer for the Alpha Engine.

Importing this package registers all built-in factors.  Any code that needs
to use the registry should import from here:

    from factors import get_factor, list_factors, compute_factor_matrix

or

    from factors.factor_registry import get_factor

Both work because this ``__init__.py`` imports all factor modules, which
triggers the ``@register`` decorator on each factor class.
"""

from factors.factor_registry import (  # noqa: F401
    FACTOR_REGISTRY,
    compute_factor_matrix,
    factor_metadata,
    get_factor,
    list_factors,
    register,
)

# Import factor modules to trigger @register decorators
from factors.momentum_12_1 import Momentum121Factor  # noqa: F401
from factors.mean_reversion_5d import MeanReversion5dFactor  # noqa: F401
