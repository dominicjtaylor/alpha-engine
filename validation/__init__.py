"""
Statistical Validation Tools.

Out-of-sample testing and parameter stability analysis to guard against
overfitting in systematic strategy research.
"""

from validation.oos import (
    OOSResult,
    run_oos_test,
    run_walk_forward,
    WalkForwardResult,
)
from validation.stability import (
    run_parameter_stability,
    StabilityResult,
)

__all__ = [
    "OOSResult",
    "run_oos_test",
    "run_walk_forward",
    "WalkForwardResult",
    "run_parameter_stability",
    "StabilityResult",
]
