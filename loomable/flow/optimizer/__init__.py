"""Flow optimizer: opt-in rewrite pass (Catalyst-style).

Applies semantics-preserving optimization rules to produce a cheaper/faster
equivalent flow before execution.
"""

from .optimizer import Optimizer
from .rules import (
    CommonSubexpressionRule,
    DeadNodeEliminationRule,
    ModelTierRule,
    OptimizationRule,
    ParallelizeRule,
)

__all__ = [
    "CommonSubexpressionRule",
    "DeadNodeEliminationRule",
    "ModelTierRule",
    "OptimizationRule",
    "Optimizer",
    "ParallelizeRule",
]
