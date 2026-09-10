"""Risk-set and support-aware CUSUM-RL building blocks.

Phase 1 exposes construction, support diagnostics, FQI, toy statistics, and a
cluster multiplier null generator.  It deliberately exposes no p-value or
OhioT1DM experiment runner.
"""

from .types import RiskSetPanel, SupportPanel, SupportTransition, TransitionRecord

__all__ = ["RiskSetPanel", "SupportPanel", "SupportTransition", "TransitionRecord"]
