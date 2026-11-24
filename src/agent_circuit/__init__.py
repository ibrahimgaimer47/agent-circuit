"""agent-circuit: Zero-dependency circuit breaker and guardrail for AI agents and MCP tools."""

from .breaker import AgentCircuit, CircuitConfig, CircuitOpenError
from .pricing import DEFAULT_PRICING, ModelPricing, estimate_cost_usd

__version__ = "0.1.0"
__all__ = [
    "AgentCircuit",
    "CircuitConfig",
    "CircuitOpenError",
    "ModelPricing",
    "DEFAULT_PRICING",
    "estimate_cost_usd",
]
