# ⚡ agent-circuit

**Zero-dependency circuit breaker, infinite-loop detector, and token budget guardrail for AI agents and MCP tools.**

[![PyPI version](https://img.shields.io/pypi/v/agent-circuit)](https://pypi.org/project/agent-circuit/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Problem

The hardest part of running autonomous AI agents isn't building them — it's **controlling them at runtime**.

- Agents enter **infinite tool loops** (A → A → A) or **oscillating loops** (A → B → A → B) that most monitoring tools miss.
- A single runaway agent can silently burn **$50+ in tokens** in minutes.
- Existing observability platforms (Langfuse, LangSmith) **log the disaster after it happens** and require external cloud infrastructure.

## The Solution

`agent-circuit` is a **lightweight Python library** (pure stdlib, zero dependencies) that acts as a **circuit breaker** for any AI agent or MCP server:

| Guard | What it catches |
|---|---|
| 🔁 **Identical Loop Detection** | `A → A → A` — same tool, same args, repeated N times |
| 🏓 **Ping-Pong Loop Detection** | `A → B → A → B` — oscillating between two calls |
| 💰 **USD Budget Cap** | Trips when estimated spend exceeds your limit |
| 🎟️ **Token Budget Cap** | Trips when total input+output tokens exceed your limit |
| ⏱️ **Rate Limiting** | Trips when tool calls/minute exceed your threshold |
| 📊 **Live Telemetry** | `.summary()` returns real-time session stats |

---

## Installation

```bash
pip install agent-circuit
```

## Quick Start

```python
from agent_circuit import AgentCircuit, CircuitConfig, CircuitOpenError

circuit = AgentCircuit(CircuitConfig(
    max_budget_usd=1.00,
    max_total_tokens=100_000,
    max_identical_calls=3,
    max_ping_pong_cycles=2,
    max_calls_per_minute=30,
))

circuit.record_llm_usage(
    model="claude-sonnet-4",
    input_tokens=2000,
    output_tokens=500,
)

circuit.record_tool_call("web_search", {"query": "python async"})

@circuit.protect_tool()
def search_web(query: str) -> str:
    return call_search_api(query)

print(circuit.summary())
```

### When a limit is hit:

```python
try:
    circuit.record_tool_call("web_search", {"query": "python async"})
except CircuitOpenError as e:
    print(e.code)       # "IDENTICAL_TOOL_LOOP"
    print(e.reason)     # "Tool 'web_search' executed with identical arguments 3 times..."
    print(e.telemetry)  # Full session snapshot
```

## Built-in Pricing

Comes with up-to-date pricing for popular models. Override or extend at runtime:

```python
from agent_circuit import ModelPricing, CircuitConfig

config = CircuitConfig(
    custom_pricing={
        "my-fine-tuned-model": ModelPricing(
            input_per_1m=5.00,
            output_per_1m=15.00,
            cached_input_per_1m=0.50,
        ),
    }
)
```

## Running Tests

```bash
python -m unittest discover tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
