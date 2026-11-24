"""Tests for agent_circuit core circuit breaker and loop detection."""

import os
import sys
import threading
import unittest

# Ensure src/ is on sys.path for direct test runs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent_circuit import (
    AgentCircuit,
    CircuitConfig,
    CircuitOpenError,
    estimate_cost_usd,
)


class TestAgentCircuit(unittest.TestCase):
    def test_pricing_calculation(self):
        cost = estimate_cost_usd("claude-sonnet-4", input_tokens=10_000, output_tokens=0)
        self.assertAlmostEqual(cost, 0.03, places=5)

        cost_prefix = estimate_cost_usd("gpt-4o-2024-08-06", input_tokens=1_000_000, output_tokens=0)
        self.assertAlmostEqual(cost_prefix, 2.50, places=5)

    def test_identical_tool_loop_trips(self):
        config = CircuitConfig(max_identical_calls=3)
        circuit = AgentCircuit(config)

        circuit.record_tool_call("web_search", {"query": "python"})
        circuit.record_tool_call("web_search", {"query": "python"})
        self.assertFalse(circuit.is_open)

        with self.assertRaises(CircuitOpenError) as ctx:
            circuit.record_tool_call("web_search", {"query": "python"})

        self.assertEqual(ctx.exception.code, "IDENTICAL_TOOL_LOOP")
        self.assertTrue(circuit.is_open)

    def test_ping_pong_loop_trips(self):
        config = CircuitConfig(max_ping_pong_cycles=2, loop_window_size=10)
        circuit = AgentCircuit(config)

        circuit.record_tool_call("tool_a", {"step": 1})
        circuit.record_tool_call("tool_b", {"step": 2})
        self.assertFalse(circuit.is_open)

        circuit.record_tool_call("tool_a", {"step": 1})
        with self.assertRaises(CircuitOpenError) as ctx:
            circuit.record_tool_call("tool_b", {"step": 2})

        self.assertEqual(ctx.exception.code, "PING_PONG_LOOP")
        self.assertTrue(circuit.is_open)

    def test_token_budget_trips(self):
        config = CircuitConfig(max_total_tokens=10_000)
        circuit = AgentCircuit(config)

        circuit.record_llm_usage("gpt-4o-mini", input_tokens=4_000, output_tokens=2_000)
        self.assertFalse(circuit.is_open)

        with self.assertRaises(CircuitOpenError) as ctx:
            circuit.record_llm_usage("gpt-4o-mini", input_tokens=3_000, output_tokens=2_000)

        self.assertEqual(ctx.exception.code, "MAX_TOKENS_EXCEEDED")
        self.assertTrue(circuit.is_open)

    def test_usd_budget_trips(self):
        config = CircuitConfig(max_budget_usd=0.05)
        circuit = AgentCircuit(config)

        circuit.record_llm_usage("claude-opus-4", input_tokens=2_000, output_tokens=0)
        self.assertFalse(circuit.is_open)

        with self.assertRaises(CircuitOpenError) as ctx:
            circuit.record_llm_usage("claude-opus-4", input_tokens=2_000, output_tokens=0)

        self.assertEqual(ctx.exception.code, "MAX_BUDGET_EXCEEDED")
        self.assertTrue(circuit.is_open)

    def test_rate_limiting(self):
        config = CircuitConfig(max_calls_per_minute=5)
        circuit = AgentCircuit(config)

        for i in range(5):
            circuit.record_tool_call(f"tool_{i}", {"val": i})

        self.assertFalse(circuit.is_open)

        with self.assertRaises(CircuitOpenError) as ctx:
            circuit.record_tool_call("tool_extra", {"val": 99})

        self.assertEqual(ctx.exception.code, "RATE_LIMIT_EXCEEDED")
        self.assertTrue(circuit.is_open)

    def test_decorator_protection(self):
        circuit = AgentCircuit(CircuitConfig(max_identical_calls=2))

        @circuit.protect_tool()
        def calculate_sum(a: int, b: int) -> int:
            return a + b

        self.assertEqual(calculate_sum(1, 2), 3)

        with self.assertRaises(CircuitOpenError) as ctx:
            calculate_sum(1, 2)

        self.assertEqual(ctx.exception.code, "IDENTICAL_TOOL_LOOP")

    def test_reset_functionality(self):
        circuit = AgentCircuit(CircuitConfig(max_budget_usd=0.01))
        with self.assertRaises(CircuitOpenError):
            circuit.record_llm_usage("claude-opus-4", input_tokens=2_000, output_tokens=0)

        self.assertTrue(circuit.is_open)
        circuit.reset()
        self.assertFalse(circuit.is_open)
        summary = circuit.summary()
        self.assertFalse(summary["circuit_open"])
        self.assertEqual(summary["estimated_cost_usd"], 0.0)

    def test_thread_safety(self):
        circuit = AgentCircuit(CircuitConfig(max_calls_per_minute=1000, max_identical_calls=500))

        def worker(worker_id: int):
            for i in range(20):
                circuit.record_tool_call(f"worker_tool_{worker_id}", {"index": i})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(circuit.total_tool_calls, 100)


if __name__ == "__main__":
    unittest.main()
