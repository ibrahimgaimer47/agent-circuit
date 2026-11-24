"""Core circuit breaker for autonomous AI agents and tool execution."""

from __future__ import annotations

import functools
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from .pricing import ModelPricing, estimate_cost_usd

F = TypeVar("F", bound=Callable[..., Any])


class CircuitOpenError(RuntimeError):
    """Raised when an agent violates budget, rate, or loop safety thresholds."""

    def __init__(self, reason: str, code: str, telemetry: dict[str, Any]) -> None:
        super().__init__(f"[agent-circuit:{code}] {reason}")
        self.reason = reason
        self.code = code
        self.telemetry = telemetry


@dataclass
class CircuitConfig:
    max_budget_usd: float = 2.00
    max_total_tokens: int = 250_000
    max_identical_calls: int = 3
    max_ping_pong_cycles: int = 2
    max_calls_per_minute: int = 40
    loop_window_size: int = 12
    custom_pricing: dict[str, ModelPricing] = field(default_factory=dict)


class AgentCircuit:
    """Thread-safe local guardrail for tool loops, rate spikes, and token spend."""

    def __init__(self, config: CircuitConfig | None = None) -> None:
        self.config = config or CircuitConfig()
        self._lock = threading.Lock()
        self._tripped_reason: str | None = None
        self._tripped_code: str | None = None

        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.total_tool_calls: int = 0

        self._recent_signatures: deque[str] = deque(maxlen=self.config.loop_window_size)
        self._call_timestamps: deque[float] = deque()

    @property
    def is_open(self) -> bool:
        return self._tripped_reason is not None

    def reset(self) -> None:
        """Reset all counters and close the circuit."""
        with self._lock:
            self._tripped_reason = None
            self._tripped_code = None
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cached_tokens = 0
            self.total_cost_usd = 0.0
            self.total_tool_calls = 0
            self._recent_signatures.clear()
            self._call_timestamps.clear()

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of current session telemetry."""
        return {
            "circuit_open": self.is_open,
            "trip_code": self._tripped_code,
            "trip_reason": self._tripped_reason,
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cached_tokens": self.total_cached_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
        }

    def _trip(self, code: str, reason: str) -> None:
        self._tripped_code = code
        self._tripped_reason = reason
        raise CircuitOpenError(reason=reason, code=code, telemetry=self.summary())

    def record_llm_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Track LLM usage and trip immediately if token or USD budget is exceeded."""
        with self._lock:
            if self.is_open:
                raise CircuitOpenError(
                    reason=self._tripped_reason or "Circuit already open",
                    code=self._tripped_code or "ALREADY_OPEN",
                    telemetry=self.summary(),
                )

            call_cost = estimate_cost_usd(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                custom_pricing=self.config.custom_pricing,
            )

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cached_tokens += cached_tokens
            self.total_cost_usd += call_cost

            total_tokens = self.total_input_tokens + self.total_output_tokens
            if total_tokens > self.config.max_total_tokens:
                self._trip(
                    "MAX_TOKENS_EXCEEDED",
                    f"Token usage {total_tokens:,} exceeded limit of {self.config.max_total_tokens:,}.",
                )

            if self.total_cost_usd > self.config.max_budget_usd:
                self._trip(
                    "MAX_BUDGET_EXCEEDED",
                    f"Estimated spend ${self.total_cost_usd:.4f} exceeded budget limit of ${self.config.max_budget_usd:.2f}.",
                )

            return call_cost

    def _compute_signature(self, tool_name: str, args: Any, kwargs: Any) -> str:
        """Create a deterministic hash signature for tool invocations."""
        def serialize_safe(obj: Any) -> Any:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return repr(obj)

        payload = {
            "name": tool_name,
            "args": serialize_safe(args),
            "kwargs": serialize_safe(kwargs),
        }
        serialized = json.dumps(payload, sort_keys=True, default=repr)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def record_tool_call(
        self,
        tool_name: str,
        args: Any = None,
        kwargs: Any = None,
    ) -> None:
        """Record and validate a tool execution against loop and rate policies."""
        with self._lock:
            if self.is_open:
                raise CircuitOpenError(
                    reason=self._tripped_reason or "Circuit already open",
                    code=self._tripped_code or "ALREADY_OPEN",
                    telemetry=self.summary(),
                )

            now = time.monotonic()

            # 1. Rate Limiting Check
            while self._call_timestamps and (now - self._call_timestamps[0]) > 60.0:
                self._call_timestamps.popleft()

            if len(self._call_timestamps) >= self.config.max_calls_per_minute:
                self._trip(
                    "RATE_LIMIT_EXCEEDED",
                    f"Tool execution rate exceeded {self.config.max_calls_per_minute} calls/minute.",
                )

            self._call_timestamps.append(now)

            # 2. Signature computation & tracking
            sig = self._compute_signature(tool_name, args, kwargs)
            self._recent_signatures.append(sig)
            self.total_tool_calls += 1

            # 3. Direct identical loop detection (A -> A -> A)
            if len(self._recent_signatures) >= self.config.max_identical_calls:
                identical_count = 0
                for s in reversed(self._recent_signatures):
                    if s == sig:
                        identical_count += 1
                    else:
                        break
                if identical_count >= self.config.max_identical_calls:
                    self._trip(
                        "IDENTICAL_TOOL_LOOP",
                        f"Tool '{tool_name}' executed with identical arguments {identical_count} times in succession.",
                    )

            # 4. Oscillating ping-pong loop detection (A -> B -> A -> B)
            cycles = self.config.max_ping_pong_cycles
            window_len = cycles * 2
            if len(self._recent_signatures) >= window_len:
                recent_list = list(self._recent_signatures)[-window_len:]
                a = recent_list[0]
                b = recent_list[1]
                if a != b:
                    is_ping_pong = True
                    for i in range(cycles):
                        if recent_list[2 * i] != a or recent_list[2 * i + 1] != b:
                            is_ping_pong = False
                            break
                    if is_ping_pong:
                        self._trip(
                            "PING_PONG_LOOP",
                            f"Oscillating ping-pong loop detected: alternating between calls {cycles} consecutive times.",
                        )

    def protect_tool(self, tool_name: str | None = None) -> Callable[[F], F]:
        """Decorator to automatically guard any tool or function execution."""
        def decorator(fn: F) -> F:
            name = tool_name or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.record_tool_call(tool_name=name, args=args, kwargs=kwargs)
                return fn(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator
