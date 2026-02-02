"""Metrics collection and timing utilities for benchmarks.

Provides structured result storage and precise timing for latency measurements.
"""

from dataclasses import dataclass, asdict, field
from time import perf_counter_ns
from typing import Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.
    
    Captures accuracy, latency, and resource usage for one configuration
    (method + dataset + noise level).
    """
    
    method: str
    dataset: str
    noise_level: float
    num_queries: int
    num_stored: int
    dim: int
    
    # Accuracy metrics
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr: float  # Mean Reciprocal Rank
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Resource metrics
    memory_mb: float
    throughput_qps: float
    
    # Metadata
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def log(self) -> None:
        """Emit structured log entry."""
        logger.info(
            "benchmark_result",
            extra={
                "method": self.method,
                "dataset": self.dataset,
                "noise_level": self.noise_level,
                "hit_at_1": self.hit_at_1,
                "hit_at_5": self.hit_at_5,
                "mrr": self.mrr,
                "avg_latency_ms": self.avg_latency_ms,
                "p99_latency_ms": self.p99_latency_ms,
                "memory_mb": self.memory_mb,
                "throughput_qps": self.throughput_qps,
            }
        )


class Timer:
    """Context manager for precise timing.
    
    Uses perf_counter_ns for nanosecond precision.
    
    Example:
        with Timer() as t:
            do_something()
        print(f"Took {t.elapsed_ms:.2f}ms")
    """
    
    def __init__(self) -> None:
        self._start: int = 0
        self._end: int = 0
    
    def __enter__(self) -> "Timer":
        self._start = perf_counter_ns()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self._end = perf_counter_ns()
    
    @property
    def elapsed_ns(self) -> int:
        """Elapsed time in nanoseconds."""
        return self._end - self._start
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000
    
    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000


def compute_hit_at_k(
    expected_key: str, 
    results: list[tuple[str, float]], 
    k: int
) -> bool:
    """Check if expected key is in top-k results.
    
    Args:
        expected_key: The correct answer
        results: List of (key, score) from query
        k: Consider top-k results
        
    Returns:
        True if expected_key is in top-k
    """
    top_k_keys = [r[0] for r in results[:k]]
    return expected_key in top_k_keys


def compute_reciprocal_rank(
    expected_key: str, 
    results: list[tuple[str, float]]
) -> float:
    """Compute reciprocal rank for a single query.
    
    Args:
        expected_key: The correct answer
        results: List of (key, score) from query
        
    Returns:
        1/rank if found, 0 if not found
    """
    for i, (key, _) in enumerate(results):
        if key == expected_key:
            return 1.0 / (i + 1)
    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Compute percentile of a list of values.
    
    Args:
        values: List of numeric values
        p: Percentile (0-100)
        
    Returns:
        The p-th percentile value
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * p / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]
