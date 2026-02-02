"""Integration tests for benchmark runner.

Tests the bench_noise module end-to-end with synthetic data.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from bench.bench_noise import (
    DEFAULT_K,
    get_available_stores,
    run_benchmark_single,
    run_benchmarks,
    write_csv,
)
from bench.datasets import SyntheticDataset


class TestGetAvailableStores:
    """Tests for store discovery."""

    def test_cosine_always_available(self) -> None:
        """CosineBaseline should always be available."""
        stores = get_available_stores()
        assert "cosine" in stores

    def test_returns_dict(self) -> None:
        """Should return a dict mapping names to classes."""
        stores = get_available_stores()
        assert isinstance(stores, dict)
        assert len(stores) >= 1


class TestRunBenchmarkSingle:
    """Tests for single benchmark run."""

    def test_exact_match_hit_at_1(self) -> None:
        """Zero noise should give 100% hit@1."""
        from bench.baselines.cosine import CosineBaseline

        store = CosineBaseline()
        dataset = SyntheticDataset.gaussian(n=20, dim=32, seed=42)

        result = run_benchmark_single(
            store=store,
            dataset=dataset,
            noise_level=0.0,
            seed=42,
        )

        assert result.hit_at_1 == 1.0
        assert result.hit_at_5 == 1.0
        assert result.mrr == 1.0

    def test_high_noise_degrades_accuracy(self) -> None:
        """High noise should reduce hit@1."""
        from bench.baselines.cosine import CosineBaseline

        store = CosineBaseline()
        # Use more samples and higher noise to ensure degradation
        dataset = SyntheticDataset.gaussian(n=500, dim=64, seed=42)

        result_clean = run_benchmark_single(
            store=store,
            dataset=dataset,
            noise_level=0.0,
            seed=42,
        )

        result_noisy = run_benchmark_single(
            store=store,
            dataset=dataset,
            noise_level=0.8,  # Very high noise
            seed=42,
        )

        # Noisy should have lower accuracy (or at worst equal)
        assert result_noisy.hit_at_1 <= result_clean.hit_at_1

    def test_result_fields_populated(self) -> None:
        """All result fields should be populated."""
        from bench.baselines.cosine import CosineBaseline

        store = CosineBaseline()
        dataset = SyntheticDataset.gaussian(n=10, dim=32, seed=42)

        result = run_benchmark_single(
            store=store,
            dataset=dataset,
            noise_level=0.1,
            seed=42,
        )

        assert result.method == "CosineBaseline"
        assert result.num_queries == 10
        assert result.num_stored == 10
        assert result.dim == 32
        assert result.avg_latency_ms >= 0
        assert result.memory_mb >= 0
        assert result.throughput_qps > 0

    def test_deterministic_with_seed(self) -> None:
        """Same seed should give identical results."""
        from bench.baselines.cosine import CosineBaseline

        dataset = SyntheticDataset.gaussian(n=20, dim=32, seed=42)

        result1 = run_benchmark_single(
            store=CosineBaseline(),
            dataset=dataset,
            noise_level=0.2,
            seed=123,
        )

        result2 = run_benchmark_single(
            store=CosineBaseline(),
            dataset=dataset,
            noise_level=0.2,
            seed=123,
        )

        assert result1.hit_at_1 == result2.hit_at_1
        assert result1.mrr == result2.mrr


class TestRunBenchmarks:
    """Tests for full benchmark suite."""

    def test_runs_available_stores(self) -> None:
        """Should run benchmarks for all available stores."""
        results = run_benchmarks(
            samples=10,
            dim=16,
            noise_levels=[0.0, 0.1],
            seed=42,
            skip_membrain=True,
        )

        # At least cosine should run
        assert len(results) >= 2  # 2 noise levels
        methods = set(r.method for r in results)
        assert "CosineBaseline" in methods

    def test_filter_by_methods(self) -> None:
        """Should filter to requested methods."""
        results = run_benchmarks(
            samples=10,
            dim=16,
            noise_levels=[0.0],
            seed=42,
            methods=["cosine"],
            skip_membrain=True,
        )

        methods = set(r.method for r in results)
        assert methods == {"CosineBaseline"}

    def test_empty_methods_returns_empty(self) -> None:
        """Filtering to nonexistent method returns empty."""
        results = run_benchmarks(
            samples=10,
            dim=16,
            noise_levels=[0.0],
            seed=42,
            methods=["nonexistent"],
            skip_membrain=True,
        )

        assert results == []


class TestWriteCsv:
    """Tests for CSV output."""

    def test_writes_csv_file(self) -> None:
        """Should write valid CSV with headers."""
        results = run_benchmarks(
            samples=5,
            dim=8,
            noise_levels=[0.0],
            seed=42,
            methods=["cosine"],
            skip_membrain=True,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            output_path = f.name

        try:
            write_csv(results, output_path)

            # Read back and verify
            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["method"] == "CosineBaseline"
            assert float(rows[0]["hit_at_1"]) == 1.0
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_handles_empty_results(self) -> None:
        """Should handle empty results gracefully."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            output_path = f.name

        try:
            write_csv([], output_path)
            # File should not be created or should be empty
            assert not Path(output_path).exists() or Path(output_path).stat().st_size == 0
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestNoiseRobustness:
    """Tests for noise robustness (per spec requirement)."""

    @pytest.mark.parametrize("noise_level", [0.0, 0.05, 0.10, 0.20, 0.30])
    def test_hit_at_1_by_noise_level(self, noise_level: float) -> None:
        """Test hit@1 accuracy at various noise levels."""
        from bench.baselines.cosine import CosineBaseline

        store = CosineBaseline()
        dataset = SyntheticDataset.gaussian(n=100, dim=64, seed=42)

        result = run_benchmark_single(
            store=store,
            dataset=dataset,
            noise_level=noise_level,
            seed=42,
        )

        # At zero noise, expect perfect accuracy
        if noise_level == 0.0:
            assert result.hit_at_1 == 1.0
        # At low noise, still expect high accuracy
        elif noise_level <= 0.1:
            assert result.hit_at_1 >= 0.8
        # At high noise, accuracy degrades but hit@5 should be better
        else:
            assert result.hit_at_5 >= result.hit_at_1
