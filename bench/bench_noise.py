#!/usr/bin/env python3
"""Noise robustness benchmark runner.

Compares vector stores across noise levels to measure degradation.
Outputs CSV with hit@k, MRR, and latency metrics.

Usage:
    python -m bench.bench_noise --output results.csv
    python -m bench.bench_noise --noise-levels 0.0 0.1 0.2 0.3 --samples 500
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np

from bench.datasets import SyntheticDataset, add_noise
from bench.metrics import (
    BenchmarkResult,
    Timer,
    compute_hit_at_k,
    compute_reciprocal_rank,
    percentile,
)
from bench.protocol import VectorStore

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SAMPLES = 100
DEFAULT_DIM = 128
DEFAULT_K = 10
DEFAULT_NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
DEFAULT_SEED = 42
DEFAULT_DATASET = "gaussian"
VALID_DATASETS = ["gaussian", "clustered"]


def get_available_stores(dim: int) -> dict[str, type[VectorStore] | tuple[type[VectorStore], dict]]:
    """Discover available VectorStore implementations.
    
    Returns:
        Dict mapping method names to store classes or (class, kwargs) tuples.
        Only includes stores with available dependencies.
    """
    stores: dict[str, type[VectorStore] | tuple[type[VectorStore], dict]] = {}
    
    # Always available
    from bench.baselines.cosine import CosineBaseline
    stores["cosine"] = CosineBaseline
    
    # Optional: FAISS (requires dim argument)
    try:
        from bench.baselines.faiss_flat import FAISSFlatBaseline
        stores["faiss_flat"] = (FAISSFlatBaseline, {"dim": dim})
    except ImportError:
        logger.debug("FAISS not available, skipping faiss_flat baseline")
    
    # Optional: Membrain (requires running server)
    try:
        from bench.baselines.membrain_client import MembrainStore
        stores["membrain"] = MembrainStore
    except ImportError:
        logger.debug("Membrain client not available, skipping membrain baseline")
    
    return stores


def check_membrain_available(
    host: str = "localhost",
    port: int = 50051,
    timeout_s: float = 2.0,
) -> bool:
    """Check if Membrain gRPC server is reachable.
    
    Args:
        host: Server hostname
        port: Server port
        timeout_s: Connection timeout
        
    Returns:
        True if server responds to health check
    """
    try:
        import grpc
        from membrain_pb2 import PingRequest
        from membrain_pb2_grpc import MembrainServiceStub
        
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = MembrainServiceStub(channel)
        stub.Ping(PingRequest(), timeout=timeout_s)
        channel.close()
        return True
    except Exception:
        return False


def run_benchmark_single(
    store: VectorStore,
    dataset: SyntheticDataset,
    dataset_name: str,
    noise_level: float,
    seed: int,
    k: int = DEFAULT_K,
) -> BenchmarkResult:
    """Run benchmark for a single store at a single noise level.
    
    Args:
        store: VectorStore implementation to test
        dataset: Source vectors
        dataset_name: Name of dataset for result logging
        noise_level: Gaussian noise std to add to queries
        seed: RNG seed for reproducibility
        k: Max k for hit@k metrics
        
    Returns:
        BenchmarkResult with all metrics
    """
    rng = np.random.default_rng(seed)
    
    # Store all vectors - stream from dataset directly
    store.clear()
    for key, vector in dataset:
        store.store(key, vector)
    
    # Query with noise - re-iterate dataset
    
    # Query with noise - re-iterate dataset
    latencies_ms: list[float] = []
    hits_at_1: list[bool] = []
    hits_at_5: list[bool] = []
    hits_at_10: list[bool] = []
    reciprocal_ranks: list[float] = []
    
    for key, vector in dataset:
        # Add noise to query
        query = add_noise(vector, noise_level, rng)
        
        # Time the query
        with Timer() as t:
            results = store.query(query, k=k)
        
        latencies_ms.append(t.elapsed_ms)
        hits_at_1.append(compute_hit_at_k(key, results, 1))
        hits_at_5.append(compute_hit_at_k(key, results, 5))
        hits_at_10.append(compute_hit_at_k(key, results, 10))
        reciprocal_ranks.append(compute_reciprocal_rank(key, results))
    
    # Compute aggregates
    n_queries = len(latencies_ms)
    total_time_s = sum(latencies_ms) / 1000.0
    
    return BenchmarkResult(
        method=store.__class__.__name__,
        dataset=dataset_name,
        noise_level=noise_level,
        num_queries=n_queries,
        num_stored=store.count,
        dim=dataset.dim,
        hit_at_1=sum(hits_at_1) / n_queries,
        hit_at_5=sum(hits_at_5) / n_queries,
        hit_at_10=sum(hits_at_10) / n_queries,
        mrr=sum(reciprocal_ranks) / n_queries,
        avg_latency_ms=sum(latencies_ms) / n_queries,
        p50_latency_ms=percentile(latencies_ms, 50),
        p95_latency_ms=percentile(latencies_ms, 95),
        p99_latency_ms=percentile(latencies_ms, 99),
        min_latency_ms=min(latencies_ms),
        max_latency_ms=max(latencies_ms),
        memory_mb=store.memory_mb,
        throughput_qps=n_queries / total_time_s if total_time_s > 0 else 0.0,
        seed=seed,
    )


def run_benchmarks(
    samples: int = DEFAULT_SAMPLES,
    dim: int = DEFAULT_DIM,
    noise_levels: list[float] | None = None,
    seed: int = DEFAULT_SEED,
    methods: list[str] | None = None,
    skip_membrain: bool = False,
    dataset_type: str = DEFAULT_DATASET,
) -> list[BenchmarkResult]:
    """Run full benchmark suite across all stores and noise levels.
    
    Args:
        samples: Number of vectors to generate
        dim: Vector dimensionality
        noise_levels: List of noise levels to test
        seed: RNG seed
        methods: Filter to specific methods (None = all available)
        skip_membrain: Skip Membrain even if available
        dataset_type: Type of dataset ("gaussian" or "clustered")
        
    Returns:
        List of BenchmarkResult for each (method, noise_level) combination
    """
    if noise_levels is None:
        noise_levels = DEFAULT_NOISE_LEVELS
    
    # Discover available stores (pass dim for FAISS)
    available = get_available_stores(dim)
    
    # Filter to requested methods
    if methods:
        available = {k: v for k, v in available.items() if k in methods}
    
    # Skip Membrain if requested or unavailable
    if skip_membrain and "membrain" in available:
        del available["membrain"]
    elif "membrain" in available and not check_membrain_available():
        logger.warning("Membrain server not available, skipping membrain baseline")
        del available["membrain"]
    
    if not available:
        logger.error("No vector stores available to benchmark")
        return []
    
    logger.info(
        "benchmark_start",
        extra={
            "methods": list(available.keys()),
            "samples": samples,
            "dim": dim,
            "noise_levels": noise_levels,
            "seed": seed,
            "dataset": dataset_type,
        },
    )
    
    # Generate dataset
    if dataset_type == "clustered":
        dataset = SyntheticDataset.clustered(n=samples, dim=dim, seed=seed)
        dataset_name = "synthetic_clustered"
    else:
        dataset = SyntheticDataset.gaussian(n=samples, dim=dim, seed=seed)
        dataset_name = "synthetic_gaussian"
    
    results: list[BenchmarkResult] = []
    
    for method_name, store_entry in available.items():
        logger.info(f"Running benchmark for {method_name}...")
        
        try:
            # Handle stores that need kwargs (like FAISS with dim)
            if isinstance(store_entry, tuple):
                store_cls, kwargs = store_entry
                store = store_cls(**kwargs)
            else:
                store = store_entry()
        except Exception as e:
            logger.error(f"Failed to instantiate {method_name}: {e}")
            continue
        
        for noise_level in noise_levels:
            logger.info(f"  noise_level={noise_level:.2f}")
            
            try:
                result = run_benchmark_single(
                    store=store,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    noise_level=noise_level,
                    seed=seed,
                )
                results.append(result)
                result.log()
            except Exception as e:
                logger.error(f"Benchmark failed for {method_name} at noise={noise_level}: {e}")
    
    return results


def write_csv(results: list[BenchmarkResult], output_path: str) -> None:
    """Write results to CSV file.
    
    Args:
        results: List of benchmark results
        output_path: Path to output CSV file
    """
    if not results:
        logger.warning("No results to write")
        return
    
    fieldnames = list(asdict(results[0]).keys())
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            # Convert extra dict to string for CSV
            row["extra"] = str(row["extra"]) if row["extra"] else ""
            writer.writerow(row)
    
    logger.info(f"Results written to {output_path}")


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print human-readable summary to stdout.
    
    Args:
        results: List of benchmark results
    """
    if not results:
        print("No results to display")
        return
    
    # Group by method
    methods = sorted(set(r.method for r in results))
    noise_levels = sorted(set(r.noise_level for r in results))
    
    # Header
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Hit@1 table
    print("\nHit@1 by Noise Level:")
    print("-" * 60)
    header = f"{'Method':<20} " + " ".join(f"{n:.2f}" for n in noise_levels)
    print(header)
    print("-" * 60)
    
    for method in methods:
        method_results = [r for r in results if r.method == method]
        values = []
        for noise in noise_levels:
            r = next(
                (x for x in method_results if math.isclose(x.noise_level, noise)),
                None,
            )
            values.append(f"{r.hit_at_1:.2f}" if r else "N/A ")
        print(f"{method:<20} " + "  ".join(values))
    
    # Latency summary
    print("\nLatency (avg ms) by Noise Level:")
    print("-" * 60)
    print(header)
    print("-" * 60)
    
    for method in methods:
        method_results = [r for r in results if r.method == method]
        values = []
        for noise in noise_levels:
            r = next(
                (x for x in method_results if math.isclose(x.noise_level, noise)),
                None,
            )
            values.append(f"{r.avg_latency_ms:.2f}" if r else "N/A ")
        print(f"{method:<20} " + "  ".join(values))
    
    print("\n" + "=" * 80)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark vector stores across noise levels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of vectors to benchmark",
    )
    parser.add_argument(
        "--dim", "-d",
        type=int,
        default=DEFAULT_DIM,
        help="Vector dimensionality",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=DEFAULT_NOISE_LEVELS,
        help="Noise levels to test",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--methods", "-m",
        type=str,
        nargs="+",
        default=None,
        help="Methods to benchmark (default: all available)",
    )
    parser.add_argument(
        "--skip-membrain",
        action="store_true",
        help="Skip Membrain even if server is available",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=VALID_DATASETS,
        default=DEFAULT_DATASET,
        help="Dataset type: 'gaussian' (easy) or 'clustered' (realistic)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    
    # Run benchmarks
    results = run_benchmarks(
        samples=args.samples,
        dim=args.dim,
        noise_levels=args.noise_levels,
        seed=args.seed,
        methods=args.methods,
        skip_membrain=args.skip_membrain,
        dataset_type=args.dataset,
    )
    
    if not results:
        logger.error("No benchmark results generated")
        return 1
    
    # Write output
    if args.output:
        write_csv(results, args.output)
    
    # Print summary
    print_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
