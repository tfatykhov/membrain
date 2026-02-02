"""Benchmark for attractor cleanup effectiveness.

Measures how much AttractorMemory improves pattern matching
compared to direct similarity search (no cleanup).

This validates whether attractor dynamics add value before
integrating with the full Membrain pipeline.

NOTE: The numpy Hopfield prototype has limited capacity (~0.14N patterns).
Expect poor performance when num_patterns > 0.1 * dim. The Nengo
implementation will use more sophisticated dynamics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from membrain.attractor import AttractorMemory

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_PATTERNS = 20
DEFAULT_DIM = 128
DEFAULT_NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_QUERIES_PER_PATTERN = 5
DEFAULT_SEED = 42
DEFAULT_LEARNING_RATE = 0.3
DEFAULT_MAX_STEPS = 100


@dataclass
class CleanupBenchResult:
    """Result for a single noise level."""

    noise_level: float
    num_patterns: int
    num_queries: int
    dim: int

    # Accuracy metrics
    hit_at_1_direct: float
    """Hit@1 using direct similarity (no cleanup)."""

    hit_at_1_cleaned: float
    """Hit@1 after attractor cleanup."""

    hit_at_1_gain: float
    """Improvement from cleanup."""

    # Similarity metrics
    avg_input_similarity: float
    """Average cosine similarity between original and noisy."""

    avg_output_similarity: float
    """Average cosine similarity between original and cleaned."""

    avg_cleanup_gain: float
    """Average improvement in similarity from cleanup."""

    # Convergence
    avg_steps: float
    """Average steps to convergence."""

    converged_pct: float
    """Percentage of queries that converged."""

    seed: int


def cosine_similarity(a: NDArray, b: NDArray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_best_match(
    query: NDArray,
    patterns_matrix: NDArray,
) -> int:
    """Find index of most similar pattern using vectorized similarity.
    
    Args:
        query: Query vector (normalized)
        patterns_matrix: Matrix of patterns, shape (num_patterns, dim)
        
    Returns:
        Index of best matching pattern
    """
    # Vectorized dot product: (N, D) @ (D,) -> (N,)
    similarities = patterns_matrix @ query
    return int(np.argmax(similarities))


def run_cleanup_benchmark(
    num_patterns: int = DEFAULT_PATTERNS,
    dim: int = DEFAULT_DIM,
    noise_levels: list[float] | None = None,
    queries_per_pattern: int = DEFAULT_QUERIES_PER_PATTERN,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
) -> list[CleanupBenchResult]:
    """Run cleanup benchmark across noise levels.

    Args:
        num_patterns: Number of patterns to store
        dim: Pattern dimensionality
        noise_levels: List of noise standard deviations to test
        queries_per_pattern: Queries per pattern per noise level
        learning_rate: Hebbian learning rate
        max_steps: Max dynamics iterations
        seed: RNG seed

    Returns:
        List of results per noise level
    """
    if noise_levels is None:
        noise_levels = DEFAULT_NOISE_LEVELS

    rng = np.random.default_rng(seed)

    # Generate patterns
    patterns: list[NDArray[np.float32]] = []
    for _ in range(num_patterns):
        p = rng.standard_normal(dim).astype(np.float32)
        p /= np.linalg.norm(p)
        patterns.append(p)

    # Stack into matrix for vectorized similarity search
    patterns_matrix = np.stack(patterns, axis=0)  # (num_patterns, dim)

    # Create and populate attractor memory
    attractor = AttractorMemory(
        dimensions=dim,
        learning_rate=learning_rate,
        max_steps=max_steps,
        seed=seed,
    )
    for p in patterns:
        attractor.store(p)

    logger.info(
        "Benchmark setup complete",
        extra={
            "num_patterns": num_patterns,
            "dim": dim,
            "noise_levels": noise_levels,
        },
    )

    results: list[CleanupBenchResult] = []

    for noise_level in noise_levels:
        hits_direct = 0
        hits_cleaned = 0
        input_sims: list[float] = []
        output_sims: list[float] = []
        cleanup_gains: list[float] = []
        steps_list: list[int] = []
        converged_count = 0

        total_queries = num_patterns * queries_per_pattern

        for pattern_idx, pattern in enumerate(patterns):
            for _ in range(queries_per_pattern):
                # Generate noisy query
                noise = noise_level * rng.standard_normal(dim).astype(np.float32)
                noisy = pattern + noise
                noisy_norm = np.linalg.norm(noisy)
                if noisy_norm > 1e-10:
                    noisy = noisy / noisy_norm

                # Direct matching (no cleanup)
                direct_match = find_best_match(noisy, patterns_matrix)
                if direct_match == pattern_idx:
                    hits_direct += 1

                # Cleanup then match
                result = attractor.complete(noisy)
                cleaned_match = find_best_match(result.cleaned, patterns_matrix)
                if cleaned_match == pattern_idx:
                    hits_cleaned += 1

                # Track metrics
                input_sim = cosine_similarity(pattern, noisy)
                output_sim = cosine_similarity(pattern, result.cleaned)
                input_sims.append(input_sim)
                output_sims.append(output_sim)
                cleanup_gains.append(output_sim - input_sim)
                steps_list.append(result.steps)
                if result.converged:
                    converged_count += 1

        # Aggregate
        result = CleanupBenchResult(
            noise_level=noise_level,
            num_patterns=num_patterns,
            num_queries=total_queries,
            dim=dim,
            hit_at_1_direct=hits_direct / total_queries,
            hit_at_1_cleaned=hits_cleaned / total_queries,
            hit_at_1_gain=(hits_cleaned - hits_direct) / total_queries,
            avg_input_similarity=float(np.mean(input_sims)),
            avg_output_similarity=float(np.mean(output_sims)),
            avg_cleanup_gain=float(np.mean(cleanup_gains)),
            avg_steps=float(np.mean(steps_list)),
            converged_pct=converged_count / total_queries * 100,
            seed=seed,
        )
        results.append(result)

        logger.info(
            "Noise level complete",
            extra={
                "noise": noise_level,
                "hit@1_direct": result.hit_at_1_direct,
                "hit@1_cleaned": result.hit_at_1_cleaned,
                "gain": result.hit_at_1_gain,
            },
        )

    return results


def print_results(results: list[CleanupBenchResult]) -> None:
    """Print results as formatted table."""
    print("\n" + "=" * 80)
    print("CLEANUP BENCHMARK RESULTS")
    print("=" * 80)

    # Hit@1 comparison
    print("\nHit@1 Accuracy:")
    print("-" * 60)
    print(f"{'Noise':>8} {'Direct':>10} {'Cleaned':>10} {'Gain':>10}")
    print("-" * 60)
    for r in results:
        gain_str = f"{r.hit_at_1_gain:+.2f}"
        print(f"{r.noise_level:>8.2f} {r.hit_at_1_direct:>10.2f} {r.hit_at_1_cleaned:>10.2f} {gain_str:>10}")

    # Similarity metrics
    print("\nSimilarity Metrics:")
    print("-" * 60)
    print(f"{'Noise':>8} {'Input Sim':>12} {'Output Sim':>12} {'Gain':>10}")
    print("-" * 60)
    for r in results:
        gain_str = f"{r.avg_cleanup_gain:+.3f}"
        print(f"{r.noise_level:>8.2f} {r.avg_input_similarity:>12.3f} {r.avg_output_similarity:>12.3f} {gain_str:>10}")

    # Convergence
    print("\nConvergence:")
    print("-" * 60)
    print(f"{'Noise':>8} {'Avg Steps':>12} {'Converged %':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r.noise_level:>8.2f} {r.avg_steps:>12.1f} {r.converged_pct:>12.1f}")

    print("=" * 80)


def write_json(results: list[CleanupBenchResult], path: Path) -> None:
    """Write results to JSON file."""
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results written to {path}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark attractor cleanup effectiveness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--num-patterns",
        type=int,
        default=DEFAULT_PATTERNS,
        help="Number of patterns to store",
    )
    parser.add_argument(
        "-d", "--dim",
        type=int,
        default=DEFAULT_DIM,
        help="Pattern dimensionality",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=DEFAULT_NOISE_LEVELS,
        help="Noise levels to test",
    )
    parser.add_argument(
        "-q", "--queries-per-pattern",
        type=int,
        default=DEFAULT_QUERIES_PER_PATTERN,
        help="Queries per pattern per noise level",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Hebbian learning rate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max dynamics iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Run benchmark
    results = run_cleanup_benchmark(
        num_patterns=args.num_patterns,
        dim=args.dim,
        noise_levels=args.noise_levels,
        queries_per_pattern=args.queries_per_pattern,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Output
    print_results(results)

    if args.output:
        write_json(results, args.output)

    # Return non-zero if cleanup hurt accuracy at any level
    for r in results:
        if r.hit_at_1_gain < -0.1:  # Allow small regression
            logger.warning(f"Cleanup hurt accuracy at noise={r.noise_level}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
