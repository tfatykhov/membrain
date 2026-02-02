"""Tests for VectorStore protocol and compliance checking."""

import numpy as np
import pytest

from bench.protocol import VectorStore, check_protocol_compliance
from bench.baselines.cosine import CosineBaseline


class TestProtocolCompliance:
    """Verify protocol compliance checker works correctly."""
    
    def test_cosine_baseline_is_protocol_compliant(self) -> None:
        """CosineBaseline should pass all protocol checks."""
        store = CosineBaseline()
        errors = check_protocol_compliance(store, dim=64)
        assert errors == [], f"Protocol violations: {errors}"
    
    def test_cosine_baseline_isinstance_check(self) -> None:
        """CosineBaseline should be recognized as VectorStore."""
        store = CosineBaseline()
        assert isinstance(store, VectorStore)


class TestCosineBaseline:
    """Unit tests for CosineBaseline implementation."""
    
    @pytest.fixture
    def store(self) -> CosineBaseline:
        return CosineBaseline()
    
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)
    
    def test_empty_store(self, store: CosineBaseline) -> None:
        """Empty store should have count=0 and dim=0."""
        assert store.count == 0
        assert store.dim == 0
        assert store.memory_mb == 0.0
    
    def test_store_single_vector(self, store: CosineBaseline, rng: np.random.Generator) -> None:
        """Storing a vector should increment count and set dim."""
        vec = rng.standard_normal(64)
        store.store("test", vec)
        
        assert store.count == 1
        assert store.dim == 64
        assert store.memory_mb > 0
    
    def test_query_exact_match(self, store: CosineBaseline, rng: np.random.Generator) -> None:
        """Querying with exact stored vector should return score ≈ 1.0."""
        vec = rng.standard_normal(64)
        store.store("test", vec)
        
        results = store.query(vec, k=1)
        
        assert len(results) == 1
        assert results[0][0] == "test"
        assert results[0][1] >= 0.99  # Should be very close to 1.0
    
    def test_query_top_k(self, store: CosineBaseline, rng: np.random.Generator) -> None:
        """Query should return correct number of results."""
        for i in range(10):
            store.store(f"vec_{i}", rng.standard_normal(64))
        
        results = store.query(rng.standard_normal(64), k=5)
        
        assert len(results) == 5
        # Scores should be sorted descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_query_k_larger_than_count(self, store: CosineBaseline, rng: np.random.Generator) -> None:
        """Query with k > count should return all vectors."""
        for i in range(3):
            store.store(f"vec_{i}", rng.standard_normal(64))
        
        results = store.query(rng.standard_normal(64), k=10)
        
        assert len(results) == 3
    
    def test_clear(self, store: CosineBaseline, rng: np.random.Generator) -> None:
        """Clear should reset store to empty state."""
        store.store("test", rng.standard_normal(64))
        assert store.count == 1
        
        store.clear()
        
        assert store.count == 0
        assert store.dim == 0
    
    def test_duplicate_key_raises(self, store: CosineBaseline, rng: np.random.Generator) -> None:
        """Storing duplicate key should raise ValueError."""
        vec = rng.standard_normal(64)
        store.store("test", vec)
        
        with pytest.raises(ValueError, match="Duplicate key"):
            store.store("test", vec)
    
    def test_store_batch(self, store: CosineBaseline, rng: np.random.Generator) -> None:
        """Batch store should add all items."""
        items = [(f"vec_{i}", rng.standard_normal(64)) for i in range(10)]
        store.store_batch(items)
        
        assert store.count == 10
    
    def test_normalization(self, store: CosineBaseline) -> None:
        """Store should normalize vectors internally."""
        # Non-unit vector
        vec = np.array([3.0, 4.0, 0.0])  # Norm = 5
        store.store("test", vec)
        
        # Query with same direction, different magnitude
        query = np.array([6.0, 8.0, 0.0])  # Norm = 10
        results = store.query(query, k=1)
        
        assert results[0][0] == "test"
        assert results[0][1] >= 0.99  # Same direction = similarity 1.0
