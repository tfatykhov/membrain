"""Tests for FAISS baselines.

These tests are skipped if faiss is not installed.
"""

import numpy as np
import pytest

# Skip all tests if faiss not available
faiss = pytest.importorskip("faiss")

from bench.baselines.faiss_flat import FAISSFlatBaseline, FAISSIVFBaseline
from bench.protocol import VectorStore, check_protocol_compliance


class TestFAISSFlatBaseline:
    """Unit tests for FAISSFlatBaseline."""
    
    @pytest.fixture
    def store(self) -> FAISSFlatBaseline:
        return FAISSFlatBaseline(dim=64)
    
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)
    
    def test_protocol_compliance(self, store: FAISSFlatBaseline) -> None:
        """FAISSFlatBaseline should pass protocol compliance."""
        errors = check_protocol_compliance(store, dim=64)
        assert errors == [], f"Protocol violations: {errors}"
    
    def test_isinstance_check(self, store: FAISSFlatBaseline) -> None:
        """Should be recognized as VectorStore."""
        assert isinstance(store, VectorStore)
    
    def test_empty_store(self, store: FAISSFlatBaseline) -> None:
        """Empty store should have count=0."""
        assert store.count == 0
        assert store.dim == 64
    
    def test_store_and_query(self, store: FAISSFlatBaseline, rng: np.random.Generator) -> None:
        """Basic store and query."""
        vec = rng.standard_normal(64)
        store.store("test", vec)
        
        assert store.count == 1
        
        results = store.query(vec, k=1)
        assert len(results) == 1
        assert results[0][0] == "test"
        assert results[0][1] >= 0.99
    
    def test_batch_store(self, store: FAISSFlatBaseline, rng: np.random.Generator) -> None:
        """Batch insert should work."""
        items = [(f"vec_{i}", rng.standard_normal(64)) for i in range(100)]
        store.store_batch(items)
        
        assert store.count == 100
    
    def test_query_top_k(self, store: FAISSFlatBaseline, rng: np.random.Generator) -> None:
        """Query should return k results sorted by score."""
        for i in range(20):
            store.store(f"vec_{i}", rng.standard_normal(64))
        
        results = store.query(rng.standard_normal(64), k=5)
        
        assert len(results) == 5
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_duplicate_key_raises(self, store: FAISSFlatBaseline, rng: np.random.Generator) -> None:
        """Duplicate key should raise ValueError."""
        vec = rng.standard_normal(64)
        store.store("test", vec)
        
        with pytest.raises(ValueError, match="Duplicate key"):
            store.store("test", vec)
    
    def test_duplicate_key_in_batch_raises(self, store: FAISSFlatBaseline, rng: np.random.Generator) -> None:
        """Duplicate keys within a batch should raise ValueError."""
        items = [
            ("vec_1", rng.standard_normal(64)),
            ("vec_2", rng.standard_normal(64)),
            ("vec_1", rng.standard_normal(64)),  # Duplicate
        ]
        
        with pytest.raises(ValueError, match="Duplicate key in batch"):
            store.store_batch(items)
    
    def test_dimension_mismatch_raises(self, store: FAISSFlatBaseline, rng: np.random.Generator) -> None:
        """Wrong dimension should raise ValueError."""
        items = [
            ("vec_1", rng.standard_normal(64)),
            ("vec_2", rng.standard_normal(32)),  # Wrong dim
        ]
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            store.store_batch(items)
    
    def test_clear(self, store: FAISSFlatBaseline, rng: np.random.Generator) -> None:
        """Clear should reset the index."""
        store.store("test", rng.standard_normal(64))
        assert store.count == 1
        
        store.clear()
        
        assert store.count == 0
    
    def test_memory_reporting(self, store: FAISSFlatBaseline, rng: np.random.Generator) -> None:
        """Memory should increase with stored vectors."""
        initial_mem = store.memory_mb
        
        items = [(f"vec_{i}", rng.standard_normal(64)) for i in range(1000)]
        store.store_batch(items)
        
        assert store.memory_mb > initial_mem


class TestFAISSIVFBaseline:
    """Unit tests for FAISSIVFBaseline (approximate search)."""
    
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)
    
    @pytest.fixture
    def trained_store(self, rng: np.random.Generator) -> FAISSIVFBaseline:
        """Return a trained IVF index."""
        store = FAISSIVFBaseline(dim=64, nlist=10, nprobe=5)
        # Train with random vectors (need ~30× nlist)
        training_data = rng.standard_normal((500, 64))
        store.train(training_data)
        return store
    
    def test_requires_training(self, rng: np.random.Generator) -> None:
        """Store should fail if not trained."""
        store = FAISSIVFBaseline(dim=64, nlist=10)
        
        with pytest.raises(RuntimeError, match="must be trained"):
            store.store("test", rng.standard_normal(64))
    
    def test_train_and_store(self, trained_store: FAISSIVFBaseline, rng: np.random.Generator) -> None:
        """Trained store should accept vectors."""
        assert trained_store.is_trained
        
        trained_store.store("test", rng.standard_normal(64))
        assert trained_store.count == 1
    
    def test_approximate_recall(self, trained_store: FAISSIVFBaseline, rng: np.random.Generator) -> None:
        """IVF should have reasonable recall (not perfect)."""
        # Store vectors
        vectors = []
        for i in range(100):
            vec = rng.standard_normal(64)
            trained_store.store(f"vec_{i}", vec)
            vectors.append(vec)
        
        # Query with exact vectors — should mostly find them
        hits = 0
        for i, vec in enumerate(vectors[:20]):
            results = trained_store.query(vec, k=5)
            if any(r[0] == f"vec_{i}" for r in results):
                hits += 1
        
        # With nprobe=5 on 10 clusters, should get decent recall
        assert hits >= 15, f"Expected at least 15/20 hits, got {hits}"
    
    def test_clear_keeps_training(self, trained_store: FAISSIVFBaseline, rng: np.random.Generator) -> None:
        """Clear should remove data but keep training."""
        trained_store.store("test", rng.standard_normal(64))
        assert trained_store.count == 1
        
        trained_store.clear()
        
        assert trained_store.count == 0
        assert trained_store.is_trained  # Still trained
        
        # Can add more without re-training
        trained_store.store("test2", rng.standard_normal(64))
        assert trained_store.count == 1
