"""Tests for dataset generation and noise functions."""

import numpy as np
import pytest

from bench.datasets import SyntheticDataset, add_noise, batch_add_noise


class TestSyntheticDataset:
    """Tests for SyntheticDataset generation."""
    
    def test_gaussian_shape(self) -> None:
        """Generated dataset should have correct shape."""
        ds = SyntheticDataset.gaussian(n=100, dim=64, seed=42)
        
        assert len(ds) == 100
        assert ds.vectors.shape == (100, 64)
        assert len(ds.keys) == 100
        assert ds.dim == 64
    
    def test_gaussian_normalized(self) -> None:
        """All vectors should be unit length."""
        ds = SyntheticDataset.gaussian(n=100, dim=64, seed=42)
        
        norms = np.linalg.norm(ds.vectors, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(100))
    
    def test_gaussian_reproducible(self) -> None:
        """Same seed should produce identical datasets."""
        ds1 = SyntheticDataset.gaussian(n=100, dim=64, seed=42)
        ds2 = SyntheticDataset.gaussian(n=100, dim=64, seed=42)
        
        np.testing.assert_array_equal(ds1.vectors, ds2.vectors)
        assert ds1.keys == ds2.keys
    
    def test_gaussian_different_seeds(self) -> None:
        """Different seeds should produce different datasets."""
        ds1 = SyntheticDataset.gaussian(n=100, dim=64, seed=42)
        ds2 = SyntheticDataset.gaussian(n=100, dim=64, seed=123)
        
        assert not np.allclose(ds1.vectors, ds2.vectors)
    
    def test_key_format(self) -> None:
        """Keys should be zero-padded."""
        ds = SyntheticDataset.gaussian(n=10, dim=64, seed=42)
        
        assert ds.keys[0] == "vec_00000"
        assert ds.keys[9] == "vec_00009"
    
    def test_iteration(self) -> None:
        """Should be iterable as (key, vector) pairs."""
        ds = SyntheticDataset.gaussian(n=10, dim=64, seed=42)
        
        items = list(ds)
        assert len(items) == 10
        assert items[0][0] == "vec_00000"
        assert items[0][1].shape == (64,)
    
    def test_subset(self) -> None:
        """Subset should return first n items."""
        ds = SyntheticDataset.gaussian(n=100, dim=64, seed=42)
        subset = ds.subset(10)
        
        assert len(subset) == 10
        np.testing.assert_array_equal(subset.vectors, ds.vectors[:10])
        assert subset.keys == ds.keys[:10]
    
    def test_clustered_shape(self) -> None:
        """Clustered dataset should have correct shape."""
        ds = SyntheticDataset.clustered(n=100, dim=64, n_clusters=5, seed=42)
        
        assert len(ds) == 100
        assert ds.dim == 64
    
    def test_clustered_normalized(self) -> None:
        """Clustered vectors should be unit length."""
        ds = SyntheticDataset.clustered(n=100, dim=64, n_clusters=5, seed=42)
        
        norms = np.linalg.norm(ds.vectors, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(100))


class TestAddNoise:
    """Tests for noise injection functions."""
    
    @pytest.fixture
    def unit_vector(self) -> np.ndarray:
        v = np.array([1.0, 0.0, 0.0])
        return v / np.linalg.norm(v)
    
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)
    
    def test_zero_noise(self, unit_vector: np.ndarray, rng: np.random.Generator) -> None:
        """Zero noise should return identical vector."""
        result = add_noise(unit_vector, level=0.0, rng=rng)
        np.testing.assert_array_equal(result, unit_vector)
    
    def test_output_normalized(self, unit_vector: np.ndarray, rng: np.random.Generator) -> None:
        """Output should always be unit length."""
        result = add_noise(unit_vector, level=0.5, rng=rng)
        np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)
    
    def test_noise_changes_vector(self, unit_vector: np.ndarray, rng: np.random.Generator) -> None:
        """Non-zero noise should change the vector."""
        result = add_noise(unit_vector, level=0.1, rng=rng)
        assert not np.allclose(result, unit_vector)
    
    def test_noise_reproducible(self, unit_vector: np.ndarray) -> None:
        """Same RNG state should produce same noise."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        result1 = add_noise(unit_vector, level=0.1, rng=rng1)
        result2 = add_noise(unit_vector, level=0.1, rng=rng2)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_higher_noise_more_deviation(self, unit_vector: np.ndarray) -> None:
        """Higher noise level should cause more deviation on average."""
        deviations = []
        for level in [0.05, 0.1, 0.2, 0.3]:
            rng = np.random.default_rng(42)
            total_deviation = 0.0
            for _ in range(100):
                result = add_noise(unit_vector, level=level, rng=rng)
                # Cosine similarity (dot product of unit vectors)
                similarity = np.dot(result, unit_vector)
                total_deviation += 1.0 - similarity
            deviations.append(total_deviation / 100)
        
        # Deviations should increase with noise level
        assert deviations == sorted(deviations)


class TestBatchAddNoise:
    """Tests for batch noise injection."""
    
    def test_batch_shape_preserved(self) -> None:
        """Output shape should match input."""
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 64))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        
        result = batch_add_noise(vectors, level=0.1, rng=rng)
        
        assert result.shape == vectors.shape
    
    def test_batch_normalized(self) -> None:
        """All output vectors should be unit length."""
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 64))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        
        result = batch_add_noise(vectors, level=0.1, rng=rng)
        norms = np.linalg.norm(result, axis=1)
        
        np.testing.assert_array_almost_equal(norms, np.ones(100))
    
    def test_batch_zero_noise(self) -> None:
        """Zero noise should return copy of input."""
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((10, 64))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        
        result = batch_add_noise(vectors, level=0.0, rng=rng)
        
        np.testing.assert_array_equal(result, vectors)
