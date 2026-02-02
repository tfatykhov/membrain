"""
Tests for FlyHash encoder.

Validates:
- Output dimensions and sparsity
- Locality-sensitive hashing properties (similar vectors → similar hashes)
- Reproducibility with seed
- Memory efficiency (int8 projection)
- Edge cases and error handling
"""

import numpy as np
import pytest

from membrain.encoder import FlyHash


class TestFlyHashInit:
    """Tests for FlyHash initialization."""

    def test_default_initialization(self) -> None:
        """FlyHash should initialize with sensible defaults."""
        encoder = FlyHash()
        assert encoder.input_dim == 1536
        assert encoder.output_dim == int(1536 * 13.0)
        assert encoder.active_bits == 50

    def test_custom_parameters(self) -> None:
        """FlyHash should accept custom parameters."""
        encoder = FlyHash(
            input_dim=512,
            expansion_ratio=10.0,
            active_bits=25,
            seed=42,
        )
        assert encoder.input_dim == 512
        assert encoder.output_dim == 5120
        assert encoder.active_bits == 25

    def test_invalid_active_bits_raises(self) -> None:
        """Should raise if active_bits >= output_dim."""
        with pytest.raises(ValueError, match="active_bits"):
            FlyHash(input_dim=100, expansion_ratio=1.0, active_bits=100)

    def test_projection_matrix_shape(self) -> None:
        """Projection matrix should have correct shape (output_dim, input_dim)."""
        encoder = FlyHash(input_dim=256, expansion_ratio=10.0, seed=42)
        assert encoder._projection.shape == (2560, 256)

    def test_projection_dtype_is_int8(self) -> None:
        """Projection matrix should use int8 for memory efficiency."""
        encoder = FlyHash(input_dim=256, seed=42)
        assert encoder._projection.dtype == np.int8

    def test_projection_values_are_binary(self) -> None:
        """Projection matrix should contain only {-1, +1}."""
        encoder = FlyHash(input_dim=256, seed=42)
        unique = np.unique(encoder._projection)
        assert set(unique) == {-1, 1}

    def test_memory_efficient(self) -> None:
        """Default config should use < 50 MB for projection."""
        encoder = FlyHash(input_dim=1536, expansion_ratio=13.0, seed=42)
        # 19968 * 1536 * 1 byte = ~30.7 MB
        assert encoder._projection.nbytes < 50_000_000
        assert encoder.memory_bytes < 50_000_000

    def test_memory_bytes_property(self) -> None:
        """memory_bytes property should return projection size."""
        encoder = FlyHash(input_dim=64, expansion_ratio=4.0, seed=42)
        expected = 64 * 4 * 64  # output_dim * input_dim * 1 byte
        assert encoder.memory_bytes == expected


class TestFlyHashEncode:
    """Tests for FlyHash encoding."""

    @pytest.fixture
    def encoder(self) -> FlyHash:
        return FlyHash(input_dim=1536, seed=42)

    def test_output_dimension(self, encoder: FlyHash) -> None:
        """Output should have correct expanded dimension."""
        vector = np.random.default_rng(0).standard_normal(1536).astype(np.float32)
        output = encoder.encode(vector)
        assert output.shape == (encoder.output_dim,)

    def test_output_is_binary(self, encoder: FlyHash) -> None:
        """Output should contain only 0s and 1s."""
        vector = np.random.default_rng(0).standard_normal(1536).astype(np.float32)
        output = encoder.encode(vector)
        assert set(np.unique(output)).issubset({0.0, 1.0})

    def test_exact_k_active_bits(self, encoder: FlyHash) -> None:
        """Exactly k bits should be active."""
        vector = np.random.default_rng(0).standard_normal(1536).astype(np.float32)
        output = encoder.encode(vector)
        assert np.sum(output) == encoder.active_bits

    def test_output_sparsity(self, encoder: FlyHash) -> None:
        """Output should be >90% sparse."""
        vector = np.random.default_rng(0).standard_normal(1536).astype(np.float32)
        output = encoder.encode(vector)
        sparsity = 1.0 - (np.sum(output) / len(output))
        assert sparsity > 0.90

    def test_invalid_input_dimension_raises(self, encoder: FlyHash) -> None:
        """Should raise error for wrong input dimension."""
        vector = np.random.default_rng(0).standard_normal(512).astype(np.float32)
        with pytest.raises(ValueError, match="Expected vector of shape"):
            encoder.encode(vector)

    def test_invalid_input_ndim_raises(self, encoder: FlyHash) -> None:
        """Should raise error for multi-dimensional input."""
        vector = np.random.default_rng(0).standard_normal((10, 1536)).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 1-D vector"):
            encoder.encode(vector)


class TestFlyHashReproducibility:
    """Tests for reproducibility."""

    def test_same_seed_same_matrix(self) -> None:
        """Same seed should produce identical projection matrices."""
        enc1 = FlyHash(seed=42)
        enc2 = FlyHash(seed=42)
        assert np.array_equal(enc1._projection, enc2._projection)

    def test_same_seed_same_output(self) -> None:
        """Same seed and input should produce identical output."""
        enc1 = FlyHash(seed=42)
        enc2 = FlyHash(seed=42)

        vector = np.random.default_rng(100).standard_normal(1536).astype(np.float32)
        h1 = enc1.encode(vector)
        h2 = enc2.encode(vector)

        assert np.array_equal(h1, h2)

    def test_different_seed_different_matrix(self) -> None:
        """Different seeds should produce different matrices."""
        enc1 = FlyHash(seed=42)
        enc2 = FlyHash(seed=123)
        assert not np.array_equal(enc1._projection, enc2._projection)


class TestFlyHashLocalitySensitivity:
    """Tests for locality-sensitive hashing properties."""

    @pytest.fixture
    def encoder(self) -> FlyHash:
        return FlyHash(input_dim=1536, seed=42)

    def test_similar_vectors_similar_hashes(self, encoder: FlyHash) -> None:
        """Similar vectors should produce similar hash codes."""
        rng = np.random.default_rng(0)
        v1 = rng.standard_normal(1536).astype(np.float32)
        # Small perturbation (10% noise)
        v2 = v1 + 0.1 * rng.standard_normal(1536).astype(np.float32)

        h1 = encoder.encode(v1)
        h2 = encoder.encode(v2)

        similarity = encoder.jaccard_similarity(h1, h2)
        # Expect significant overlap for similar vectors
        assert similarity > 0.2, f"Expected similarity > 0.2, got {similarity}"

    def test_dissimilar_vectors_different_hashes(self, encoder: FlyHash) -> None:
        """Dissimilar vectors should produce different hash codes."""
        rng = np.random.default_rng(0)
        v1 = rng.standard_normal(1536).astype(np.float32)
        # Completely unrelated vector
        v2 = np.random.default_rng(999).standard_normal(1536).astype(np.float32)

        h1 = encoder.encode(v1)
        h2 = encoder.encode(v2)

        similarity = encoder.jaccard_similarity(h1, h2)
        # Expect low overlap for dissimilar vectors
        assert similarity < 0.2, f"Expected similarity < 0.2, got {similarity}"

    def test_semantic_preservation(self, encoder: FlyHash) -> None:
        """
        FlyHash should preserve semantic relationships.

        Test case from PRD:
        - "King" and "Queen" embeddings should have lower Hamming distance
        - "King" and "Apple" embeddings should have higher Hamming distance
        """
        rng = np.random.default_rng(100)

        # Simulate embeddings (in real use, get from OpenAI)
        king = rng.standard_normal(1536).astype(np.float32)
        # Queen is similar to King (small perturbation)
        queen = king + 0.2 * rng.standard_normal(1536).astype(np.float32)
        # Apple is unrelated
        apple = np.random.default_rng(200).standard_normal(1536).astype(np.float32)

        h_king = encoder.encode(king)
        h_queen = encoder.encode(queen)
        h_apple = encoder.encode(apple)

        dist_king_queen = encoder.hamming_distance(h_king, h_queen)
        dist_king_apple = encoder.hamming_distance(h_king, h_apple)

        # King-Queen should be closer than King-Apple
        assert dist_king_queen < dist_king_apple, (
            f"King-Queen distance ({dist_king_queen}) should be less than "
            f"King-Apple distance ({dist_king_apple})"
        )


class TestFlyHashBatch:
    """Tests for batch encoding."""

    @pytest.fixture
    def encoder(self) -> FlyHash:
        return FlyHash(input_dim=1536, seed=42)

    def test_batch_output_shape(self, encoder: FlyHash) -> None:
        """Batch encoding should produce correct output shape."""
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 1536)).astype(np.float32)
        outputs = encoder.encode_batch(vectors)
        assert outputs.shape == (10, encoder.output_dim)

    def test_batch_each_has_k_active(self, encoder: FlyHash) -> None:
        """Each vector in batch should have exactly k active bits."""
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 1536)).astype(np.float32)
        outputs = encoder.encode_batch(vectors)

        for i, output in enumerate(outputs):
            assert np.sum(output) == encoder.active_bits, f"Vector {i} has wrong count"

    def test_batch_matches_individual(self, encoder: FlyHash) -> None:
        """Batch encoding should match individual encoding."""
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((5, 1536)).astype(np.float32)

        batch_outputs = encoder.encode_batch(vectors)

        for i, vector in enumerate(vectors):
            individual_output = encoder.encode(vector)
            assert np.array_equal(batch_outputs[i], individual_output)

    def test_batch_invalid_shape_raises(self, encoder: FlyHash) -> None:
        """Should raise for invalid batch shape."""
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 512)).astype(np.float32)
        with pytest.raises(ValueError, match="Expected vectors of shape"):
            encoder.encode_batch(vectors)


class TestFlyHashMetrics:
    """Tests for similarity metrics."""

    @pytest.fixture
    def encoder(self) -> FlyHash:
        return FlyHash(input_dim=1536, seed=42)

    def test_hamming_distance_identical(self, encoder: FlyHash) -> None:
        """Hamming distance of identical hashes should be 0."""
        rng = np.random.default_rng(0)
        vector = rng.standard_normal(1536).astype(np.float32)
        h = encoder.encode(vector)
        assert encoder.hamming_distance(h, h) == 0

    def test_jaccard_similarity_identical(self, encoder: FlyHash) -> None:
        """Jaccard similarity of identical hashes should be 1.0."""
        rng = np.random.default_rng(0)
        vector = rng.standard_normal(1536).astype(np.float32)
        h = encoder.encode(vector)
        assert encoder.jaccard_similarity(h, h) == 1.0

    def test_sparsity_calculation(self, encoder: FlyHash) -> None:
        """Sparsity should be correctly calculated."""
        expected_sparsity = 1.0 - (50 / encoder.output_dim)
        assert abs(encoder.get_sparsity() - expected_sparsity) < 1e-6

    def test_repr(self, encoder: FlyHash) -> None:
        """__repr__ should return informative string."""
        r = repr(encoder)
        assert "FlyHash" in r
        assert "input_dim=1536" in r
        assert "active_bits=50" in r
        assert "memory=" in r  # New: includes memory usage
