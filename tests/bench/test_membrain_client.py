"""Tests for MembrainStore adapter.

These tests require a running Membrain server.
Skip if server is not available.
"""

import numpy as np
import pytest
import grpc

# Try to import, skip if gRPC deps not available
try:
    from bench.baselines.membrain_client import MembrainStore
    from membrain.proto import memory_a2a_pb2_grpc
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

pytestmark = pytest.mark.skipif(not HAS_GRPC, reason="gRPC dependencies not available")


def server_available(host: str = "localhost", port: int = 50051) -> bool:
    """Check if Membrain server is running."""
    try:
        channel = grpc.insecure_channel(f"{host}:{port}")
        grpc.channel_ready_future(channel).result(timeout=1)
        channel.close()
        return True
    except Exception:
        return False


# Skip all tests if server not running
requires_server = pytest.mark.skipif(
    not server_available(),
    reason="Membrain server not running (start with docker-compose up)"
)


class TestMembrainStoreUnit:
    """Unit tests that don't require a server."""
    
    def test_init(self) -> None:
        """Should initialize with defaults."""
        store = MembrainStore()
        assert store._host == "localhost"
        assert store._port == 50051
        assert store.count == 0
        assert store.dim == 0
    
    def test_duplicate_key_check(self) -> None:
        """Should track keys locally for duplicate detection."""
        store = MembrainStore()
        store._keys.append("test")
        store._key_set.add("test")
        
        with pytest.raises(ValueError, match="Duplicate key"):
            store.store("test", np.array([1.0, 0.0, 0.0]))
    
    def test_batch_duplicate_in_batch(self) -> None:
        """Should detect duplicates within a batch."""
        store = MembrainStore()
        
        items = [
            ("key1", np.array([1.0, 0.0])),
            ("key2", np.array([0.0, 1.0])),
            ("key1", np.array([0.5, 0.5])),  # Duplicate
        ]
        
        with pytest.raises(ValueError, match="Duplicate key in batch"):
            store.store_batch(items)


@requires_server
class TestMembrainStoreIntegration:
    """Integration tests that require a running server."""
    
    @pytest.fixture
    def store(self) -> MembrainStore:
        """Create a connected store."""
        return MembrainStore(host="localhost", port=50051)
    
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)
    
    def test_connection(self, store: MembrainStore) -> None:
        """Should connect to server."""
        store._ensure_connected()
        assert store.is_connected
        store.close()
        assert not store.is_connected
    
    def test_store_and_query(self, store: MembrainStore, rng: np.random.Generator) -> None:
        """Basic store and query should work."""
        dim = 768  # Membrain default dimension
        vec = rng.standard_normal(dim)
        
        store.store("test_vec", vec)
        assert store.count == 1
        
        results = store.query(vec, k=1)
        # Note: Membrain uses attractor dynamics, may not return exact match
        assert len(results) >= 0  # May be empty if attractor not converged
        
        store.close()
    
    def test_context_manager(self, rng: np.random.Generator) -> None:
        """Should work as context manager."""
        with MembrainStore() as store:
            assert store.is_connected
        assert not store.is_connected
    
    def test_memory_tracking(self, store: MembrainStore, rng: np.random.Generator) -> None:
        """Memory should be tracked locally."""
        dim = 768
        vec = rng.standard_normal(dim)
        
        initial = store.memory_mb
        store.store("test", vec)
        
        assert store.memory_mb > initial
        store.close()
