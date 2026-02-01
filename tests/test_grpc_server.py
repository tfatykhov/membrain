"""
Tests for the Membrain gRPC server.

Tests the MemoryUnitServicer and server lifecycle.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip all tests if grpc not available
grpc = pytest.importorskip("grpc")

from membrain.proto import memory_a2a_pb2
from membrain.server import MemoryUnitServicer, MembrainServer


class TestPing:
    """Test the Ping RPC."""

    @pytest.fixture
    def servicer(self) -> MemoryUnitServicer:
        """Create a servicer with small dimensions for testing."""
        return MemoryUnitServicer(
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
        )

    def test_ping_returns_success(self, servicer: MemoryUnitServicer) -> None:
        """Ping should return success=True."""
        request = memory_a2a_pb2.Empty()
        context = MagicMock()

        response = servicer.Ping(request, context)

        assert response.success is True
        assert response.message == "pong"

    def test_ping_does_not_set_error(self, servicer: MemoryUnitServicer) -> None:
        """Ping should not set error codes."""
        request = memory_a2a_pb2.Empty()
        context = MagicMock()

        servicer.Ping(request, context)

        context.set_code.assert_not_called()


class TestRemember:
    """Test the Remember RPC."""

    @pytest.fixture
    def servicer(self) -> MemoryUnitServicer:
        """Create a servicer with small dimensions for testing."""
        return MemoryUnitServicer(
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
        )

    def test_remember_stores_memory(self, servicer: MemoryUnitServicer) -> None:
        """Remember should store a memory successfully."""
        vector = np.random.randn(64).astype(np.float32).tolist()
        request = memory_a2a_pb2.MemoryPacket(
            context_id="test-001",
            vector=vector,
            importance=0.8,
        )
        context = MagicMock()

        response = servicer.Remember(request, context)

        assert response.success is True
        assert "test-001" in response.message
        assert servicer.memory.get_memory_count() == 1

    def test_remember_invalid_dimensions(self, servicer: MemoryUnitServicer) -> None:
        """Remember should reject wrong vector dimensions."""
        vector = np.random.randn(32).astype(np.float32).tolist()  # Wrong size
        request = memory_a2a_pb2.MemoryPacket(
            context_id="test-bad",
            vector=vector,
            importance=0.5,
        )
        context = MagicMock()

        response = servicer.Remember(request, context)

        assert response.success is False
        assert "dimensions" in response.message.lower()
        context.set_code.assert_called_once()

    def test_remember_invalid_importance(self, servicer: MemoryUnitServicer) -> None:
        """Remember should reject importance outside 0-1 range."""
        vector = np.random.randn(64).astype(np.float32).tolist()
        request = memory_a2a_pb2.MemoryPacket(
            context_id="test-bad",
            vector=vector,
            importance=1.5,  # Invalid
        )
        context = MagicMock()

        response = servicer.Remember(request, context)

        assert response.success is False
        assert "importance" in response.message.lower()

    def test_remember_multiple_memories(self, servicer: MemoryUnitServicer) -> None:
        """Remember should store multiple distinct memories."""
        context = MagicMock()

        for i in range(3):
            vector = np.random.randn(64).astype(np.float32).tolist()
            request = memory_a2a_pb2.MemoryPacket(
                context_id=f"test-{i:03d}",
                vector=vector,
                importance=0.8,
            )
            response = servicer.Remember(request, context)
            assert response.success is True

        assert servicer.memory.get_memory_count() == 3


class TestRecall:
    """Test the Recall RPC."""

    @pytest.fixture
    def servicer_with_memories(self) -> MemoryUnitServicer:
        """Create a servicer with pre-stored memories."""
        servicer = MemoryUnitServicer(
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
        )
        context = MagicMock()

        # Store a memory
        vector = np.random.randn(64).astype(np.float32)
        request = memory_a2a_pb2.MemoryPacket(
            context_id="stored-001",
            vector=vector.tolist(),
            importance=1.0,
        )
        servicer.Remember(request, context)

        # Store the vector for querying
        servicer._test_vector = vector

        return servicer

    def test_recall_returns_response(
        self, servicer_with_memories: MemoryUnitServicer
    ) -> None:
        """Recall should return a ContextResponse."""
        vector = np.random.randn(64).astype(np.float32).tolist()
        request = memory_a2a_pb2.QueryPacket(
            vector=vector,
            threshold=0.1,  # Low threshold to get results
            max_results=5,
        )
        context = MagicMock()

        response = servicer_with_memories.Recall(request, context)

        assert isinstance(response, memory_a2a_pb2.ContextResponse)

    def test_recall_invalid_dimensions(
        self, servicer_with_memories: MemoryUnitServicer
    ) -> None:
        """Recall should reject wrong vector dimensions."""
        vector = np.random.randn(32).astype(np.float32).tolist()
        request = memory_a2a_pb2.QueryPacket(
            vector=vector,
            threshold=0.5,
            max_results=5,
        )
        context = MagicMock()

        response = servicer_with_memories.Recall(request, context)

        context.set_code.assert_called_once()

    def test_recall_empty_when_no_match(
        self, servicer_with_memories: MemoryUnitServicer
    ) -> None:
        """Recall with high threshold on random vector should return empty."""
        vector = np.random.randn(64).astype(np.float32).tolist()
        request = memory_a2a_pb2.QueryPacket(
            vector=vector,
            threshold=0.99,  # Very high threshold
            max_results=5,
        )
        context = MagicMock()

        response = servicer_with_memories.Recall(request, context)

        assert len(response.context_ids) == 0


class TestConsolidate:
    """Test the Consolidate RPC."""

    @pytest.fixture
    def servicer(self) -> MemoryUnitServicer:
        """Create a servicer with small dimensions for testing."""
        return MemoryUnitServicer(
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
        )

    def test_consolidate_runs(self, servicer: MemoryUnitServicer) -> None:
        """Consolidate should complete successfully."""
        request = memory_a2a_pb2.SleepSignal(
            duration_ms=10,
            prune_weak=False,
        )
        context = MagicMock()

        response = servicer.Consolidate(request, context)

        assert response.success is True
        assert "10ms" in response.message

    def test_consolidate_with_pruning(self, servicer: MemoryUnitServicer) -> None:
        """Consolidate with pruning should report pruned count."""
        # Store a weak memory
        vector = np.random.randn(64).astype(np.float32).tolist()
        remember_request = memory_a2a_pb2.MemoryPacket(
            context_id="weak-001",
            vector=vector,
            importance=0.05,  # Very low importance
        )
        servicer.Remember(remember_request, MagicMock())

        request = memory_a2a_pb2.SleepSignal(
            duration_ms=10,
            prune_weak=True,
        )
        context = MagicMock()

        response = servicer.Consolidate(request, context)

        assert response.success is True
        assert "pruned" in response.message.lower()


class TestServerLifecycle:
    """Test server start/stop lifecycle."""

    def test_server_creates_servicer(self) -> None:
        """Server should create a MemoryUnitServicer."""
        server = MembrainServer(
            port=50099,
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
        )

        assert server.servicer is not None
        assert isinstance(server.servicer, MemoryUnitServicer)

    def test_server_start_stop(self) -> None:
        """Server should start and stop cleanly."""
        server = MembrainServer(
            port=50098,
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
        )

        server.start()
        assert server.server is not None

        server.stop(grace=0.1)
        assert server._shutdown_requested is True


class TestIntegration:
    """Integration tests for remember + recall flow."""

    @pytest.fixture
    def servicer(self) -> MemoryUnitServicer:
        """Create a servicer with small dimensions for testing."""
        return MemoryUnitServicer(
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
        )

    def test_remember_then_recall_same_vector(
        self, servicer: MemoryUnitServicer
    ) -> None:
        """Storing and recalling the same vector should work."""
        context = MagicMock()

        # Store a memory
        vector = np.random.randn(64).astype(np.float32)
        remember_request = memory_a2a_pb2.MemoryPacket(
            context_id="integration-001",
            vector=vector.tolist(),
            importance=1.0,
        )
        remember_response = servicer.Remember(remember_request, context)
        assert remember_response.success is True

        # Recall with the same vector
        recall_request = memory_a2a_pb2.QueryPacket(
            vector=vector.tolist(),
            threshold=0.1,  # Low threshold to ensure match
            max_results=5,
        )
        recall_response = servicer.Recall(recall_request, context)

        # Should get at least some results (threshold-dependent)
        assert isinstance(recall_response, memory_a2a_pb2.ContextResponse)
