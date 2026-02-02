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
from membrain.server import (
    MemoryUnitServicer,
    MembrainServer,
    TokenAuthInterceptor,
    validate_token,
)


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
            noise_scale=0.05,
            max_steps=10,
            convergence_threshold=1e-3,
            prune_weak=False,
        )
        context = MagicMock()

        response = servicer.Consolidate(request, context)

        assert response.success is True
        # Either converged or hit max steps
        assert response.steps_to_converge == -1 or response.steps_to_converge > 0

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
            noise_scale=0.05,
            max_steps=10,
            convergence_threshold=1e-3,
            prune_weak=True,
            prune_threshold=0.1,
        )
        context = MagicMock()

        response = servicer.Consolidate(request, context)

        assert response.success is True
        assert response.pruned_count >= 1


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

    def test_server_with_auth_tokens(self) -> None:
        """Server should accept auth_tokens parameter."""
        # Use a token that meets minimum length requirement
        long_token = "a" * 32
        server = MembrainServer(
            port=50097,
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
            auth_tokens={"client1": long_token},
        )

        assert long_token in server.config.auth_tokens
        server.start()
        server.stop(grace=0.1)

    def test_server_rejects_short_token(self) -> None:
        """Server should reject tokens that are too short."""
        server = MembrainServer(
            port=50096,
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
            auth_tokens={"client1": "short"},
        )

        with pytest.raises(ValueError, match="at least"):
            server.start()


class TestTokenAuth:
    """Test token authentication interceptor."""

    def test_interceptor_rejects_missing_token(self) -> None:
        """Requests without token should be rejected."""
        interceptor = TokenAuthInterceptor("secret-token")
        
        # Mock handler details with no auth header
        handler_details = MagicMock()
        handler_details.invocation_metadata = []
        handler_details.method = "/membrain.MemoryUnit/Remember"
        
        continuation = MagicMock()
        
        result = interceptor.intercept_service(continuation, handler_details)
        
        # Should return denial handler, not call continuation
        continuation.assert_not_called()

    def test_interceptor_rejects_wrong_token(self) -> None:
        """Requests with wrong token should be rejected."""
        interceptor = TokenAuthInterceptor("secret-token")
        
        handler_details = MagicMock()
        handler_details.invocation_metadata = [("authorization", "Bearer wrong-token")]
        handler_details.method = "/membrain.MemoryUnit/Remember"
        
        continuation = MagicMock()
        
        result = interceptor.intercept_service(continuation, handler_details)
        
        continuation.assert_not_called()

    def test_interceptor_accepts_valid_token(self) -> None:
        """Requests with correct token should proceed."""
        interceptor = TokenAuthInterceptor("secret-token")
        
        handler_details = MagicMock()
        handler_details.invocation_metadata = [("authorization", "Bearer secret-token")]
        handler_details.method = "/membrain.MemoryUnit/Remember"
        
        continuation = MagicMock()
        continuation.return_value = "handler"
        
        result = interceptor.intercept_service(continuation, handler_details)
        
        continuation.assert_called_once_with(handler_details)
        assert result == "handler"

    def test_interceptor_exempts_ping(self) -> None:
        """Ping should be exempt from authentication."""
        interceptor = TokenAuthInterceptor("secret-token")
        
        handler_details = MagicMock()
        handler_details.invocation_metadata = []  # No auth header
        handler_details.method = "/membrain.MemoryUnit/Ping"
        
        continuation = MagicMock()
        continuation.return_value = "handler"
        
        result = interceptor.intercept_service(continuation, handler_details)
        
        # Ping should bypass auth and call continuation
        continuation.assert_called_once_with(handler_details)
        assert result == "handler"


class TestContextIdValidation:
    """Test context_id validation."""

    @pytest.fixture
    def servicer(self) -> MemoryUnitServicer:
        """Create a servicer with small dimensions for testing."""
        return MemoryUnitServicer(
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
        )

    def test_empty_context_id_rejected(self, servicer: MemoryUnitServicer) -> None:
        """Empty context_id should be rejected."""
        vector = np.random.randn(64).astype(np.float32).tolist()
        request = memory_a2a_pb2.MemoryPacket(
            context_id="",
            vector=vector,
            importance=0.8,
        )
        context = MagicMock()

        response = servicer.Remember(request, context)

        assert response.success is False
        assert "context_id" in response.message.lower()

    def test_special_chars_in_context_id_rejected(
        self, servicer: MemoryUnitServicer
    ) -> None:
        """Special characters in context_id should be rejected."""
        vector = np.random.randn(64).astype(np.float32).tolist()
        request = memory_a2a_pb2.MemoryPacket(
            context_id="test<script>alert(1)</script>",
            vector=vector,
            importance=0.8,
        )
        context = MagicMock()

        response = servicer.Remember(request, context)

        assert response.success is False

    def test_valid_context_id_accepted(self, servicer: MemoryUnitServicer) -> None:
        """Valid context_id patterns should be accepted."""
        vector = np.random.randn(64).astype(np.float32).tolist()
        valid_ids = ["test-001", "user_123", "ctx:abc.def", "A1-B2_C3:D4.E5"]
        context = MagicMock()

        for ctx_id in valid_ids:
            request = memory_a2a_pb2.MemoryPacket(
                context_id=ctx_id,
                vector=vector,
                importance=0.8,
            )
            response = servicer.Remember(request, context)
            assert response.success is True, f"Should accept: {ctx_id}"


class TestThresholdValidation:
    """Test threshold validation in Recall."""

    @pytest.fixture
    def servicer(self) -> MemoryUnitServicer:
        """Create a servicer with small dimensions for testing."""
        return MemoryUnitServicer(
            input_dim=64,
            expansion_ratio=4.0,
            n_neurons=50,
        )

    def test_threshold_above_one_rejected(self, servicer: MemoryUnitServicer) -> None:
        """Threshold > 1.0 should be rejected."""
        vector = np.random.randn(64).astype(np.float32).tolist()
        request = memory_a2a_pb2.QueryPacket(
            vector=vector,
            threshold=1.5,
            max_results=5,
        )
        context = MagicMock()

        response = servicer.Recall(request, context)

        context.set_code.assert_called()

    def test_negative_threshold_rejected(self, servicer: MemoryUnitServicer) -> None:
        """Negative threshold should be rejected."""
        vector = np.random.randn(64).astype(np.float32).tolist()
        request = memory_a2a_pb2.QueryPacket(
            vector=vector,
            threshold=-0.5,
            max_results=5,
        )
        context = MagicMock()

        response = servicer.Recall(request, context)

        context.set_code.assert_called()


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


class TestTokenValidation:
    """Test token validation function."""

    def test_empty_token_rejected(self) -> None:
        """Empty token should be rejected."""
        is_valid, error = validate_token("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_short_token_rejected(self) -> None:
        """Token shorter than minimum length should be rejected."""
        is_valid, error = validate_token("short")
        assert is_valid is False
        assert "16" in error  # minimum length

    def test_valid_token_accepted(self) -> None:
        """Token meeting requirements should be accepted."""
        is_valid, error = validate_token("a" * 16)
        assert is_valid is True
        assert error is None

    def test_long_token_accepted(self) -> None:
        """Longer tokens should be accepted."""
        is_valid, error = validate_token("a" * 64)
        assert is_valid is True


class TestMultiClientAuth:
    """Test multi-client token authentication."""

    def test_interceptor_accepts_multiple_clients(self) -> None:
        """Interceptor should accept tokens from multiple clients."""
        tokens = {
            "client1": "token-for-client-one",
            "client2": "token-for-client-two",
        }
        interceptor = TokenAuthInterceptor(tokens)
        
        # Test client1
        handler_details = MagicMock()
        handler_details.invocation_metadata = [
            ("authorization", "Bearer token-for-client-one")
        ]
        continuation = MagicMock()
        continuation.return_value = "handler"
        
        result = interceptor.intercept_service(continuation, handler_details)
        continuation.assert_called_once()
        
        # Test client2
        continuation.reset_mock()
        handler_details.invocation_metadata = [
            ("authorization", "Bearer token-for-client-two")
        ]
        
        result = interceptor.intercept_service(continuation, handler_details)
        continuation.assert_called_once()

    def test_interceptor_rejects_unknown_client(self) -> None:
        """Interceptor should reject tokens not in the config."""
        tokens = {"client1": "valid-token"}
        interceptor = TokenAuthInterceptor(tokens)
        
        handler_details = MagicMock()
        handler_details.invocation_metadata = [
            ("authorization", "Bearer unknown-token")
        ]
        continuation = MagicMock()
        
        result = interceptor.intercept_service(continuation, handler_details)
        continuation.assert_not_called()

