"""
Membrain gRPC Server

Serves the MemoryUnit A2A interface for LLM agents.
Provides Remember, Recall, Consolidate, and Ping operations.

# TODO: Implement TLS for production deployments
# - Add ssl_server_credentials with certificate/key
# - Replace add_insecure_port with add_secure_port
# - See: https://grpc.io/docs/guides/auth/
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import threading
from concurrent import futures
from typing import Any, Callable

import grpc
import numpy as np

from membrain.core import BiCameralMemory
from membrain.encoder import FlyHash
from membrain.proto import memory_a2a_pb2, memory_a2a_pb2_grpc

# Configuration from environment
DEFAULT_PORT = 50051
DEFAULT_MAX_WORKERS = 10
DEFAULT_INPUT_DIM = 1536  # OpenAI Ada-002 embedding dimension
DEFAULT_EXPANSION_RATIO = 13.0  # FlyHash expansion (output = input * ratio)
DEFAULT_N_NEURONS = 1000

# Validation constants
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0
MIN_IMPORTANCE = 0.0
MAX_IMPORTANCE = 1.0
CONTEXT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-:.]{1,256}$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TokenAuthInterceptor(grpc.ServerInterceptor):
    """
    gRPC interceptor for bearer token authentication.
    
    Validates the 'authorization' metadata header against configured token.
    """

    def __init__(self, token: str) -> None:
        """
        Initialize the auth interceptor.

        Args:
            token: The expected bearer token for authentication.
        """
        self._token = token
        self._auth_header = f"bearer {token}"

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Any:
        """Intercept and validate incoming requests."""
        # Extract authorization header
        metadata = dict(handler_call_details.invocation_metadata)
        auth_value = metadata.get("authorization", "")

        if auth_value.lower() != self._auth_header:
            return grpc.unary_unary_rpc_method_handler(
                lambda req, ctx: self._deny(ctx)
            )

        return continuation(handler_call_details)

    def _deny(self, context: grpc.ServicerContext) -> None:
        """Deny the request with UNAUTHENTICATED status."""
        context.set_code(grpc.StatusCode.UNAUTHENTICATED)
        context.set_details("Invalid or missing authentication token")
        return None


class MemoryUnitServicer(memory_a2a_pb2_grpc.MemoryUnitServicer):
    """
    gRPC servicer implementing the MemoryUnit A2A interface.

    Bridges LLM agents to the neuromorphic memory system via:
    - FlyHash encoder for sparse distributed representations
    - BiCameralMemory for spiking neural network storage
    
    Thread-safe: uses locks for memory operations.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        expansion_ratio: float = DEFAULT_EXPANSION_RATIO,
        n_neurons: int = DEFAULT_N_NEURONS,
    ) -> None:
        """
        Initialize the memory servicer.

        Args:
            input_dim: Dimension of input embeddings (e.g., 1536 for Ada-002).
            expansion_ratio: FlyHash expansion ratio (output = input * ratio).
            n_neurons: Number of neurons in BiCameralMemory.
        """
        self.input_dim = input_dim
        self.expansion_ratio = expansion_ratio

        # Thread safety lock for memory operations
        self._lock = threading.RLock()

        # Initialize FlyHash encoder
        self.encoder = FlyHash(
            input_dim=input_dim,
            expansion_ratio=expansion_ratio,
        )
        self.output_dim = self.encoder.output_dim

        # Initialize BiCameralMemory
        self.memory = BiCameralMemory(
            n_neurons=n_neurons,
            dimensions=self.output_dim,
        )

        # Build the simulator
        self.memory._ensure_simulator()

        logger.info(
            f"MemoryUnitServicer initialized: "
            f"input_dim={input_dim}, output_dim={self.output_dim}, n_neurons={n_neurons}"
        )

    def _validate_context_id(self, context_id: str) -> str | None:
        """
        Validate and sanitize context_id.
        
        Returns error message if invalid, None if valid.
        """
        if not context_id:
            return "context_id is required"
        if not CONTEXT_ID_PATTERN.match(context_id):
            return (
                "context_id must be 1-256 characters, "
                "alphanumeric with _-:. allowed"
            )
        return None

    def _validate_threshold(self, threshold: float) -> str | None:
        """
        Validate threshold value.
        
        Returns error message if invalid, None if valid.
        """
        if threshold < MIN_THRESHOLD or threshold > MAX_THRESHOLD:
            return f"threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}"
        return None

    def Ping(
        self,
        request: memory_a2a_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> memory_a2a_pb2.Ack:
        """Health check endpoint."""
        return memory_a2a_pb2.Ack(success=True, message="pong")

    def Remember(
        self,
        request: memory_a2a_pb2.MemoryPacket,
        context: grpc.ServicerContext,
    ) -> memory_a2a_pb2.Ack:
        """
        Store a memory with learning enabled.

        Takes a dense embedding vector, encodes it via FlyHash,
        and stores it in the spiking neural network.
        """
        try:
            # Validate context_id
            context_id_error = self._validate_context_id(request.context_id)
            if context_id_error:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(context_id_error)
                return memory_a2a_pb2.Ack(success=False, message=context_id_error)

            # Validate input vector
            vector = np.array(request.vector, dtype=np.float32)
            if vector.shape != (self.input_dim,):
                msg = f"Vector must be {self.input_dim} dimensions, got {vector.shape[0]}"
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return memory_a2a_pb2.Ack(success=False, message=msg)

            # Validate importance - reject invalid values instead of silent override
            importance = request.importance
            if importance == 0.0:
                # Default to 1.0 if not specified (protobuf default is 0)
                importance = 1.0
            elif not MIN_IMPORTANCE <= importance <= MAX_IMPORTANCE:
                msg = f"importance must be between {MIN_IMPORTANCE} and {MAX_IMPORTANCE}"
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return memory_a2a_pb2.Ack(success=False, message=msg)

            # Encode via FlyHash
            sparse_vector = self.encoder.encode(vector)

            # Store in memory (thread-safe)
            with self._lock:
                self.memory.remember(
                    context_id=request.context_id,
                    sparse_vector=sparse_vector,
                    importance=importance,
                )

            logger.info(f"Stored memory: {request.context_id}")
            return memory_a2a_pb2.Ack(
                success=True,
                message=f"Memory stored: {request.context_id}",
            )

        except Exception as e:
            logger.exception(f"Remember failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return memory_a2a_pb2.Ack(success=False, message="Internal server error")

    def Recall(
        self,
        request: memory_a2a_pb2.QueryPacket,
        context: grpc.ServicerContext,
    ) -> memory_a2a_pb2.ContextResponse:
        """
        Query for associated memories via pattern completion.

        Takes a query embedding, encodes it via FlyHash,
        and retrieves matching memories from the SNN.
        """
        try:
            # Validate input vector
            vector = np.array(request.vector, dtype=np.float32)
            if vector.shape != (self.input_dim,):
                msg = f"Vector must be {self.input_dim} dimensions, got {vector.shape[0]}"
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return memory_a2a_pb2.ContextResponse()

            # Validate and apply threshold
            threshold = request.threshold if request.threshold > 0 else 0.7
            threshold_error = self._validate_threshold(threshold)
            if threshold_error:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(threshold_error)
                return memory_a2a_pb2.ContextResponse()

            max_results = request.max_results if request.max_results > 0 else 5

            # Encode via FlyHash
            sparse_vector = self.encoder.encode(vector)

            # Recall from memory (thread-safe)
            with self._lock:
                results = self.memory.recall(
                    query_vector=sparse_vector,
                    threshold=threshold,
                    max_results=max_results,
                )

            if not results:
                return memory_a2a_pb2.ContextResponse(
                    context_ids=[],
                    confidences=[],
                    overall_confidence=0.0,
                )

            context_ids = [r.context_id for r in results]
            confidences = [r.confidence for r in results]
            overall_confidence = float(np.mean(confidences))

            logger.info(f"Recalled {len(results)} memories")
            return memory_a2a_pb2.ContextResponse(
                context_ids=context_ids,
                confidences=confidences,
                overall_confidence=overall_confidence,
            )

        except Exception as e:
            logger.exception(f"Recall failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return memory_a2a_pb2.ContextResponse()

    def Consolidate(
        self,
        request: memory_a2a_pb2.SleepSignal,
        context: grpc.ServicerContext,
    ) -> memory_a2a_pb2.Ack:
        """
        Trigger sleep phase for memory consolidation.

        Runs the network without input to allow attractor states to settle.
        Optionally prunes weak associations.
        """
        try:
            duration_ms = request.duration_ms if request.duration_ms > 0 else 1000

            # Consolidate with lock (thread-safe)
            with self._lock:
                pruned = self.memory.consolidate(
                    duration_ms=duration_ms,
                    prune_weak=request.prune_weak,
                )

            message = f"Consolidation complete ({duration_ms}ms)"
            if request.prune_weak:
                message += f", pruned {pruned} weak memories"

            logger.info(message)
            return memory_a2a_pb2.Ack(success=True, message=message)

        except Exception as e:
            logger.exception(f"Consolidate failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return memory_a2a_pb2.Ack(success=False, message="Internal server error")

    def shutdown(self) -> None:
        """Clean up resources."""
        with self._lock:
            if hasattr(self, "memory"):
                self.memory.reset()
        logger.info("MemoryUnitServicer shutdown complete")


class MembrainServer:
    """
    gRPC server wrapper with lifecycle management.

    Handles graceful shutdown on SIGTERM/SIGINT.
    Supports optional token authentication.
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        max_workers: int = DEFAULT_MAX_WORKERS,
        input_dim: int = DEFAULT_INPUT_DIM,
        expansion_ratio: float = DEFAULT_EXPANSION_RATIO,
        n_neurons: int = DEFAULT_N_NEURONS,
        auth_token: str | None = None,
    ) -> None:
        """
        Initialize the server.

        Args:
            port: Port to listen on.
            max_workers: Maximum thread pool workers.
            input_dim: Dimension of input embeddings.
            expansion_ratio: FlyHash expansion ratio.
            n_neurons: Number of neurons in BiCameralMemory.
            auth_token: Optional bearer token for authentication.
        """
        self.port = port
        self.max_workers = max_workers
        self.auth_token = auth_token

        self.servicer = MemoryUnitServicer(
            input_dim=input_dim,
            expansion_ratio=expansion_ratio,
            n_neurons=n_neurons,
        )

        self.server: grpc.Server | None = None
        self._shutdown_requested = False

    def start(self) -> None:
        """Start the gRPC server."""
        # Build interceptors list
        interceptors = []
        if self.auth_token:
            interceptors.append(TokenAuthInterceptor(self.auth_token))
            logger.info("Token authentication enabled")

        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            interceptors=interceptors,
        )
        memory_a2a_pb2_grpc.add_MemoryUnitServicer_to_server(
            self.servicer, self.server
        )
        # TODO: Replace with add_secure_port for TLS in production
        self.server.add_insecure_port(f"[::]:{self.port}")
        self.server.start()
        logger.info(f"Membrain server started on port {self.port}")

    def wait_for_termination(self) -> None:
        """Block until server terminates."""
        if self.server:
            self.server.wait_for_termination()

    def stop(self, grace: float = 5.0) -> None:
        """
        Gracefully stop the server.

        Args:
            grace: Grace period in seconds for in-flight requests.
        """
        if self._shutdown_requested:
            return

        self._shutdown_requested = True
        logger.info("Initiating graceful shutdown...")

        if self.server:
            self.server.stop(grace)
        self.servicer.shutdown()
        logger.info("Membrain server stopped")


def serve(
    port: int | None = None,
    max_workers: int | None = None,
    input_dim: int | None = None,
    expansion_ratio: float | None = None,
    n_neurons: int | None = None,
    auth_token: str | None = None,
) -> None:
    """
    Start the gRPC server with configuration from environment.

    Environment variables:
        MEMBRAIN_PORT: Server port (default: 50051)
        MEMBRAIN_MAX_WORKERS: Thread pool size (default: 10)
        MEMBRAIN_INPUT_DIM: Input embedding dimension (default: 1536)
        MEMBRAIN_EXPANSION_RATIO: FlyHash expansion ratio (default: 13.0)
        MEMBRAIN_N_NEURONS: Number of SNN neurons (default: 1000)
        MEMBRAIN_AUTH_TOKEN: Bearer token for authentication (optional)
    """
    # Read configuration from environment with defaults
    port = port or int(os.environ.get("MEMBRAIN_PORT", DEFAULT_PORT))
    max_workers = max_workers or int(
        os.environ.get("MEMBRAIN_MAX_WORKERS", DEFAULT_MAX_WORKERS)
    )
    input_dim = input_dim or int(
        os.environ.get("MEMBRAIN_INPUT_DIM", DEFAULT_INPUT_DIM)
    )
    expansion_ratio = expansion_ratio or float(
        os.environ.get("MEMBRAIN_EXPANSION_RATIO", DEFAULT_EXPANSION_RATIO)
    )
    n_neurons = n_neurons or int(
        os.environ.get("MEMBRAIN_N_NEURONS", DEFAULT_N_NEURONS)
    )
    auth_token = auth_token or os.environ.get("MEMBRAIN_AUTH_TOKEN")

    if not auth_token:
        logger.warning(
            "No MEMBRAIN_AUTH_TOKEN set - server running without authentication!"
        )

    server = MembrainServer(
        port=port,
        max_workers=max_workers,
        input_dim=input_dim,
        expansion_ratio=expansion_ratio,
        n_neurons=n_neurons,
        auth_token=auth_token,
    )

    # Register signal handlers for graceful shutdown
    def handle_signal(signum: int, frame: object) -> None:
        logger.info(f"Received signal {signum}")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    server.start()
    server.wait_for_termination()


def main() -> None:
    """Entry point for the server module."""
    serve()


if __name__ == "__main__":
    main()
