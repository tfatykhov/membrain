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

import hmac
import logging
import re
import signal
import sys
import threading
from concurrent import futures
from typing import Any, Callable

import grpc
import numpy as np

from membrain.config import MembrainConfig
from membrain.core import BiCameralMemory
from membrain.encoder import FlyHash
from membrain.interceptors import LoggingInterceptor
from membrain.logging import get_logger, setup_logging
from membrain.proto import memory_a2a_pb2, memory_a2a_pb2_grpc

# Security constants
MIN_TOKEN_LENGTH = 16  # Minimum token length for security

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
logger = get_logger(__name__)


def validate_token(token: str) -> tuple[bool, str | None]:
    """
    Validate token strength.

    Returns (is_valid, error_message).
    """
    if not token:
        return False, "Token cannot be empty"
    if len(token) < MIN_TOKEN_LENGTH:
        return False, f"Token must be at least {MIN_TOKEN_LENGTH} characters"
    return True, None


class TokenAuthInterceptor(grpc.ServerInterceptor):
    """
    gRPC interceptor for bearer token authentication.

    Validates the 'authorization' metadata header against configured tokens.
    Uses timing-safe comparison to prevent timing attacks.
    Supports multiple client tokens.

    The Ping method is exempt from authentication for health checks.
    """

    # Methods that don't require authentication
    AUTH_EXEMPT_METHODS = frozenset(["/membrain.MemoryUnit/Ping"])

    def __init__(self, tokens: dict[str, str]) -> None:
        """
        Initialize the auth interceptor.

        Args:
            tokens: Dict mapping client_id -> token for authentication.
                   Can also pass a single token string for backward compatibility.
        """
        # Support both single token (string) and multi-client (dict)
        if isinstance(tokens, str):
            tokens = {"default": tokens}

        self._tokens = tokens
        # Pre-compute lowercase bearer prefixes for each token
        self._valid_headers: dict[str, bytes] = {}
        for client_id, token in tokens.items():
            header = f"bearer {token}".lower().encode("utf-8")
            self._valid_headers[client_id] = header

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Any:
        """Intercept and validate incoming requests."""
        # Check if method is exempt from authentication
        method = handler_call_details.method
        if method in self.AUTH_EXEMPT_METHODS:
            return continuation(handler_call_details)

        # Extract authorization header
        metadata = dict(handler_call_details.invocation_metadata)
        auth_value = metadata.get("authorization", "")
        auth_bytes = auth_value.lower().encode("utf-8")

        # Check against all valid tokens using timing-safe comparison
        authenticated = False
        for _client_id, valid_header in self._valid_headers.items():
            if len(auth_bytes) == len(valid_header):
                if hmac.compare_digest(auth_bytes, valid_header):
                    authenticated = True
                    # Could log client_id here for audit, but NOT the token
                    break

        if not authenticated:
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

    Thread-safe: uses locks for all encoder and memory operations.
    """

    def __init__(
        self,
        config: MembrainConfig | None = None,
        *,
        input_dim: int | None = None,
        expansion_ratio: float | None = None,
        n_neurons: int | None = None,
    ) -> None:
        """
        Initialize the memory servicer.

        Args:
            config: MembrainConfig instance (preferred).
            input_dim: Dimension of input embeddings (legacy, use config).
            expansion_ratio: FlyHash expansion ratio (legacy, use config).
            n_neurons: Number of neurons (legacy, use config).
        """
        # Use config if provided, otherwise fall back to legacy args
        if config is None:
            config = MembrainConfig(
                input_dim=input_dim or 1536,
                expansion_ratio=expansion_ratio or 13.0,
                n_neurons=n_neurons or 1000,
            )

        self.config = config
        self.input_dim = config.input_dim
        self.expansion_ratio = config.expansion_ratio

        # Thread safety lock for encoder and memory operations
        self._lock = threading.RLock()

        # Initialize FlyHash encoder
        self.encoder = FlyHash(
            input_dim=config.input_dim,
            expansion_ratio=config.expansion_ratio,
            active_bits=config.active_bits if config.active_bits else 50,
            seed=config.seed,
        )
        self.output_dim = self.encoder.output_dim

        # Initialize BiCameralMemory
        self.memory = BiCameralMemory(
            n_neurons=config.n_neurons,
            dimensions=self.output_dim,
            synapse=config.synapse,
            dt=config.dt,
            seed=config.seed,
            use_attractor=config.use_attractor,
            attractor_learning_rate=config.attractor_learning_rate,
            attractor_max_steps=config.attractor_max_steps,
            use_pes=config.use_pes,
            pes_learning_rate=config.pes_learning_rate,
        )

        # Build the simulator
        self.memory._ensure_simulator()

        logger.info(
            "MemoryUnitServicer initialized: "
            "input_dim=%d, output_dim=%d, n_neurons=%d, seed=%s, use_attractor=%s, use_pes=%s",
            config.input_dim, self.output_dim, config.n_neurons,
            config.seed or "random", config.use_attractor, config.use_pes
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

            # Thread-safe encode and store
            with self._lock:
                # Encode via FlyHash (inside lock for thread safety)
                sparse_vector = self.encoder.encode(vector)

                # Store in memory
                self.memory.remember(
                    context_id=request.context_id,
                    sparse_vector=sparse_vector,
                    importance=importance,
                )

            logger.info("Stored memory: %s", request.context_id)
            return memory_a2a_pb2.Ack(
                success=True,
                message=f"Memory stored: {request.context_id}",
            )

        except Exception:
            logger.exception("Remember failed")
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

            # Validate threshold - reject negative values
            threshold = request.threshold
            if threshold < 0:
                msg = "threshold cannot be negative"
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return memory_a2a_pb2.ContextResponse()

            # Apply defaults for zero/unset values
            if threshold == 0:
                threshold = 0.7

            # Validate threshold range
            threshold_error = self._validate_threshold(threshold)
            if threshold_error:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(threshold_error)
                return memory_a2a_pb2.ContextResponse()

            max_results = request.max_results if request.max_results > 0 else 5

            # Thread-safe encode and recall
            with self._lock:
                # Encode via FlyHash (inside lock for thread safety)
                sparse_vector = self.encoder.encode(vector)

                # Recall from memory
                results = self.memory.recall(
                    query_vector=sparse_vector,
                    threshold=threshold,
                    max_results=max_results,
                    bypass_snn=request.bypass_snn,
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

            logger.info("Recalled %d memories", len(results))
            return memory_a2a_pb2.ContextResponse(
                context_ids=context_ids,
                confidences=confidences,
                overall_confidence=overall_confidence,
            )

        except Exception:
            logger.exception("Recall failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return memory_a2a_pb2.ContextResponse()

    def Consolidate(
        self,
        request: memory_a2a_pb2.SleepSignal,
        context: grpc.ServicerContext,
    ) -> memory_a2a_pb2.ConsolidateResponse:  # type: ignore[name-defined]
        """
        Trigger stochastic consolidation phase (attractor dynamics).

        Injects Gaussian white noise into the network state, then iterates
        recurrent dynamics until the system settles into an attractor state.
        Optionally prunes weak associations.
        """
        try:
            # Use request values or defaults from config
            noise_scale = request.noise_scale if request.noise_scale > 0 else self.config.noise_scale
            max_steps = request.max_steps if request.max_steps > 0 else self.config.max_consolidation_steps
            convergence_threshold = (
                request.convergence_threshold
                if request.convergence_threshold > 0
                else self.config.convergence_threshold
            )
            prune_threshold = (
                request.prune_threshold
                if request.prune_threshold > 0
                else self.config.prune_threshold
            )

            # Consolidate with lock (thread-safe)
            with self._lock:
                steps_to_converge, pruned = self.memory.consolidate(
                    noise_scale=noise_scale,
                    max_steps=max_steps,
                    convergence_threshold=convergence_threshold,
                    prune_weak=request.prune_weak,
                    prune_threshold=prune_threshold,
                )

            if steps_to_converge >= 0:
                message = f"Settled into attractor in {steps_to_converge} steps"
            else:
                message = f"Max steps ({max_steps}) reached without convergence"

            if request.prune_weak:
                message += f", pruned {pruned} weak memories"

            logger.info(message)
            return memory_a2a_pb2.ConsolidateResponse(  # type: ignore[attr-defined]
                success=True,
                steps_to_converge=steps_to_converge,
                pruned_count=pruned,
                message=message,
            )

        except Exception:
            logger.exception("Consolidate failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return memory_a2a_pb2.ConsolidateResponse(  # type: ignore[attr-defined]
                success=False,
                steps_to_converge=-1,
                pruned_count=0,
                message="Internal server error",
            )

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
    Supports token authentication with multiple clients.
    """

    def __init__(
        self,
        config: MembrainConfig | None = None,
        *,
        port: int | None = None,
        max_workers: int | None = None,
        input_dim: int | None = None,
        expansion_ratio: float | None = None,
        n_neurons: int | None = None,
        auth_tokens: dict[str, str] | str | None = None,
    ) -> None:
        """
        Initialize the server.

        Args:
            config: MembrainConfig instance (preferred).
            port: Port to listen on (legacy, use config).
            max_workers: Maximum thread pool workers (legacy, use config).
            input_dim: Dimension of input embeddings (legacy, use config).
            expansion_ratio: FlyHash expansion ratio (legacy, use config).
            n_neurons: Number of neurons (legacy, use config).
            auth_tokens: Dict of client_id->token (legacy, use config).
        """
        # Use config if provided, otherwise build from legacy args
        if config is None:
            tokens = []
            if auth_tokens:
                if isinstance(auth_tokens, str):
                    tokens = [auth_tokens]
                else:
                    tokens = list(auth_tokens.values())

            config = MembrainConfig(
                port=port or 50051,
                max_workers=max_workers or 10,
                input_dim=input_dim or 1536,
                expansion_ratio=expansion_ratio or 13.0,
                n_neurons=n_neurons or 1000,
                auth_tokens=tokens,
            )

        self.config = config
        self.port = config.port
        self.max_workers = config.max_workers

        self.servicer = MemoryUnitServicer(config=config)

        self.server: grpc.Server | None = None
        self._shutdown_requested = False

    def start(self) -> None:
        """Start the gRPC server."""
        # Build interceptors list
        interceptors: list[grpc.ServerInterceptor] = [LoggingInterceptor()]
        if self.config.auth_tokens:
            # Validate all tokens
            for token in self.config.auth_tokens:
                is_valid, error = validate_token(token)
                if not is_valid:
                    raise ValueError(f"Invalid token: {error}")

            # Build token dict for interceptor
            tokens = {f"client_{i}": t for i, t in enumerate(self.config.auth_tokens)}
            interceptors.append(TokenAuthInterceptor(tokens))
            # Log client count, NOT the tokens
            logger.info(
                "Token authentication enabled for %d client(s)",
                len(self.config.auth_tokens)
            )

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
        logger.info("Membrain server started on port %d", self.port)

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


def serve(config: MembrainConfig | None = None) -> None:
    """
    Start the gRPC server with configuration.

    Args:
        config: MembrainConfig instance. If None, loads from environment.

    Environment variables (when config is None):
        MEMBRAIN_PORT: Server port (default: 50051)
        MEMBRAIN_MAX_WORKERS: Thread pool size (default: 10)
        MEMBRAIN_INPUT_DIM: Input embedding dimension (default: 1536)
        MEMBRAIN_EXPANSION_RATIO: FlyHash expansion ratio (default: 13.0)
        MEMBRAIN_N_NEURONS: Number of SNN neurons (default: 1000)
        MEMBRAIN_SEED: Random seed for reproducibility
        MEMBRAIN_AUTH_TOKEN: Single bearer token
        MEMBRAIN_AUTH_TOKENS: Comma-separated tokens for multi-client
    """
    # Load config from environment if not provided
    if config is None:
        config = MembrainConfig.from_env()

    # Validate configuration
    config.validate()

    # Set up structured logging
    setup_logging(
        level=config.log_level,
        log_format=config.log_format,
        log_file=config.log_file,
        include_trace=config.log_include_trace,
    )

    # Log resolved config
    config.log_config(logger)

    if not config.auth_tokens:
        logger.warning(
            "No authentication tokens configured - server running without "
            "authentication! Set MEMBRAIN_AUTH_TOKEN or MEMBRAIN_AUTH_TOKENS "
            "for security."
        )

    server = MembrainServer(config=config)

    # Register signal handlers for graceful shutdown
    def handle_signal(signum: int, frame: object) -> None:
        logger.info("Received signal %d", signum)
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
