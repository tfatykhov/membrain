"""Membrain gRPC adapter — wraps Membrain server for benchmarking.

Adapts the Membrain gRPC interface to the VectorStore protocol,
enabling direct comparison with baseline implementations.

Requires a running Membrain server (see docker-compose.yml).
"""

from typing import Any
import sys

import grpc
import numpy as np
from numpy.typing import NDArray

from membrain.proto import memory_a2a_pb2
from membrain.proto import memory_a2a_pb2_grpc


class MembrainStore:
    """VectorStore adapter for Membrain gRPC server.
    
    Wraps the Membrain MemoryUnit service to implement the VectorStore protocol.
    Enables benchmarking Membrain against baseline vector stores.
    
    Note: Unlike in-memory baselines, this requires a running server.
    Use docker-compose up to start the server before benchmarking.
    
    Attributes:
        host: Membrain server hostname
        port: Membrain server port
        api_key: Optional API key for authentication
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        api_key: str | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        """Initialize connection to Membrain server.
        
        Args:
            host: Server hostname
            port: Server port
            api_key: API key for authentication (if server requires it)
            timeout_s: RPC timeout in seconds
        """
        self._host = host
        self._port = port
        self._api_key = api_key
        self._timeout = timeout_s
        
        # Connection state
        self._channel: grpc.Channel | None = None
        self._stub: memory_a2a_pb2_grpc.MemoryUnitStub | None = None
        
        # Local tracking (server doesn't expose count/keys directly)
        self._keys: list[str] = []
        self._key_set: set[str] = set()
        self._dim: int = 0
        self._stored_bytes: int = 0
        
    def _ensure_connected(self) -> memory_a2a_pb2_grpc.MemoryUnitStub:
        """Ensure gRPC channel is connected, return stub."""
        if self._stub is None:
            target = f"{self._host}:{self._port}"
            self._channel = grpc.insecure_channel(target)
            self._stub = memory_a2a_pb2_grpc.MemoryUnitStub(self._channel)
            
            # Verify connection with ping
            try:
                self._call_with_auth(
                    self._stub.Ping,
                    memory_a2a_pb2.Empty()
                )
            except grpc.RpcError as e:
                self._stub = None
                self._channel = None
                raise ConnectionError(f"Failed to connect to Membrain at {target}: {e}")
        
        return self._stub
    
    def _call_with_auth(self, method: Any, request: Any) -> Any:
        """Call gRPC method with optional authentication."""
        metadata: list[tuple[str, str]] = []
        if self._api_key:
            metadata.append(("authorization", f"Bearer {self._api_key}"))
        
        return method(request, timeout=self._timeout, metadata=metadata or None)
    
    def store(self, key: str, vector: NDArray[np.floating]) -> None:
        """Store a vector in Membrain.
        
        Maps to Remember RPC with context_id = key.
        """
        if key in self._key_set:
            raise ValueError(f"Duplicate key: {key}")
        
        stub = self._ensure_connected()
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            raise ValueError("Cannot store zero vector")
        normalized = vector / norm
        
        # Create MemoryPacket
        packet = memory_a2a_pb2.MemoryPacket(
            context_id=key,
            vector=normalized.astype(np.float32).tolist(),
            importance=1.0,
        )
        
        response = self._call_with_auth(stub.Remember, packet)
        
        if not response.success:
            raise RuntimeError(f"Remember failed: {response.message}")
        
        self._keys.append(key)
        self._key_set.add(key)
        if self._dim == 0:
            self._dim = len(vector)
        self._stored_bytes += len(vector) * 4  # float32
    
    def store_batch(self, items: list[tuple[str, NDArray[np.floating]]]) -> None:
        """Bulk insert — calls store() for each item.
        
        Membrain doesn't have a batch RPC, so we iterate.
        """
        if not items:
            return
        
        # Check for duplicates (within batch and existing)
        seen: set[str] = set()
        for key, vec in items:
            if key in self._key_set:
                raise ValueError(f"Duplicate key: {key}")
            if key in seen:
                raise ValueError(f"Duplicate key in batch: {key}")
            seen.add(key)
            
            # Validate dimensions
            if self._dim > 0 and len(vec) != self._dim:
                raise ValueError(
                    f"Dimension mismatch for key '{key}': "
                    f"expected {self._dim}, got {len(vec)}"
                )
        
        for key, vec in items:
            self.store(key, vec)
    
    def query(
        self,
        vector: NDArray[np.floating],
        k: int = 1,
    ) -> list[tuple[str, float]]:
        """Query Membrain for similar memories.
        
        Maps to Recall RPC.
        """
        if not self._keys:
            return []
        
        stub = self._ensure_connected()
        
        # Normalize query
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return []
        normalized = vector / norm
        
        # Create QueryPacket
        query = memory_a2a_pb2.QueryPacket(
            vector=normalized.astype(np.float32).tolist(),
            threshold=0.0,  # Return all matches
            max_results=k,
        )
        
        response = self._call_with_auth(stub.Recall, query)
        
        # Combine context_ids and confidences
        results: list[tuple[str, float]] = []
        for ctx_id, conf in zip(response.context_ids, response.confidences):
            results.append((ctx_id, float(conf)))
        
        # Sort by confidence descending (should already be sorted, but ensure)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def clear(self) -> None:
        """Clear local tracking state.
        
        Note: This does NOT clear the Membrain server.
        Server-side clearing requires restart or a dedicated RPC (not implemented).
        For benchmarks, restart the server between runs.
        """
        self._keys.clear()
        self._key_set.clear()
        self._dim = 0
        self._stored_bytes = 0
    
    def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
    
    @property
    def count(self) -> int:
        """Number of stored vectors (local tracking)."""
        return len(self._keys)
    
    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim
    
    @property
    def memory_mb(self) -> float:
        """Approximate memory usage.
        
        Note: This only tracks client-side. Server memory is separate.
        """
        key_bytes = sum(sys.getsizeof(k) for k in self._keys)
        set_bytes = sys.getsizeof(self._key_set)
        return (self._stored_bytes + key_bytes + set_bytes) / (1024 * 1024)
    
    @property
    def is_connected(self) -> bool:
        """Whether currently connected to server."""
        return self._stub is not None
    
    def __enter__(self) -> "MembrainStore":
        """Context manager entry."""
        self._ensure_connected()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
