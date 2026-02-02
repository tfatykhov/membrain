"""
gRPC Health Check for Docker.

Validates that the Membrain server is responding to Ping RPC.
Used as Docker HEALTHCHECK command.

Exit codes:
    0 - Healthy (Ping succeeded)
    1 - Unhealthy (Ping failed or timeout)
"""

from __future__ import annotations

import os
import sys
import time

import grpc

from membrain.logging import get_logger
from membrain.proto import memory_a2a_pb2, memory_a2a_pb2_grpc

logger = get_logger(__name__)


def check_health(host: str = "localhost", port: int = 50051, timeout: float = 5.0) -> bool:
    """
    Check if Membrain server is healthy by calling Ping RPC.

    Args:
        host: Server hostname.
        port: Server port.
        timeout: RPC timeout in seconds.

    Returns:
        True if healthy, False otherwise.
    """
    start_time = time.perf_counter()
    try:
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = memory_a2a_pb2_grpc.MemoryUnitStub(channel)
        response = stub.Ping(memory_a2a_pb2.Empty(), timeout=timeout)  # type: ignore[attr-defined]
        channel.close()
        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.success:
            logger.info(
                "Health check passed",
                extra={"host": host, "port": port, "latency_ms": round(latency_ms, 2)},
            )
            return True
        else:
            logger.warning(
                "Health check failed: server returned failure",
                extra={"host": host, "port": port},
            )
            return False
    except grpc.RpcError as e:
        logger.error(
            "Health check failed: RPC error",
            extra={"host": host, "port": port, "error": str(e)},
        )
        return False
    except Exception as e:
        logger.error(
            "Health check failed: unexpected error",
            extra={"host": host, "port": port, "error": str(e)},
        )
        return False


def main() -> int:
    """
    Main entry point for health check.

    Reads configuration from environment variables:
        MEMBRAIN_HOST: Server host (default: localhost)
        MEMBRAIN_PORT: Server port (default: 50051)
        MEMBRAIN_HEALTH_TIMEOUT: Timeout in seconds (default: 5)

    Returns:
        Exit code (0 = healthy, 1 = unhealthy).
    """
    host = os.environ.get("MEMBRAIN_HOST", "localhost")
    port = int(os.environ.get("MEMBRAIN_PORT", "50051"))
    timeout = float(os.environ.get("MEMBRAIN_HEALTH_TIMEOUT", "5"))

    if check_health(host, port, timeout):
        print("healthy")
        return 0
    else:
        print("unhealthy")
        return 1


if __name__ == "__main__":
    sys.exit(main())
