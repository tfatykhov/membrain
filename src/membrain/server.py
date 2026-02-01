"""
Membrain gRPC Server

Serves the MemoryUnit A2A interface for LLM agents.
"""

import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def serve() -> None:
    """Start the gRPC server."""
    # TODO: Implement actual gRPC server
    logger.info("Membrain server starting...")
    logger.info("Server not yet implemented. This is a stub.")
    logger.info("See PRD for implementation details.")

    # Keep alive for container health checks
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")


def main() -> None:
    """Entry point for the server module."""
    serve()


if __name__ == "__main__":
    main()
