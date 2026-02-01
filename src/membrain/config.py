"""Central configuration for Membrain server.

All configuration is loaded from environment variables with sensible defaults.
Use MembrainConfig.from_env() to load configuration at startup.
"""

import logging
import os
from dataclasses import dataclass, field


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer from string."""
    if value is None or value.strip() == "":
        return None
    return int(value)


def _parse_optional_float(value: str | None) -> float | None:
    """Parse an optional float from string."""
    if value is None or value.strip() == "":
        return None
    return float(value)


def _parse_token_list(value: str | None) -> list[str]:
    """Parse comma-separated token list."""
    if value is None or value.strip() == "":
        return []
    return [t.strip() for t in value.split(",") if t.strip()]


@dataclass(frozen=True)
class MembrainConfig:
    """Central configuration for Membrain server.

    All fields have sensible defaults for production use.
    Use from_env() to load from environment variables.
    Use for_testing() to get a fast configuration for CI.
    """

    # Server
    port: int = 50051
    max_workers: int = 10

    # Encoder (FlyHash)
    input_dim: int = 1536
    expansion_ratio: float = 13.0
    active_bits: int | None = None  # None = use default (input_dim // 16)

    # Neural Network
    n_neurons: int = 1000
    dt: float = 0.001  # Simulation timestep (seconds)
    synapse: float = 0.01  # Synapse time constant (seconds)

    # Reproducibility
    seed: int | None = None

    # Authentication
    auth_tokens: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "MembrainConfig":
        """Load configuration from environment variables.

        Environment variables:
            MEMBRAIN_PORT: gRPC server port (default: 50051)
            MEMBRAIN_MAX_WORKERS: Thread pool size (default: 10)
            MEMBRAIN_INPUT_DIM: Input embedding dimension (default: 1536)
            MEMBRAIN_EXPANSION_RATIO: FlyHash expansion ratio (default: 13.0)
            MEMBRAIN_ACTIVE_BITS: Number of active bits in WTA (default: auto)
            MEMBRAIN_N_NEURONS: SNN neuron count (default: 1000)
            MEMBRAIN_DT: Simulation timestep in seconds (default: 0.001)
            MEMBRAIN_SYNAPSE: Synapse time constant in seconds (default: 0.01)
            MEMBRAIN_SEED: Random seed for reproducibility (default: None)
            MEMBRAIN_AUTH_TOKEN: Single auth token (legacy)
            MEMBRAIN_AUTH_TOKENS: Comma-separated auth tokens
        """
        # Parse auth tokens (support both single and multi-token formats)
        tokens = _parse_token_list(os.environ.get("MEMBRAIN_AUTH_TOKENS"))
        single_token = os.environ.get("MEMBRAIN_AUTH_TOKEN", "").strip()
        if single_token and single_token not in tokens:
            tokens.append(single_token)

        return cls(
            port=int(os.environ.get("MEMBRAIN_PORT", 50051)),
            max_workers=int(os.environ.get("MEMBRAIN_MAX_WORKERS", 10)),
            input_dim=int(os.environ.get("MEMBRAIN_INPUT_DIM", 1536)),
            expansion_ratio=float(os.environ.get("MEMBRAIN_EXPANSION_RATIO", 13.0)),
            active_bits=_parse_optional_int(os.environ.get("MEMBRAIN_ACTIVE_BITS")),
            n_neurons=int(os.environ.get("MEMBRAIN_N_NEURONS", 1000)),
            dt=float(os.environ.get("MEMBRAIN_DT", 0.001)),
            synapse=float(os.environ.get("MEMBRAIN_SYNAPSE", 0.01)),
            seed=_parse_optional_int(os.environ.get("MEMBRAIN_SEED")),
            auth_tokens=tokens,
        )

    @classmethod
    def for_testing(cls) -> "MembrainConfig":
        """Get a fast configuration for CI/testing.

        Uses small dimensions to speed up tests significantly.
        """
        return cls(
            port=50051,
            max_workers=2,
            input_dim=64,
            expansion_ratio=4.0,
            active_bits=4,
            n_neurons=50,
            dt=0.001,
            synapse=0.01,
            seed=42,
            auth_tokens=["test-token-for-ci"],
        )

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port} (must be 1-65535)")

        if self.max_workers < 1:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")

        if self.input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")

        if self.expansion_ratio < 1.0:
            raise ValueError(
                f"expansion_ratio must be >= 1.0, got {self.expansion_ratio}"
            )

        if self.active_bits is not None and self.active_bits < 1:
            raise ValueError(f"active_bits must be positive, got {self.active_bits}")

        if self.n_neurons < 1:
            raise ValueError(f"n_neurons must be positive, got {self.n_neurons}")

        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

        if self.synapse <= 0:
            raise ValueError(f"synapse must be positive, got {self.synapse}")

        # Validate auth tokens (if any are provided)
        for token in self.auth_tokens:
            if len(token) < 16:
                raise ValueError(
                    f"Auth token too short (min 16 chars): {token[:4]}..."
                )

    def log_config(self, logger: logging.Logger) -> None:
        """Log resolved configuration (masking sensitive values)."""
        logger.info("Membrain configuration:")
        logger.info(f"  port: {self.port}")
        logger.info(f"  max_workers: {self.max_workers}")
        logger.info(f"  input_dim: {self.input_dim}")
        logger.info(f"  expansion_ratio: {self.expansion_ratio}")
        logger.info(f"  active_bits: {self.active_bits or 'auto'}")
        logger.info(f"  n_neurons: {self.n_neurons}")
        logger.info(f"  dt: {self.dt}")
        logger.info(f"  synapse: {self.synapse}")
        logger.info(f"  seed: {self.seed or 'random'}")
        logger.info(f"  auth_tokens: {len(self.auth_tokens)} configured")
