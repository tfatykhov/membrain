"""Tests for Membrain configuration system."""

import os
from unittest.mock import patch

import pytest

from membrain.config import MembrainConfig


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_port(self) -> None:
        config = MembrainConfig()
        assert config.port == 50051

    def test_default_input_dim(self) -> None:
        config = MembrainConfig()
        assert config.input_dim == 1536

    def test_default_expansion_ratio(self) -> None:
        config = MembrainConfig()
        assert config.expansion_ratio == 13.0

    def test_default_n_neurons(self) -> None:
        config = MembrainConfig()
        assert config.n_neurons == 1000

    def test_default_seed_is_none(self) -> None:
        config = MembrainConfig()
        assert config.seed is None

    def test_default_auth_tokens_empty(self) -> None:
        config = MembrainConfig()
        assert config.auth_tokens == []


class TestConfigFromEnv:
    """Test loading configuration from environment variables."""

    def test_loads_port_from_env(self) -> None:
        with patch.dict(os.environ, {"MEMBRAIN_PORT": "8080"}):
            config = MembrainConfig.from_env()
            assert config.port == 8080

    def test_loads_input_dim_from_env(self) -> None:
        with patch.dict(os.environ, {"MEMBRAIN_INPUT_DIM": "768"}):
            config = MembrainConfig.from_env()
            assert config.input_dim == 768

    def test_loads_expansion_ratio_from_env(self) -> None:
        with patch.dict(os.environ, {"MEMBRAIN_EXPANSION_RATIO": "10.5"}):
            config = MembrainConfig.from_env()
            assert config.expansion_ratio == 10.5

    def test_loads_seed_from_env(self) -> None:
        with patch.dict(os.environ, {"MEMBRAIN_SEED": "42"}):
            config = MembrainConfig.from_env()
            assert config.seed == 42

    def test_loads_single_auth_token(self) -> None:
        with patch.dict(os.environ, {"MEMBRAIN_AUTH_TOKEN": "my-secret-token-1234"}):
            config = MembrainConfig.from_env()
            assert "my-secret-token-1234" in config.auth_tokens

    def test_loads_multiple_auth_tokens(self) -> None:
        with patch.dict(
            os.environ, {"MEMBRAIN_AUTH_TOKENS": "token-one-1234,token-two-5678"}
        ):
            config = MembrainConfig.from_env()
            assert "token-one-1234" in config.auth_tokens
            assert "token-two-5678" in config.auth_tokens

    def test_combines_single_and_multiple_tokens(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MEMBRAIN_AUTH_TOKEN": "single-token-here",
                "MEMBRAIN_AUTH_TOKENS": "multi-token-1234",
            },
        ):
            config = MembrainConfig.from_env()
            assert "single-token-here" in config.auth_tokens
            assert "multi-token-1234" in config.auth_tokens

    def test_empty_env_uses_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = MembrainConfig.from_env()
            assert config.port == 50051
            assert config.input_dim == 1536


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config_passes(self) -> None:
        config = MembrainConfig()
        config.validate()  # Should not raise

    def test_invalid_port_zero_raises(self) -> None:
        config = MembrainConfig(port=0)
        with pytest.raises(ValueError, match="Invalid port"):
            config.validate()

    def test_invalid_port_negative_raises(self) -> None:
        config = MembrainConfig(port=-1)
        with pytest.raises(ValueError, match="Invalid port"):
            config.validate()

    def test_invalid_port_too_high_raises(self) -> None:
        config = MembrainConfig(port=70000)
        with pytest.raises(ValueError, match="Invalid port"):
            config.validate()

    def test_invalid_input_dim_raises(self) -> None:
        config = MembrainConfig(input_dim=0)
        with pytest.raises(ValueError, match="input_dim must be positive"):
            config.validate()

    def test_invalid_expansion_ratio_raises(self) -> None:
        config = MembrainConfig(expansion_ratio=0.5)
        with pytest.raises(ValueError, match="expansion_ratio must be"):
            config.validate()

    def test_invalid_n_neurons_raises(self) -> None:
        config = MembrainConfig(n_neurons=0)
        with pytest.raises(ValueError, match="n_neurons must be positive"):
            config.validate()

    def test_invalid_dt_raises(self) -> None:
        config = MembrainConfig(dt=0)
        with pytest.raises(ValueError, match="dt must be positive"):
            config.validate()

    def test_short_auth_token_raises(self) -> None:
        config = MembrainConfig(auth_tokens=["short"])
        with pytest.raises(ValueError, match="Auth token too short"):
            config.validate()

    def test_valid_auth_token_passes(self) -> None:
        config = MembrainConfig(auth_tokens=["this-is-a-valid-token"])
        config.validate()  # Should not raise


class TestConfigForTesting:
    """Test the for_testing() factory method."""

    def test_for_testing_has_small_dims(self) -> None:
        config = MembrainConfig.for_testing()
        assert config.input_dim == 64
        assert config.n_neurons == 50

    def test_for_testing_has_seed(self) -> None:
        config = MembrainConfig.for_testing()
        assert config.seed == 42

    def test_for_testing_has_test_token(self) -> None:
        config = MembrainConfig.for_testing()
        assert len(config.auth_tokens) == 1

    def test_for_testing_is_valid(self) -> None:
        config = MembrainConfig.for_testing()
        config.validate()  # Should not raise


class TestConfigImmutability:
    """Test that config is immutable (frozen dataclass)."""

    def test_cannot_modify_port(self) -> None:
        config = MembrainConfig()
        with pytest.raises(AttributeError):
            config.port = 8080  # type: ignore[misc]

    def test_cannot_modify_input_dim(self) -> None:
        config = MembrainConfig()
        with pytest.raises(AttributeError):
            config.input_dim = 768  # type: ignore[misc]
