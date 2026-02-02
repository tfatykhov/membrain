"""Tests for health check module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from membrain.health_check import check_health, main


class TestCheckHealth:
    """Tests for check_health function."""

    def test_returns_true_on_success(self) -> None:
        """Should return True when Ping succeeds."""
        with patch("membrain.health_check.grpc") as mock_grpc:
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel

            with patch("membrain.health_check.memory_a2a_pb2_grpc") as mock_stub_module:
                mock_stub = MagicMock()
                mock_stub_module.MemoryUnitStub.return_value = mock_stub
                mock_response = MagicMock()
                mock_response.success = True
                mock_stub.Ping.return_value = mock_response

                result = check_health("localhost", 50051, 5.0)

                assert result is True
                mock_grpc.insecure_channel.assert_called_once_with("localhost:50051")
                mock_channel.close.assert_called_once()

    def test_returns_false_on_failure(self) -> None:
        """Should return False when Ping fails."""
        with patch("membrain.health_check.grpc") as mock_grpc:
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel

            with patch("membrain.health_check.memory_a2a_pb2_grpc") as mock_stub_module:
                mock_stub = MagicMock()
                mock_stub_module.MemoryUnitStub.return_value = mock_stub
                mock_response = MagicMock()
                mock_response.success = False
                mock_stub.Ping.return_value = mock_response

                result = check_health("localhost", 50051, 5.0)

                assert result is False

    def test_returns_false_on_rpc_error(self) -> None:
        """Should return False when RPC raises error."""
        with patch("membrain.health_check.grpc.insecure_channel") as mock_channel:
            mock_channel.side_effect = Exception("Connection failed")

            result = check_health("localhost", 50051, 5.0)

            assert result is False


class TestMain:
    """Tests for main entry point."""

    def test_returns_0_when_healthy(self) -> None:
        """Should return exit code 0 when healthy."""
        with patch("membrain.health_check.check_health", return_value=True):
            result = main()
            assert result == 0

    def test_returns_1_when_unhealthy(self) -> None:
        """Should return exit code 1 when unhealthy."""
        with patch("membrain.health_check.check_health", return_value=False):
            result = main()
            assert result == 1

    def test_reads_env_vars(self) -> None:
        """Should read configuration from environment variables."""
        env = {
            "MEMBRAIN_HOST": "custom-host",
            "MEMBRAIN_PORT": "9999",
            "MEMBRAIN_HEALTH_TIMEOUT": "10",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("membrain.health_check.check_health", return_value=True) as mock_check:
                main()
                mock_check.assert_called_once_with("custom-host", 9999, 10.0)

    def test_uses_defaults(self) -> None:
        """Should use default values when env vars not set."""
        # Clear relevant env vars
        env_clear = {
            "MEMBRAIN_HOST": "",
            "MEMBRAIN_PORT": "",
            "MEMBRAIN_HEALTH_TIMEOUT": "",
        }
        with patch.dict(os.environ, env_clear, clear=False):
            # Remove keys entirely
            for key in ["MEMBRAIN_HOST", "MEMBRAIN_PORT", "MEMBRAIN_HEALTH_TIMEOUT"]:
                os.environ.pop(key, None)

            with patch("membrain.health_check.check_health", return_value=True) as mock_check:
                main()
                mock_check.assert_called_once_with("localhost", 50051, 5.0)
