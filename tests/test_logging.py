"""Tests for structured logging."""

import json
import logging
from io import StringIO

import pytest

from membrain.logging import (
    JSONFormatter,
    StructuredLoggerAdapter,
    TextFormatter,
    get_logger,
    request_id_var,
    session_id_var,
    setup_logging,
)


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    @pytest.fixture
    def formatter(self) -> JSONFormatter:
        return JSONFormatter(include_trace=False)

    def test_formats_as_valid_json(self, formatter: JSONFormatter) -> None:
        """Output should be valid JSON."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_includes_context_vars(self, formatter: JSONFormatter) -> None:
        """Should include session_id and request_id when set."""
        session_id_var.set("sess-123")
        request_id_var.set("req-456")

        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )
            output = formatter.format(record)
            parsed = json.loads(output)

            assert parsed["session_id"] == "sess-123"
            assert parsed["request_id"] == "req-456"
        finally:
            session_id_var.set(None)
            request_id_var.set(None)

    def test_includes_extra_fields(self, formatter: JSONFormatter) -> None:
        """Should include extra fields from record."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra = {"method": "Remember", "duration_ms": 12.5}  # type: ignore[attr-defined]
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["method"] == "Remember"
        assert parsed["duration_ms"] == 12.5


class TestTextFormatter:
    """Tests for TextFormatter (development mode)."""

    @pytest.fixture
    def formatter(self) -> TextFormatter:
        return TextFormatter()

    def test_formats_as_human_readable(self, formatter: TextFormatter) -> None:
        """Output should be human-readable text."""
        record = logging.LogRecord(
            name="membrain.server",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Server started",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)

        assert "INFO" in output
        assert "membrain.server" in output
        assert "Server started" in output
        # Should NOT be JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(output)

    def test_includes_request_id_prefix(self, formatter: TextFormatter) -> None:
        """Should include request_id in prefix when set."""
        request_id_var.set("abc123")

        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )
            output = formatter.format(record)

            assert "[abc123]" in output
        finally:
            request_id_var.set(None)


class TestStructuredLoggerAdapter:
    """Tests for StructuredLoggerAdapter."""

    def test_passes_extra_fields(self) -> None:
        """Extra fields should be passed to log records."""
        base_logger = logging.getLogger("test.adapter")
        adapter = StructuredLoggerAdapter(base_logger, {"default": "value"})

        # Create a handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        try:
            adapter.info("Test message", extra={"custom": "field"})
            output = stream.getvalue()
            parsed = json.loads(output.strip())

            assert parsed["message"] == "Test message"
            assert "extra" in parsed or "custom" in parsed
        finally:
            base_logger.removeHandler(handler)


class TestSetupLogging:
    """Tests for setup_logging()."""

    def test_json_format(self) -> None:
        """Should configure JSON formatting."""
        setup_logging(level="DEBUG", log_format="json")
        logger = logging.getLogger("test.setup.json")

        # Check that root logger has JSON formatter
        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_text_format(self) -> None:
        """Should configure text formatting."""
        setup_logging(level="DEBUG", log_format="text")
        logger = logging.getLogger("test.setup.text")

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, TextFormatter)

    def test_suppresses_noisy_loggers(self) -> None:
        """Should suppress nengo and grpc loggers."""
        setup_logging()

        assert logging.getLogger("nengo").level >= logging.WARNING
        assert logging.getLogger("grpc").level >= logging.WARNING


class TestGetLogger:
    """Tests for get_logger()."""

    def test_returns_adapter(self) -> None:
        """Should return a StructuredLoggerAdapter."""
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLoggerAdapter)

    def test_can_log_with_extra(self) -> None:
        """Should be able to log with extra fields."""
        setup_logging(level="DEBUG", log_format="json")
        logger = get_logger("test.extra")

        # This should not raise
        logger.info("Test", extra={"key": "value"})
