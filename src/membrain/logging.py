"""Structured JSON logging for Membrain.

Provides:
- JSONFormatter for structured log output
- Context variables for request/session tracking
- setup_logging() for consistent initialization
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variables for request correlation
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def __init__(self, include_trace: bool = False) -> None:
        """Initialize formatter.

        Args:
            include_trace: Whether to include stack traces in error logs.
        """
        super().__init__()
        self.include_trace = include_trace

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_dict: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context variables if set
        if session_id := session_id_var.get():
            log_dict["session_id"] = session_id
        if request_id := request_id_var.get():
            log_dict["request_id"] = request_id

        # Add extra fields from record
        if hasattr(record, "extra") and isinstance(record.extra, dict):  # type: ignore[attr-defined]
            log_dict.update(record.extra)  # type: ignore[attr-defined]

        # Add exception info if present
        if record.exc_info and self.include_trace:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as human-readable text.

        Args:
            record: The log record to format.

        Returns:
            Formatted string.
        """
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        level = record.levelname[:4]

        # Build prefix with context if available
        prefix_parts = [f"{timestamp} {level:4} {record.name}"]
        if request_id := request_id_var.get():
            prefix_parts.append(f"[{request_id}]")

        prefix = " ".join(prefix_parts)
        message = record.getMessage()

        # Add extra fields inline
        if hasattr(record, "extra") and isinstance(record.extra, dict):  # type: ignore[attr-defined]
            extras = " ".join(f"{k}={v}" for k, v in record.extra.items())  # type: ignore[attr-defined]
            if extras:
                message = f"{message} | {extras}"

        result = f"{prefix}: {message}"

        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)

        return result


class StructuredLoggerAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """Logger adapter that supports extra fields."""

    def process(
        self, msg: str, kwargs: Any
    ) -> tuple[str, Any]:
        """Process log message to include extra fields.

        Args:
            msg: The log message.
            kwargs: Keyword arguments.

        Returns:
            Processed message and kwargs.
        """
        extra = kwargs.get("extra", {})
        if self.extra:
            extra = {**self.extra, **extra}
        kwargs["extra"] = {"extra": extra}
        return msg, kwargs


def get_logger(name: str) -> StructuredLoggerAdapter:
    """Get a structured logger for a module.

    Args:
        name: Logger name (usually __name__).

    Returns:
        StructuredLoggerAdapter with extra field support.
    """
    return StructuredLoggerAdapter(logging.getLogger(name), {})


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: str | None = None,
    include_trace: bool = False,
) -> None:
    """Configure structured logging for Membrain.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format ('json' or 'text').
        log_file: Optional file path for log output.
        include_trace: Whether to include stack traces in error logs.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    if log_format.lower() == "json":
        formatter: logging.Formatter = JSONFormatter(include_trace=include_trace)
    else:
        formatter = TextFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("nengo").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)
