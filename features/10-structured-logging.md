# Feature 10 — Structured Logging

## Status: 🟢 Complete

## Parent Module: Infrastructure

---

## 1. Overview

Add structured JSON logging to all Membrain components for observability, debugging, and production monitoring. Logs should include session tracking, request correlation, and performance metrics.

---

## 2. Requirements

### 2.1 Log Format (JSON)

```json
{
  "timestamp": "2026-02-02T01:59:00.123Z",
  "level": "INFO",
  "logger": "membrain.server",
  "message": "Remember completed",
  "session_id": "abc-123",
  "request_id": "req-456",
  "context_id": "doc-001",
  "duration_ms": 12.5,
  "extra": {}
}
```

### 2.2 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | UTC timestamp with milliseconds |
| `level` | string | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `logger` | string | Module name (e.g., `membrain.server`) |
| `message` | string | Human-readable message |
| `session_id` | string | Optional: tracks a client session |
| `request_id` | string | Optional: correlates request/response |

### 2.3 Component-Specific Fields

**Server (gRPC)**
- `method`: RPC method name (Remember, Recall, Consolidate, Ping)
- `duration_ms`: Request processing time
- `success`: boolean
- `error_code`: gRPC status code if failed

**Core (BiCameralMemory)**
- `context_id`: Memory being stored/recalled
- `memory_count`: Current number of stored memories
- `steps_to_converge`: For consolidation

**Encoder (FlyHash)**
- `input_dim`: Input dimensions
- `sparsity`: Output sparsity rate

---

## 3. Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MEMBRAIN_LOG_LEVEL` | string | INFO | Minimum log level |
| `MEMBRAIN_LOG_FORMAT` | string | json | `json` or `text` (for dev) |
| `MEMBRAIN_LOG_FILE` | string | None | Optional file path |
| `MEMBRAIN_LOG_INCLUDE_TRACE` | bool | false | Include stack traces |

---

## 4. Implementation Details

### 4.1 Logging Infrastructure

```python
# src/membrain/logging.py

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Context variables for request tracking
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add context vars
        if session_id := session_id_var.get():
            log_dict["session_id"] = session_id
        if request_id := request_id_var.get():
            log_dict["request_id"] = request_id
            
        # Add extra fields
        if hasattr(record, "extra"):
            log_dict.update(record.extra)
            
        return json.dumps(log_dict)
```

### 4.2 gRPC Interceptor for Request Tracking

```python
# src/membrain/interceptors.py

class LoggingInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        request_id = str(uuid.uuid4())[:8]
        request_id_var.set(request_id)
        
        start_time = time.perf_counter()
        # ... call handler ...
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            "RPC completed",
            extra={"method": method, "duration_ms": duration_ms}
        )
```

---

## 5. Files / Modules

| File | Action |
|------|--------|
| `src/membrain/logging.py` | **Create** — JSONFormatter, context vars, setup_logging() |
| `src/membrain/interceptors.py` | **Create** — gRPC logging interceptor |
| `src/membrain/config.py` | **Update** — Add log config params |
| `src/membrain/server.py` | **Update** — Use structured logging, add interceptor |
| `src/membrain/core.py` | **Update** — Add structured log calls |
| `src/membrain/encoder.py` | **Update** — Add structured log calls |
| `tests/test_logging.py` | **Create** — Test JSON format, context vars |

---

## 6. Acceptance Criteria

### Test Case 6.1 (JSON Format)
- **Given** `MEMBRAIN_LOG_FORMAT=json`
- **When** any log is emitted
- **Then** output is valid JSON with required fields

### Test Case 6.2 (Request Correlation)
- **Given** a gRPC request
- **When** logs are emitted during processing
- **Then** all logs share the same `request_id`

### Test Case 6.3 (Session Tracking)
- **Given** a client with session metadata
- **When** multiple requests are made
- **Then** all logs include consistent `session_id`

### Test Case 6.4 (Performance Metrics)
- **Given** a Remember/Recall/Consolidate request
- **When** request completes
- **Then** log includes `duration_ms`

### Test Case 6.5 (Text Format Fallback)
- **Given** `MEMBRAIN_LOG_FORMAT=text`
- **When** logs are emitted
- **Then** output is human-readable (for development)

---

## 7. Why This Matters

1. **Observability** — Essential for production debugging
2. **Metrics extraction** — JSON logs can feed into monitoring (Datadog, Loki, etc.)
3. **Request tracing** — Correlate issues across components
4. **Performance profiling** — Track duration_ms for optimization
5. **Audit trail** — Know what happened and when
