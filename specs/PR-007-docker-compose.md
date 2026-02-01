# PR-007 — Docker Compose / "One-command Run" Documentation Fix

## Status: 🟡 Partial — Needs Verification

## Current State Analysis

### What Exists

`docker/docker-compose.yml` exists with basic structure:
```yaml
services:
  membrain:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "50051:50051"
    environment:
      - MEMBRAIN_PORT=50051
```

### What's Missing / Unknown

1. **Verification needed** — Does `docker compose up -d` actually work?
2. **Healthcheck alignment** — Uses socket check, not gRPC (to be fixed in PR-006)
3. **Documentation** — README may not have clear one-liner instructions
4. **Volume persistence** — No volumes for state persistence

---

## Objective

Make `docker compose up -d` actually work out of the box with a single command.

---

## Detailed Requirements

### A. Verify and Fix docker-compose.yml

Complete docker-compose.yml:

```yaml
# docker/docker-compose.yml
version: "3.8"

services:
  membrain:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: membrain:latest
    container_name: membrain
    ports:
      - "${MEMBRAIN_PORT:-50051}:50051"
    environment:
      - MEMBRAIN_PORT=50051
      - MEMBRAIN_INPUT_DIM=${MEMBRAIN_INPUT_DIM:-1536}
      - MEMBRAIN_EXPANSION_RATIO=${MEMBRAIN_EXPANSION_RATIO:-13.0}
      - MEMBRAIN_N_NEURONS=${MEMBRAIN_N_NEURONS:-1000}
      - MEMBRAIN_AUTH_TOKEN=${MEMBRAIN_AUTH_TOKEN:-}
      - MEMBRAIN_SEED=${MEMBRAIN_SEED:-}
    volumes:
      - membrain-data:/app/data  # For future persistence
    healthcheck:
      test: ["CMD", "python", "-m", "membrain.health_check"]
      interval: 30s
      timeout: 10s
      start_period: 10s
      retries: 3
    restart: unless-stopped

volumes:
  membrain-data:
```

### B. Dockerfile Verification

Ensure Dockerfile:
1. Installs all dependencies
2. Copies source correctly
3. Sets proper entrypoint
4. Exposes correct port

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY protos/ protos/

# Install package
RUN pip install --no-cache-dir .

# Expose gRPC port
EXPOSE 50051

# Health check (uses gRPC Ping after PR-006)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -m membrain.health_check || exit 1

# Run server
CMD ["python", "-m", "membrain.server"]
```

### C. README Quick Start

Add to README.md:

```markdown
## Quick Start

### Docker (Recommended)

```bash
# Clone and run
git clone https://github.com/tfatykhov/membrain.git
cd membrain

# Start with docker compose
docker compose -f docker/docker-compose.yml up -d

# Check health
docker compose -f docker/docker-compose.yml ps

# View logs
docker compose -f docker/docker-compose.yml logs -f

# Stop
docker compose -f docker/docker-compose.yml down
```

### Configuration

Set environment variables before running:

```bash
# With authentication (recommended)
export MEMBRAIN_AUTH_TOKEN="your-secure-token-here"
docker compose -f docker/docker-compose.yml up -d

# With custom dimensions
export MEMBRAIN_INPUT_DIM=3072  # For text-embedding-3-large
docker compose -f docker/docker-compose.yml up -d
```

### Test the Service

```bash
# Using grpcurl (install: https://github.com/fullstorydev/grpcurl)
grpcurl -plaintext localhost:50051 membrain.MemoryUnit/Ping
```
```

### D. .env.example File

Create `docker/.env.example`:

```bash
# Membrain Docker Configuration

# Server
MEMBRAIN_PORT=50051

# Encoder
MEMBRAIN_INPUT_DIM=1536
MEMBRAIN_EXPANSION_RATIO=13.0

# Neural Network
MEMBRAIN_N_NEURONS=1000

# Authentication (REQUIRED for production)
MEMBRAIN_AUTH_TOKEN=

# Reproducibility (optional)
MEMBRAIN_SEED=
```

---

## Files / Modules

| File | Action |
|------|--------|
| `docker/docker-compose.yml` | **Update** — Complete config |
| `docker/Dockerfile` | **Verify/Update** — Ensure it builds |
| `docker/.env.example` | **Create** — Example config |
| `README.md` | **Update** — Quick start section |

---

## Verification Steps

```bash
# 1. Clean build
docker compose -f docker/docker-compose.yml build --no-cache

# 2. Start
docker compose -f docker/docker-compose.yml up -d

# 3. Wait for health
sleep 15

# 4. Check status
docker compose -f docker/docker-compose.yml ps
# Should show: membrain ... (healthy)

# 5. Test Ping (with grpcurl)
grpcurl -plaintext localhost:50051 membrain.MemoryUnit/Ping
# Expected: { "success": true, "message": "pong" }

# 6. Cleanup
docker compose -f docker/docker-compose.yml down
```

---

## Tests

### CI Integration Test

Add `.github/workflows/docker-test.yml`:

```yaml
name: Docker Build Test

on: [push, pull_request]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker compose -f docker/docker-compose.yml build
      
      - name: Start container
        run: docker compose -f docker/docker-compose.yml up -d
      
      - name: Wait for health
        run: |
          for i in {1..30}; do
            if docker compose -f docker/docker-compose.yml ps | grep -q "(healthy)"; then
              echo "Container is healthy"
              exit 0
            fi
            sleep 2
          done
          echo "Container did not become healthy"
          docker compose -f docker/docker-compose.yml logs
          exit 1
      
      - name: Cleanup
        if: always()
        run: docker compose -f docker/docker-compose.yml down
```

---

## Acceptance Criteria

- [ ] `docker compose up -d` works from clean clone
- [ ] Health check turns green within 30s
- [ ] README has clear one-command instructions
- [ ] `.env.example` documents all config options
- [ ] CI verifies Docker build works

---

## Risks / Notes

- **Build time**: First build may take a few minutes (pip install)
- **Health check dependency**: Requires PR-006 for gRPC health check
- **Ordering**: Can merge before PR-006 with socket-based healthcheck, update later

---

## Definition of Done

- [ ] Verified `docker compose up -d` works on clean machine
- [ ] README updated with quick start
- [ ] CI job for Docker build added (optional)
- [ ] Logging shows server startup
