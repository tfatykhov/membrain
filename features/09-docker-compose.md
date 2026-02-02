# PR-007 — Docker Compose / One-Command Run

## Status: 🟡 Needs Verification

## Objective

Make `docker compose up -d` work out of the box.

---

## Requirements

### A. Complete docker-compose.yml

```yaml
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
      - MEMBRAIN_N_NEURONS=${MEMBRAIN_N_NEURONS:-1000}
      - MEMBRAIN_AUTH_TOKEN=${MEMBRAIN_AUTH_TOKEN:-}
      - MEMBRAIN_SEED=${MEMBRAIN_SEED:-}
    healthcheck:
      test: ["CMD", "python", "-m", "membrain.health_check"]
      interval: 30s
      timeout: 10s
      start_period: 10s
      retries: 3
    restart: unless-stopped
```

### B. README Quick Start

```markdown
## Quick Start

```bash
git clone https://github.com/tfatykhov/membrain.git
cd membrain
docker compose -f docker/docker-compose.yml up -d
```
```

### C. .env.example

```bash
MEMBRAIN_PORT=50051
MEMBRAIN_INPUT_DIM=1536
MEMBRAIN_N_NEURONS=1000
MEMBRAIN_AUTH_TOKEN=
```

---

## Verification Steps

```bash
docker compose -f docker/docker-compose.yml build --no-cache
docker compose -f docker/docker-compose.yml up -d
sleep 15
docker compose -f docker/docker-compose.yml ps  # Should show (healthy)
```

---

## Files / Modules

| File | Action |
|------|--------|
| `docker/docker-compose.yml` | **Update** |
| `docker/.env.example` | **Create** |
| `README.md` | **Update** |

---

## Acceptance Criteria

- [ ] `docker compose up -d` works from clean clone
- [ ] Health check turns green within 30s
- [ ] README has clear one-command instructions
