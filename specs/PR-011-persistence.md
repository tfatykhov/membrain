# PR-011 — Persistence v1 (Snapshots) + Minimal Governance

## Status: 🔴 Not Started — P1 Priority

## Current State Analysis

### What Exists
- In-memory storage only
- All memories lost on restart
- No persistence mechanism

### What's Missing
- **Snapshot capability**: Save memory state to disk
- **Restore on startup**: Load previous state
- **Version governance**: Handle schema changes safely
- **CLI commands**: Manual snapshot/restore

---

## Objective

Persist memory across restarts via snapshots with versioning and schema governance.

---

## Detailed Requirements

### A. Snapshot Format

Define a versioned snapshot format:

```python
# src/membrain/persistence.py

from __future__ import annotations

import json
import gzip
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

SNAPSHOT_VERSION = "1.0.0"

@dataclass
class SnapshotMetadata:
    """Metadata for a memory snapshot."""
    version: str
    created_at: float  # Unix timestamp
    n_memories: int
    encoder_config: dict[str, Any]
    memory_config: dict[str, Any]

@dataclass
class MemorySnapshot:
    """Complete memory state snapshot."""
    metadata: SnapshotMetadata
    entries: list[dict[str, Any]]  # Serialized MemoryEntry objects
    transitions: dict[str, Any]  # Serialized TransitionStore (PR-010)
    
    def to_bytes(self) -> bytes:
        """Serialize snapshot to compressed bytes."""
        data = {
            "metadata": asdict(self.metadata),
            "entries": self.entries,
            "transitions": self.transitions,
        }
        json_bytes = json.dumps(data, cls=NumpyEncoder).encode("utf-8")
        return gzip.compress(json_bytes)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "MemorySnapshot":
        """Deserialize snapshot from compressed bytes."""
        json_bytes = gzip.decompress(data)
        data = json.loads(json_bytes.decode("utf-8"))
        
        metadata = SnapshotMetadata(**data["metadata"])
        return cls(
            metadata=metadata,
            entries=data["entries"],
            transitions=data.get("transitions", {}),
        )
    
    def save(self, path: Path | str) -> None:
        """Save snapshot to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_bytes())
    
    @classmethod
    def load(cls, path: Path | str) -> "MemorySnapshot":
        """Load snapshot from file."""
        path = Path(path)
        return cls.from_bytes(path.read_bytes())

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__numpy__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        return super().default(obj)

def decode_numpy(obj):
    """JSON decoder hook for numpy arrays."""
    if isinstance(obj, dict) and obj.get("__numpy__"):
        return np.array(obj["data"], dtype=obj["dtype"])
    return obj
```

### B. MemoryEntry Serialization

Update `core.py` to support serialization:

```python
@dataclass
class MemoryEntry:
    context_id: str
    sparse_vector: NDArray[np.float32]
    importance: float
    stored_at: float
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "context_id": self.context_id,
            "sparse_vector": self.sparse_vector.tolist(),
            "importance": self.importance,
            "stored_at": self.stored_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Deserialize from dictionary."""
        return cls(
            context_id=data["context_id"],
            sparse_vector=np.array(data["sparse_vector"], dtype=np.float32),
            importance=data["importance"],
            stored_at=data["stored_at"],
        )
```

### C. BiCameralMemory Snapshot Methods

```python
class BiCameralMemory:
    def create_snapshot(self, encoder_config: dict, memory_config: dict) -> MemorySnapshot:
        """Create a snapshot of current memory state."""
        entries = [entry.to_dict() for entry in self._entries.values()]
        
        # Include transitions if available (PR-010)
        transitions = {}
        if hasattr(self, 'transitions'):
            transitions = self.transitions.to_dict()
        
        metadata = SnapshotMetadata(
            version=SNAPSHOT_VERSION,
            created_at=time.time(),
            n_memories=len(entries),
            encoder_config=encoder_config,
            memory_config=memory_config,
        )
        
        return MemorySnapshot(
            metadata=metadata,
            entries=entries,
            transitions=transitions,
        )
    
    def restore_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Restore memory state from snapshot."""
        # Version check
        if not self._is_compatible_version(snapshot.metadata.version):
            raise ValueError(
                f"Incompatible snapshot version: {snapshot.metadata.version}, "
                f"expected {SNAPSHOT_VERSION}"
            )
        
        # Clear current state
        self._entries.clear()
        
        # Restore entries
        for entry_data in snapshot.entries:
            entry = MemoryEntry.from_dict(entry_data)
            self._entries[entry.context_id] = entry
        
        # Restore transitions if present
        if snapshot.transitions and hasattr(self, 'transitions'):
            self.transitions = TransitionStore.from_dict(snapshot.transitions)
    
    def _is_compatible_version(self, version: str) -> bool:
        """Check if snapshot version is compatible."""
        current_major = SNAPSHOT_VERSION.split(".")[0]
        snapshot_major = version.split(".")[0]
        return current_major == snapshot_major
```

### D. Server Integration

```python
class MembrainServer:
    def __init__(self, ..., snapshot_path: str | None = None):
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        
        # Restore on startup if snapshot exists
        if self.snapshot_path and self.snapshot_path.exists():
            self._restore_from_snapshot()
    
    def _restore_from_snapshot(self) -> None:
        """Restore memory from snapshot file."""
        try:
            snapshot = MemorySnapshot.load(self.snapshot_path)
            self.servicer.memory.restore_snapshot(snapshot)
            logger.info(
                "Restored %d memories from snapshot (v%s, created %s)",
                snapshot.metadata.n_memories,
                snapshot.metadata.version,
                time.ctime(snapshot.metadata.created_at),
            )
        except Exception as e:
            logger.error("Failed to restore snapshot: %s", e)
            raise
    
    def save_snapshot(self) -> None:
        """Save current memory state to snapshot file."""
        if not self.snapshot_path:
            raise ValueError("No snapshot path configured")
        
        snapshot = self.servicer.memory.create_snapshot(
            encoder_config=self._get_encoder_config(),
            memory_config=self._get_memory_config(),
        )
        snapshot.save(self.snapshot_path)
        logger.info("Saved snapshot with %d memories", snapshot.metadata.n_memories)
```

### E. CLI Commands

Update `cli.py`:

```python
@click.group()
def cli():
    """Membrain CLI."""
    pass

@cli.command()
@click.option("--port", default=50051, help="Server port")
@click.option("--snapshot", default=None, help="Snapshot file path")
def serve(port: int, snapshot: str | None):
    """Start the Membrain server."""
    server = MembrainServer(port=port, snapshot_path=snapshot)
    server.start()
    server.wait_for_termination()

@cli.command()
@click.argument("input_path")
@click.argument("output_path")
def migrate(input_path: str, output_path: str):
    """Migrate a snapshot to current version."""
    # Load old snapshot
    old_snapshot = MemorySnapshot.load(input_path)
    
    # Migrate (future: add migration logic)
    if old_snapshot.metadata.version == SNAPSHOT_VERSION:
        click.echo("Snapshot already at current version")
        return
    
    # Save migrated snapshot
    # ... migration logic ...
    click.echo(f"Migrated {input_path} -> {output_path}")

@cli.command()
@click.argument("path")
def inspect(path: str):
    """Inspect a snapshot file."""
    snapshot = MemorySnapshot.load(path)
    click.echo(f"Version: {snapshot.metadata.version}")
    click.echo(f"Created: {time.ctime(snapshot.metadata.created_at)}")
    click.echo(f"Memories: {snapshot.metadata.n_memories}")
    click.echo(f"Encoder config: {snapshot.metadata.encoder_config}")
```

### F. Automatic Snapshots (Optional)

Add periodic auto-snapshot:

```python
class MembrainServer:
    def __init__(self, ..., auto_snapshot_interval_s: int = 300):
        if auto_snapshot_interval_s > 0 and self.snapshot_path:
            self._start_auto_snapshot_thread(auto_snapshot_interval_s)
    
    def _start_auto_snapshot_thread(self, interval: int) -> None:
        def snapshot_loop():
            while not self._shutdown_requested:
                time.sleep(interval)
                if not self._shutdown_requested:
                    try:
                        self.save_snapshot()
                    except Exception as e:
                        logger.error("Auto-snapshot failed: %s", e)
        
        thread = threading.Thread(target=snapshot_loop, daemon=True)
        thread.start()
```

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/persistence.py` | **Create** — Snapshot logic |
| `src/membrain/core.py` | **Update** — Serialization methods |
| `src/membrain/server.py` | **Update** — Snapshot integration |
| `src/membrain/cli.py` | **Update** — CLI commands |
| `tests/test_persistence.py` | **Create** — Persistence tests |

---

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MEMBRAIN_SNAPSHOT_PATH` | str | None | Path for snapshot file |
| `MEMBRAIN_AUTO_SNAPSHOT_INTERVAL` | int | 0 | Auto-snapshot interval (0=disabled) |

---

## Tests

### Unit Tests

```python
def test_snapshot_roundtrip():
    """Snapshot save/load preserves data."""
    memory = BiCameralMemory(dimensions=128, n_neurons=100)
    
    # Add some memories
    for i in range(10):
        vec = np.random.randn(128).astype(np.float32)
        memory.remember(f"test-{i}", vec, importance=0.5)
    
    # Create and save snapshot
    snapshot = memory.create_snapshot({}, {})
    snapshot.save("/tmp/test_snapshot.gz")
    
    # Load and restore
    loaded = MemorySnapshot.load("/tmp/test_snapshot.gz")
    
    new_memory = BiCameralMemory(dimensions=128, n_neurons=100)
    new_memory.restore_snapshot(loaded)
    
    # Verify all memories present
    assert len(new_memory._entries) == 10
    for i in range(10):
        assert f"test-{i}" in new_memory._entries

def test_version_mismatch_fails():
    """Loading incompatible version raises error."""
    # Create snapshot with future version
    metadata = SnapshotMetadata(
        version="99.0.0",  # Future version
        created_at=time.time(),
        n_memories=0,
        encoder_config={},
        memory_config={},
    )
    snapshot = MemorySnapshot(metadata=metadata, entries=[], transitions={})
    
    memory = BiCameralMemory(dimensions=128, n_neurons=100)
    
    with pytest.raises(ValueError, match="Incompatible"):
        memory.restore_snapshot(snapshot)
```

### Integration Tests

```python
def test_restart_preserves_memories():
    """Server restart with snapshot preserves memories."""
    # Start server with snapshot path
    # Remember some items
    # Stop server
    # Start server again
    # Recall items - should succeed
    pass
```

---

## Acceptance Criteria

- [ ] Restart test passes: remember → snapshot → restart → restore → recall works
- [ ] Version mismatch fails fast (no silent corruption)
- [ ] CLI commands for inspect/migrate work
- [ ] Snapshot includes transitions (from PR-010)
- [ ] Auto-snapshot optional but functional

---

## Risks / Notes

- **Snapshot size**: Large memories = large snapshots (compression helps)
- **Corruption risk**: Write to temp file, then atomic rename
- **Migration complexity**: Keep v1 simple, add migration in v2
- **Lock contention**: Snapshot should be quick or use copy-on-write

---

## Definition of Done

- [ ] Tests for roundtrip persistence
- [ ] Version check prevents silent corruption
- [ ] README documents persistence options
- [ ] Logging for snapshot operations
- [ ] Auto-snapshot with configurable interval
