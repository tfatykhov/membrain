# PR-011 — Persistence (Snapshots + Versioning)

## Status: 🔴 Not Started — P1 Priority

## Problem

All memories lost on restart. No persistence mechanism.

---

## Objective

Persist memory across restarts via snapshots with version governance.

---

## Snapshot Format

Create `src/membrain/persistence.py`:

```python
SNAPSHOT_VERSION = "1.0.0"

@dataclass
class SnapshotMetadata:
    version: str
    created_at: float
    n_memories: int
    encoder_config: dict
    memory_config: dict

@dataclass
class MemorySnapshot:
    metadata: SnapshotMetadata
    entries: list[dict]  # Serialized MemoryEntry objects
    transitions: dict    # From PR-010
    
    def save(self, path: Path) -> None:
        """Save compressed snapshot to file."""
        data = gzip.compress(json.dumps(self.to_dict()).encode())
        path.write_bytes(data)
    
    @classmethod
    def load(cls, path: Path) -> "MemorySnapshot":
        """Load snapshot from file."""
        data = json.loads(gzip.decompress(path.read_bytes()))
        return cls.from_dict(data)
```

---

## BiCameralMemory Integration

```python
class BiCameralMemory:
    def create_snapshot(self, config: dict) -> MemorySnapshot:
        """Create snapshot of current state."""
        entries = [e.to_dict() for e in self._entries.values()]
        return MemorySnapshot(
            metadata=SnapshotMetadata(
                version=SNAPSHOT_VERSION,
                created_at=time.time(),
                n_memories=len(entries),
                encoder_config=config,
            ),
            entries=entries,
            transitions=self.transitions.to_dict() if hasattr(self, 'transitions') else {},
        )
    
    def restore_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Restore from snapshot."""
        if not self._is_compatible(snapshot.metadata.version):
            raise ValueError(f"Incompatible version: {snapshot.metadata.version}")
        
        self._entries.clear()
        for entry_data in snapshot.entries:
            entry = MemoryEntry.from_dict(entry_data)
            self._entries[entry.context_id] = entry
```

---

## Server Integration

```python
class MembrainServer:
    def __init__(self, ..., snapshot_path: str = None):
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        
        if self.snapshot_path and self.snapshot_path.exists():
            self._restore_from_snapshot()
    
    def save_snapshot(self) -> None:
        snapshot = self.servicer.memory.create_snapshot(...)
        snapshot.save(self.snapshot_path)
```

---

## CLI Commands

```bash
membrain serve --snapshot /data/memory.snapshot
membrain inspect /data/memory.snapshot
membrain migrate old.snapshot new.snapshot
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MEMBRAIN_SNAPSHOT_PATH` | Path for snapshot file |
| `MEMBRAIN_AUTO_SNAPSHOT_INTERVAL` | Auto-snapshot interval (0=disabled) |

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/persistence.py` | **Create** |
| `src/membrain/core.py` | **Update** |
| `src/membrain/server.py` | **Update** |
| `src/membrain/cli.py` | **Update** |
| `tests/test_persistence.py` | **Create** |

---

## Tests

```python
def test_snapshot_roundtrip():
    """Save/load preserves all data."""
    memory = BiCameralMemory(...)
    # Add memories
    snapshot = memory.create_snapshot({})
    snapshot.save("/tmp/test.snapshot")
    
    loaded = MemorySnapshot.load("/tmp/test.snapshot")
    new_memory = BiCameralMemory(...)
    new_memory.restore_snapshot(loaded)
    
    assert len(new_memory._entries) == len(memory._entries)

def test_version_mismatch_fails():
    """Incompatible version raises error."""
    # Create snapshot with future version "99.0.0"
    # Attempt restore → should raise ValueError
```

---

## Acceptance Criteria

- [ ] remember → snapshot → restart → restore → recall works
- [ ] Version mismatch fails fast
- [ ] CLI commands for inspect/migrate
- [ ] Auto-snapshot with configurable interval
