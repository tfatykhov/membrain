# PR-010 — Temporal Binding (A→B Predictive Recall)

## Status: 🔴 Not Started — P1 Priority

## Current State Analysis

### What Exists
- Point-in-time memory storage (Remember stores isolated patterns)
- Associative recall (find similar patterns)
- No temporal/sequential information captured

### What's Missing
- **Sequence memory**: No concept of "what comes next"
- **Temporal binding**: No A→B associations learned
- **Predictive recall**: Cannot predict likely successors

This is a key differentiator for a "synthetic hippocampus" — real hippocampus binds events in time.

---

## Objective

Add sequence memory: recalling pattern A can return pattern B as "likely next" based on observed temporal co-occurrence.

---

## Background: Temporal Association

**Biological basis**: Hippocampal CA3 region forms auto-associative memories, while CA1 binds temporal sequences. Pattern completion in CA3 + temporal prediction in CA1.

**Computational approach**:
1. **Transition matrix**: Track T[A][B] = count of A→B transitions
2. **Hebbian sequence learning**: Strengthen A→B connections when B follows A
3. **Predictive coding**: Given A, return predicted B with confidence

---

## Detailed Requirements

### A. Transition Store

Create `src/membrain/temporal.py`:

```python
"""
Temporal binding module for sequence memory.

Tracks transitions between context_ids and enables predictive recall.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import numpy as np
import time

@dataclass
class Transition:
    """A recorded transition from one context to another."""
    from_id: str
    to_id: str
    count: int
    last_seen: float  # Timestamp
    strength: float  # Normalized transition probability

@dataclass
class PredictionResult:
    """Result of a predictive recall."""
    context_id: str
    probability: float
    evidence_count: int

class TransitionStore:
    """
    Tracks temporal transitions between context_ids.
    
    Learns A→B associations when B is stored shortly after A is recalled.
    """
    
    def __init__(
        self,
        decay_rate: float = 0.99,
        recency_window_ms: float = 5000.0,
    ):
        """
        Args:
            decay_rate: Exponential decay for old transitions (0-1).
            recency_window_ms: Time window for associating recall→store.
        """
        self.decay_rate = decay_rate
        self.recency_window_ms = recency_window_ms
        
        # T[from_id][to_id] = count
        self._transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Track recent recalls for temporal binding
        self._recent_recalls: list[tuple[str, float]] = []  # (context_id, timestamp_ms)
        
        # Total outgoing count per context (for normalization)
        self._outgoing_total: dict[str, int] = defaultdict(int)
    
    def record_recall(self, context_id: str) -> None:
        """Record that a context was just recalled."""
        now = time.time() * 1000
        self._recent_recalls.append((context_id, now))
        
        # Prune old recalls
        cutoff = now - self.recency_window_ms * 2
        self._recent_recalls = [
            (cid, ts) for cid, ts in self._recent_recalls
            if ts > cutoff
        ]
    
    def record_store(self, context_id: str) -> None:
        """
        Record that a context was just stored.
        
        Creates transitions from recently recalled contexts to this one.
        """
        now = time.time() * 1000
        cutoff = now - self.recency_window_ms
        
        for from_id, ts in self._recent_recalls:
            if ts > cutoff and from_id != context_id:
                self._transitions[from_id][context_id] += 1
                self._outgoing_total[from_id] += 1
    
    def predict_next(
        self,
        context_id: str,
        max_results: int = 5,
        min_probability: float = 0.01,
    ) -> list[PredictionResult]:
        """
        Predict likely successor contexts.
        
        Args:
            context_id: The context to predict from.
            max_results: Maximum number of predictions.
            min_probability: Minimum probability threshold.
        
        Returns:
            List of predictions sorted by probability.
        """
        if context_id not in self._transitions:
            return []
        
        total = self._outgoing_total[context_id]
        if total == 0:
            return []
        
        predictions = []
        for to_id, count in self._transitions[context_id].items():
            prob = count / total
            if prob >= min_probability:
                predictions.append(PredictionResult(
                    context_id=to_id,
                    probability=prob,
                    evidence_count=count,
                ))
        
        # Sort by probability descending
        predictions.sort(key=lambda p: p.probability, reverse=True)
        return predictions[:max_results]
    
    def decay_all(self) -> int:
        """
        Apply decay to all transition counts.
        
        Returns number of transitions pruned (count went to 0).
        """
        pruned = 0
        for from_id in list(self._transitions.keys()):
            for to_id in list(self._transitions[from_id].keys()):
                self._transitions[from_id][to_id] = int(
                    self._transitions[from_id][to_id] * self.decay_rate
                )
                if self._transitions[from_id][to_id] == 0:
                    del self._transitions[from_id][to_id]
                    pruned += 1
            
            if not self._transitions[from_id]:
                del self._transitions[from_id]
        
        # Recalculate totals
        self._outgoing_total.clear()
        for from_id, targets in self._transitions.items():
            self._outgoing_total[from_id] = sum(targets.values())
        
        return pruned
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "transitions": {k: dict(v) for k, v in self._transitions.items()},
            "outgoing_total": dict(self._outgoing_total),
        }
    
    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "TransitionStore":
        """Deserialize from persistence."""
        store = cls(**kwargs)
        store._transitions = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in data.get("transitions", {}).items()}
        )
        store._outgoing_total = defaultdict(int, data.get("outgoing_total", {}))
        return store
```

### B. Proto Extension

Update `protos/memory_a2a.proto`:

```protobuf
// Add to QueryPacket or create new message
message PredictRequest {
    string context_id = 1;
    int32 max_results = 2;
}

message PredictionResponse {
    repeated Prediction predictions = 1;
}

message Prediction {
    string context_id = 1;
    float probability = 2;
    int32 evidence_count = 3;
}

// Add RPC to MemoryUnit service
service MemoryUnit {
    // ... existing RPCs ...
    rpc PredictNext(PredictRequest) returns (PredictionResponse);
}
```

### C. Server Integration

Update `server.py`:

```python
class MemoryUnitServicer:
    def __init__(self, ...):
        # ... existing init ...
        self.transitions = TransitionStore(
            recency_window_ms=5000.0,
        )
    
    def Remember(self, request, context):
        # ... existing logic ...
        
        # Record for temporal binding
        self.transitions.record_store(request.context_id)
        
        return response
    
    def Recall(self, request, context):
        # ... existing logic ...
        
        # Record recalls for temporal binding
        for result in results:
            self.transitions.record_recall(result.context_id)
        
        return response
    
    def PredictNext(self, request, context):
        """Predict likely successor contexts."""
        predictions = self.transitions.predict_next(
            context_id=request.context_id,
            max_results=request.max_results or 5,
        )
        
        return memory_a2a_pb2.PredictionResponse(
            predictions=[
                memory_a2a_pb2.Prediction(
                    context_id=p.context_id,
                    probability=p.probability,
                    evidence_count=p.evidence_count,
                )
                for p in predictions
            ]
        )
```

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/temporal.py` | **Create** — Transition store |
| `protos/memory_a2a.proto` | **Update** — Add PredictNext RPC |
| `src/membrain/proto/*` | **Regenerate** — Updated stubs |
| `src/membrain/server.py` | **Update** — Integrate temporal binding |
| `tests/test_temporal.py` | **Create** — Temporal binding tests |

---

## Tests

### Unit Tests

```python
def test_transition_learning():
    """Transitions are learned from recall→store sequences."""
    store = TransitionStore(recency_window_ms=5000)
    
    # Simulate: recall A, then store B (within window)
    store.record_recall("A")
    time.sleep(0.1)
    store.record_store("B")
    
    predictions = store.predict_next("A")
    assert len(predictions) == 1
    assert predictions[0].context_id == "B"
    assert predictions[0].evidence_count == 1

def test_transition_strengthening():
    """Repeated transitions increase probability."""
    store = TransitionStore(recency_window_ms=5000)
    
    # Repeat A→B 10 times
    for _ in range(10):
        store.record_recall("A")
        store.record_store("B")
    
    # Add A→C once
    store.record_recall("A")
    store.record_store("C")
    
    predictions = store.predict_next("A")
    assert predictions[0].context_id == "B"
    assert predictions[0].probability > predictions[1].probability

def test_decay_reduces_old_transitions():
    """Decay reduces old transition counts."""
    store = TransitionStore(decay_rate=0.5)
    
    store.record_recall("A")
    store.record_store("B")
    
    initial = store.predict_next("A")[0].evidence_count
    store.decay_all()
    after = store.predict_next("A")[0].evidence_count
    
    assert after < initial
```

### Integration Tests

```python
def test_predict_next_grpc():
    """PredictNext RPC returns learned successors."""
    # Start server, make Remember/Recall calls in sequence
    # Then call PredictNext and verify predictions
    pass
```

---

## Acceptance Criteria

- [ ] After repeated A→B episodes, recalling A predicts B
- [ ] Predictions sorted by probability
- [ ] Transition decay prevents unbounded growth
- [ ] gRPC PredictNext RPC works
- [ ] Serialization/deserialization for persistence (PR-011 prep)

---

## Risks / Notes

- **Timing sensitivity**: recency_window_ms needs tuning
- **Spurious transitions**: May learn noise if window too wide
- **Scale**: Transition matrix grows with context count — may need pruning
- **Ordering**: Proto regeneration required before server work

---

## Future Enhancements

1. **Multi-step sequences**: A→B→C prediction
2. **Temporal context**: Weight by recency of evidence
3. **Neural temporal binding**: Replace counting with SNN plasticity

---

## Definition of Done

- [ ] Tests prove A→B prediction after repeated episodes
- [ ] gRPC PredictNext RPC implemented
- [ ] Transitions serializable for persistence
- [ ] Decay mechanism prevents unbounded growth
- [ ] Documentation updated
