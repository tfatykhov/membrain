# PR-010 — Temporal Binding (Sequence Memory)

## Status: 🔴 Not Started — P1 Priority

## Problem

Current implementation stores isolated patterns with no temporal relationships. Real hippocampus binds events in time.

---

## Objective

Add sequence memory: recalling pattern A can return pattern B as "likely next."

---

## Approach: Transition Store

Create `src/membrain/temporal.py`:

```python
class TransitionStore:
    """Tracks temporal transitions between context_ids."""
    
    def __init__(self, recency_window_ms: float = 5000.0):
        self._transitions = defaultdict(lambda: defaultdict(int))
        self._recent_recalls = []
    
    def record_recall(self, context_id: str) -> None:
        """Record that a context was recalled."""
        self._recent_recalls.append((context_id, time.time()))
    
    def record_store(self, context_id: str) -> None:
        """Record store, creating transitions from recent recalls."""
        for from_id, ts in self._recent_recalls:
            if within_window(ts) and from_id != context_id:
                self._transitions[from_id][context_id] += 1
    
    def predict_next(self, context_id: str, max_results: int = 5) -> list:
        """Predict likely successor contexts."""
        # Return sorted by probability
        pass
```

---

## Proto Extension

```protobuf
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

service MemoryUnit {
    rpc PredictNext(PredictRequest) returns (PredictionResponse);
}
```

---

## Server Integration

```python
def Remember(self, request, context):
    # ... existing logic ...
    self.transitions.record_store(request.context_id)

def Recall(self, request, context):
    # ... existing logic ...
    for result in results:
        self.transitions.record_recall(result.context_id)

def PredictNext(self, request, context):
    predictions = self.transitions.predict_next(request.context_id)
    return PredictionResponse(predictions=predictions)
```

---

## Tests

```python
def test_transition_learning():
    """Transitions are learned from recall→store sequences."""
    store = TransitionStore()
    store.record_recall("A")
    store.record_store("B")
    
    predictions = store.predict_next("A")
    assert predictions[0].context_id == "B"

def test_transition_strengthening():
    """Repeated transitions increase probability."""
    store = TransitionStore()
    for _ in range(10):
        store.record_recall("A")
        store.record_store("B")
    store.record_recall("A")
    store.record_store("C")
    
    predictions = store.predict_next("A")
    assert predictions[0].context_id == "B"  # More evidence
```

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/temporal.py` | **Create** |
| `protos/memory_a2a.proto` | **Update** |
| `src/membrain/proto/*` | **Regenerate** |
| `src/membrain/server.py` | **Update** |
| `tests/test_temporal.py` | **Create** |

---

## Acceptance Criteria

- [ ] After repeated A→B episodes, recalling A predicts B
- [ ] gRPC PredictNext RPC implemented
- [ ] Transitions serializable for persistence
