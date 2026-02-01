# Feature 04: Lava Process Integration

**Status:** Not Started  
**Priority:** P1 - Future-Proofing  
**Target File:** `src/membrain/lava_process.py`  
**Depends On:** Feature 03 (Neuromorphic Core)  
**Required By:** Hardware Migration

---

## Objective

Wrap the Nengo-based memory system in an Intel **Lava** Process to ensure future-proof migration to physical Loihi 2 hardware. This enforces the Communicating Sequential Processes (CSP) pattern and enables seamless backend swapping.

---

## Why Lava?

| Aspect | Pure Nengo | Nengo + Lava Wrapper |
|--------|------------|---------------------|
| **CPU Simulation** | ✅ Works | ✅ Works |
| **Loihi Hardware** | ⚠️ Limited | ✅ Full support |
| **Backend Swap** | ❌ Code changes | ✅ Config change only |
| **Multi-Process** | ❌ Single thread | ✅ CSP parallelism |
| **Production Ready** | ⚠️ Research | ✅ Intel-supported |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HippocampusProcess                        │
│                  (lava.magma.core.process)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                    ┌──────────────────┐   │
│  │   InPort     │───────────────────►│   BiCameralMemory │   │
│  │  (vectors)   │                    │   (Nengo Core)    │   │
│  └──────────────┘                    └────────┬─────────┘   │
│                                               │              │
│  ┌──────────────┐◄───────────────────────────┘              │
│  │   OutPort    │                                            │
│  │ (context_ids)│                                            │
│  └──────────────┘                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Specification

### Class: HippocampusProcess

```python
# src/membrain/lava_process.py

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU, Loihi2NeuroCore
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

import numpy as np
from typing import Optional
from enum import Enum

from membrain.core import BiCameralMemory
from membrain.encoder import FlyHash


class MemoryOperation(Enum):
    """Memory operation types."""
    IDLE = 0
    REMEMBER = 1
    RECALL = 2
    CONSOLIDATE = 3


class HippocampusProcess(AbstractProcess):
    """
    Lava Process wrapping the BiCameralMemory SNN.
    
    Provides a CSP-compatible interface for neuromorphic memory
    operations, enabling deployment on both CPU and Loihi hardware.
    """
    
    def __init__(
        self,
        dimensions: int = 20000,
        n_neurons: int = 1000,
        **kwargs
    ):
        """
        Initialize the HippocampusProcess.
        
        Args:
            dimensions: Sparse vector dimension (FlyHash output)
            n_neurons: Number of neurons in memory ensemble
        """
        super().__init__(**kwargs)
        
        # Process configuration
        self.dimensions = Var(shape=(1,), init=dimensions)
        self.n_neurons = Var(shape=(1,), init=n_neurons)
        
        # Input port for sparse vectors
        self.vector_in = InPort(shape=(dimensions,))
        
        # Output port for recalled context IDs
        # (encoded as integer array for CSP compatibility)
        self.ids_out = OutPort(shape=(10,))  # Max 10 results
        self.confidences_out = OutPort(shape=(10,))
        
        # Operation control
        self.operation = Var(shape=(1,), init=MemoryOperation.IDLE.value)
        self.context_id = Var(shape=(64,), init=0)  # UUID as bytes
        self.importance = Var(shape=(1,), init=1.0)
        self.threshold = Var(shape=(1,), init=0.7)
        
        # Status
        self.ready = Var(shape=(1,), init=False)
        self.success = Var(shape=(1,), init=False)


@implements(proc=HippocampusProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyHippocampusProcessModel(PyLoihiProcessModel):
    """
    Python process model for HippocampusProcess.
    
    This model runs on CPU and wraps the Nengo BiCameralMemory.
    For Loihi deployment, a separate C process model would be used.
    """
    
    # Port types
    vector_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    ids_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    confidences_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    
    # Variable types
    dimensions: np.ndarray = LavaPyType(np.ndarray, int)
    n_neurons: np.ndarray = LavaPyType(np.ndarray, int)
    operation: np.ndarray = LavaPyType(np.ndarray, int)
    context_id: np.ndarray = LavaPyType(np.ndarray, int)
    importance: np.ndarray = LavaPyType(np.ndarray, float)
    threshold: np.ndarray = LavaPyType(np.ndarray, float)
    ready: np.ndarray = LavaPyType(np.ndarray, bool)
    success: np.ndarray = LavaPyType(np.ndarray, bool)
    
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._memory: Optional[BiCameralMemory] = None
        self._encoder: Optional[FlyHash] = None
    
    def _initialize(self):
        """Initialize the memory system on first run."""
        if self._memory is None:
            dims = int(self.dimensions[0])
            neurons = int(self.n_neurons[0])
            
            self._memory = BiCameralMemory(
                n_neurons=neurons,
                dimensions=dims
            )
            self._memory.build_simulator()
            self.ready[0] = True
    
    def run_spk(self):
        """
        Execute one timestep of the process.
        
        Called by Lava runtime on each simulation step.
        """
        self._initialize()
        
        # Read input vector
        vector = self.vector_in.recv()
        
        # Determine operation
        op = MemoryOperation(self.operation[0])
        
        if op == MemoryOperation.REMEMBER:
            self._handle_remember(vector)
        elif op == MemoryOperation.RECALL:
            self._handle_recall(vector)
        elif op == MemoryOperation.CONSOLIDATE:
            self._handle_consolidate()
        else:
            # IDLE - send zeros
            self.ids_out.send(np.zeros(10, dtype=int))
            self.confidences_out.send(np.zeros(10, dtype=float))
    
    def _handle_remember(self, vector: np.ndarray):
        """Handle remember operation."""
        # Decode context_id from bytes
        context_id = bytes(self.context_id).decode('utf-8').strip('\x00')
        importance = float(self.importance[0])
        
        try:
            self._memory.remember(
                context_id=context_id,
                sparse_vector=vector,
                importance=importance
            )
            self.success[0] = True
        except Exception:
            self.success[0] = False
        
        # Send empty outputs (remember doesn't return results)
        self.ids_out.send(np.zeros(10, dtype=int))
        self.confidences_out.send(np.zeros(10, dtype=float))
    
    def _handle_recall(self, vector: np.ndarray):
        """Handle recall operation."""
        threshold = float(self.threshold[0])
        
        try:
            results = self._memory.recall(
                query_vector=vector,
                threshold=threshold,
                max_results=10
            )
            
            # Encode results for output ports
            ids = np.zeros(10, dtype=int)
            confs = np.zeros(10, dtype=float)
            
            for i, (context_id, confidence) in enumerate(results[:10]):
                # Hash context_id to int for CSP compatibility
                ids[i] = hash(context_id) % (2**31)
                confs[i] = confidence
            
            self.success[0] = True
        except Exception:
            ids = np.zeros(10, dtype=int)
            confs = np.zeros(10, dtype=float)
            self.success[0] = False
        
        self.ids_out.send(ids)
        self.confidences_out.send(confs)
    
    def _handle_consolidate(self):
        """Handle consolidate operation."""
        try:
            self._memory.consolidate()
            self.success[0] = True
        except Exception:
            self.success[0] = False
        
        self.ids_out.send(np.zeros(10, dtype=int))
        self.confidences_out.send(np.zeros(10, dtype=float))
```

---

## Lava Runtime Usage

### Running the Process

```python
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from membrain.lava_process import HippocampusProcess, MemoryOperation

# Create process
hippocampus = HippocampusProcess(
    dimensions=20000,
    n_neurons=1000
)

# Configure for CPU simulation
run_cfg = Loihi1SimCfg()

# Run one step
hippocampus.run(
    condition=RunSteps(num_steps=1),
    run_cfg=run_cfg
)

# Check status
print(f"Ready: {hippocampus.ready.get()}")

# Stop process
hippocampus.stop()
```

### Integration with gRPC Server

```python
# In server.py
class MemoryUnitServicer:
    def __init__(self):
        self.process = HippocampusProcess(
            dimensions=20000,
            n_neurons=1000
        )
        self.run_cfg = Loihi1SimCfg()
    
    def Remember(self, request, context):
        # Set operation
        self.process.operation.set(MemoryOperation.REMEMBER.value)
        
        # Set context_id
        id_bytes = request.context_id.encode('utf-8')[:64]
        self.process.context_id.set(
            np.frombuffer(id_bytes.ljust(64, b'\x00'), dtype=int)
        )
        
        # Set importance
        self.process.importance.set(np.array([request.importance]))
        
        # Run step
        self.process.run(
            condition=RunSteps(num_steps=1),
            run_cfg=self.run_cfg
        )
        
        success = bool(self.process.success.get()[0])
        return Ack(success=success)
```

---

## Hardware Migration Path

### Step 1: Current State (CPU Simulation)

```python
run_cfg = Loihi1SimCfg()  # CPU backend
```

### Step 2: Loihi Hardware Deployment

```python
from lava.magma.core.run_configs import Loihi2HwCfg

# Install nxsdk (Intel proprietary)
# Change only this line:
run_cfg = Loihi2HwCfg()  # Hardware backend
```

### Step 3: Create C Process Model (Optional)

For maximum hardware efficiency, implement `CLoihiProcessModel`:

```python
@implements(proc=HippocampusProcess, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class CLoihiHippocampusProcessModel(AbstractCProcessModel):
    """Native Loihi 2 implementation."""
    # C code would handle the actual neural computation
    pass
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LOIHI_BACKEND` | `SIM` | `SIM` for CPU, `loihi` for hardware |
| `LAVA_LOG_LEVEL` | `WARNING` | Lava logging verbosity |

---

## Acceptance Criteria

- [ ] `HippocampusProcess` inherits from `AbstractProcess`
- [ ] `InPort` and `OutPort` correctly defined
- [ ] Python process model implements `run_spk()`
- [ ] Process runs on CPU with `Loihi1SimCfg`
- [ ] All memory operations (remember/recall/consolidate) work
- [ ] Process can be started and stopped cleanly
- [ ] gRPC server integrates with Lava process

---

## Testing

### Unit Tests (`tests/test_lava_process.py`)

```python
import numpy as np
import pytest
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from membrain.lava_process import HippocampusProcess, MemoryOperation


class TestHippocampusProcess:
    @pytest.fixture
    def process(self):
        return HippocampusProcess(dimensions=1000, n_neurons=100)
    
    def test_process_initializes(self, process):
        """Process should initialize without errors."""
        assert process is not None
        assert process.vector_in is not None
        assert process.ids_out is not None
    
    def test_process_runs_on_cpu(self, process):
        """Process should run on CPU backend."""
        run_cfg = Loihi1SimCfg()
        
        process.run(
            condition=RunSteps(num_steps=1),
            run_cfg=run_cfg
        )
        
        assert process.ready.get()[0] == True
        process.stop()
    
    def test_remember_operation(self, process):
        """Remember should store a memory."""
        run_cfg = Loihi1SimCfg()
        
        # Set up remember operation
        process.operation.set(np.array([MemoryOperation.REMEMBER.value]))
        process.context_id.set(
            np.frombuffer(b"test-id".ljust(64, b'\x00'), dtype=int)
        )
        process.importance.set(np.array([1.0]))
        
        process.run(
            condition=RunSteps(num_steps=1),
            run_cfg=run_cfg
        )
        
        assert process.success.get()[0] == True
        process.stop()
    
    def test_process_cleanup(self, process):
        """Process should clean up on stop."""
        run_cfg = Loihi1SimCfg()
        
        process.run(
            condition=RunSteps(num_steps=1),
            run_cfg=run_cfg
        )
        process.stop()
        
        # No assertions needed - should not raise
```

---

## Dependencies

- `lava-nc>=0.5.0` - Intel Lava framework
- `nengo>=3.2.0` - Neural simulation (via core.py)
- `numpy>=1.24.0` - Numerical operations

---

## References

- [Intel Lava Documentation](https://lava-nc.org/)
- [Lava Process Tutorial](https://github.com/lava-nc/lava/tree/main/tutorials)
- [CSP Model](https://en.wikipedia.org/wiki/Communicating_sequential_processes)
- [Loihi 2 Architecture](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
