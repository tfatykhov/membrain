# BiCameralMemory

`src/membrain/core.py`

## Purpose
The **BiCameralMemory** is the neuromorphic core of the system. It implements a hippocampus-inspired Spiking Neural Network (SNN) using the **Nengo** framework. Its primary role is to store sparse binary patterns via synaptic plasticity and retrieve them via pattern completion dynamics.

## Design Rationale
- **Spiking Dynamics:** Uses Leaky Integrate-and-Fire (LIF) or similar neuron models (specifically `nengo.SpikingRectifiedLinear` in the current implementation) to leverage temporal dynamics.
- **Voja Learning Rule:** The Vector Oja (Voja) rule is chosen for its ability to shift neuron tuning curves towards active inputs, effectively allocating neural resources to represent incoming data distributions online.
- **Attractor Dynamics:** By training recurrent or feedforward weights, the network forms "attractor basins." When a noisy or partial version of a memory is presented, the system's state evolves towards the original stored pattern.

## Class: `BiCameralMemory`

### Initialization
```python
def __init__(
    self,
    n_neurons: int = 1000,
    dimensions: int = 20000,
    learning_rate: float = 1e-2,
    synapse: float = 0.01,
    dt: float = 0.001,
    seed: int | None = None,
    use_attractor: bool = False,
    attractor_learning_rate: float = 0.3,
    attractor_max_steps: int = 50,
)
```
- **n_neurons:** Size of the neural ensemble. More neurons allow for higher capacity and resolution.
- **dimensions:** Dimensionality of the input sparse vector (output from FlyHash).
- **learning_rate:** Controls how quickly weights adapt to new patterns (Voja rate).
- **synapse:** Post-synaptic time constant (filter) in seconds.
- **dt:** Simulation timestep (default 1ms).
- **seed:** Random seed for reproducibility.
- **use_attractor:** Enable explicit attractor dynamics for pattern cleanup.
- **attractor_learning_rate:** Hebbian learning rate for the attractor network.
- **attractor_max_steps:** Maximum iterations for attractor settling.

### Key Components (Nengo Network)
1.  **Input Node:** Feeds the sparse vector into the system.
2.  **Learning Gate:** A control node that modulates the learning rule.
    - `0.0` = Learning Enabled.
    - `-1.0` = Learning Disabled (for recall).
3.  **Memory Ensemble:** The population of neurons representing the memories.
4.  **Learning Connection:** Connects input to memory with the `Voja` learning rule.
5.  **Attractor Memory:** (Optional) A separate Hebbian network that cleans up patterns before/after SNN processing.

### Methods

#### `remember`
```python
def remember(
    self,
    context_id: str,
    sparse_vector: NDArray[np.floating],
    importance: float = 1.0,
    duration_ms: int = 50
) -> bool
```
Stores a pattern.
1.  **Gating:** Sets learning gate to `0.0` (enabled).
2.  **Scaling:** Scales the `sparse_vector` by `importance`. Higher importance leads to stronger weight updates.
3.  **Attractor Storage:** If enabled, also stores the pattern in the `AttractorMemory` using Hebbian learning (outer product rule), modulated by `importance`.
4.  **Simulation:** Runs the simulator for `duration_ms`. The Voja rule moves neuron encoders towards the input vector.
5.  **Indexing:** Stores the vector in `_memory_index` for later readout/verification.

#### `recall`
```python
def recall(
    self,
    query_vector: NDArray[np.floating],
    threshold: float = 0.7,
    max_results: int = 5,
    duration_ms: int = 20,
    bypass_snn: bool = False
) -> list[RecallResult]
```
Retrieves memories.
1.  **Attractor Cleanup:** If `use_attractor` is True, runs the query through the attractor network to clean up noise before SNN processing.
2.  **Gating:** Sets learning gate to `-1.0` (disabled) to prevent overwriting memories with the query.
3.  **Simulation:** Runs the simulator. The network state evolves based on the input query and stored weights. (Skipped if `bypass_snn=True`).
4.  **Readout:** Extracts the final "attractor state" from the output probe.
5.  **Matching:** Computes cosine similarity between the attractor state and all vectors in `_memory_index`.
6.  **Filtering:** Returns entries above `threshold`.

#### `consolidate`
```python
def consolidate(
    self,
    noise_scale: float = 0.05,
    max_steps: int = 50,
    convergence_threshold: float = 1e-4,
    prune_weak: bool = False,
    prune_threshold: float = 0.1
) -> tuple[int, int]
```
Performs stochastic consolidation (attractor dynamics).
- **Noise Injection:** Injects Gaussian white noise (`noise_scale`) into the network state to shake it out of local minima.
- **Attractor Settling:** Iterates the network dynamics until the state difference between steps falls below `convergence_threshold` or `max_steps` is reached.
- **Pruning:** Optionally removes memories from the index if their `importance` is below `prune_threshold`.
- **Returns:** A tuple `(steps_to_converge, pruned_count)`. If `steps_to_converge` is -1, it means the network did not settle within `max_steps`.

## Usage Example

```python
from membrain.core import BiCameralMemory
import numpy as np

# Initialize
memory = BiCameralMemory(n_neurons=1000, dimensions=20000)

# Create a dummy sparse vector (usually comes from FlyHash)
vector = np.zeros(20000)
vector[[1, 100, 500]] = 1.0 

# Store
with memory:
    memory.remember("memory-01", vector, importance=0.9)

    # Recall
    results = memory.recall(vector)
    print(results) 
    # Output: [RecallResult(context_id='memory-01', confidence=1.0)]
```

## Internal Implementation Details
- **Memory Index:** While the SNN "learns" the patterns in its weights, a Python dictionary `_memory_index` is currently used to map the network's output back to specific `context_id`s. The SNN acts as a highly non-linear filter/denoiser, and the dictionary performs the final classification.
- **Simulator Management:** The `nengo.Simulator` is heavy. It is initialized lazily via `_ensure_simulator()` and kept alive. The `reset()` method or context manager should be used to free resources.

## Known Limitations / TODOs
- **Scalability (Attractor Memory):** The optional `AttractorMemory` uses an $O(N^2)$ weight matrix. For default 20,000 dimensions, this requires ~1.6 GB RAM. Use with caution or reduce dimensions.
- **Scalability (Recall):** The linear scan in `recall` (comparing against all stored vectors in `_memory_index`) is O(N). For very large memory stores, an approximate nearest neighbor index (like FAISS) should replace the dictionary loop.
- **Hardware Acceleration:** Currently runs on CPU. Nengo supports GPU (nengo-ocl) and Neuromorphic hardware (nengo-loihi), which could be enabled via configuration.
- **Persistence:** The `_memory_index` and the Nengo simulator state are in-memory only. They are lost on restart. Serialization (pickling the simulator/index) is needed for durability.
