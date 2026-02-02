# FlyHash Encoder

`src/membrain/encoder.py`

## Purpose
**FlyHash** provides a bio-inspired mechanism to convert dense, continuous embeddings (like those from LLMs) into sparse, binary representations suitable for Spiking Neural Networks.

## Design Rationale
- **Locality Sensitivity:** Similar inputs should produce similar outputs (overlapping active bits). This is critical for associative memory.
- **Sparsity:** SNNs are energy and compute-efficient when processing sparse data.
- **Dimensionality Expansion:** Projecting to a higher dimension allows patterns to be more easily separated (linear separability theorem).
- **Fruit Fly Inspiration:** Based on the *Drosophila* olfactory circuit, where ~50 projection neurons expand to ~2000 Kenyon cells, followed by strong Winner-Take-All inhibition.

## Class: `FlyHash`

### Initialization
```python
def __init__(
    self,
    input_dim: int = 1536,
    expansion_ratio: float = 13.0,
    active_bits: int = 50,
    seed: int | None = None
)
```
- **input_dim:** Dimension of the source vector (e.g., 1536 for OpenAI `text-embedding-ada-002`).
- **expansion_ratio:** Multiplier for output dimension. `output_dim = input_dim * expansion_ratio`.
- **active_bits (k):** The exact number of neurons that will fire (be set to 1) in the output. This enforces a strict k-hot code.
- **seed:** Random seed for deterministic projection matrix generation.

### Algorithm Steps (`encode`)

1.  **Random Projection:**
    The input vector $x$ is multiplied by a dense, int8 projection matrix $M$.
    $$ y = M^T x $$
    - $M$ has shape `(input_dim, output_dim)`.
    - Entries in $M$ are $\{-1, +1\}$, stored as `int8` for memory efficiency.

2.  **Winner-Take-All (WTA) Inhibition:**
    To enforce sparsity, only the top-$k$ neurons with the highest activation in $y$ are kept active.
    - Identify the indices of the $k$ largest values in $y$ (where $k=$ `active_bits`).
    - Set these indices to $1.0$.
    - Set all other indices to $0.0$.

### Methods

#### `encode`
```python
def encode(self, vector: NDArray[np.floating]) -> NDArray[np.float32]
```
Encodes a single 1-D vector.
- **Input:** Dense float vector `(input_dim,)`.
- **Output:** Sparse binary float vector `(output_dim,)`.

#### `encode_batch`
```python
def encode_batch(self, vectors: NDArray[np.floating]) -> NDArray[np.float32]
```
Vectorized version for processing multiple inputs efficiently.

#### Similarity Metrics
- `hamming_distance(h1, h2)`: Count of differing bits.
- `jaccard_similarity(h1, h2)`: Intersection over Union (IoU) of active bits.

## Usage Example

```python
from membrain.encoder import FlyHash
import numpy as np

# 1. Setup
encoder = FlyHash(
    input_dim=1536,
    expansion_ratio=10.0, # Output will be 15360
    active_bits=50
)

# 2. Input (fake embedding)
dense_vec = np.random.randn(1536)

# 3. Encode
sparse_code = encoder.encode(dense_vec)

print(sparse_code.shape) # (15360,)
print(np.sum(sparse_code)) # 50.0
```

## Implementation Details
- **Optimization:** Uses `np.argpartition` instead of full sort to find top-k indices, reducing complexity from $O(N \log N)$ to $O(N)$.
- **Determinism:** The projection matrix is generated once at initialization based on the seed. It must remain constant for the lifetime of the system to ensure consistent encoding.

## Known Limitations / TODOs
- **Memory Usage:** The projection matrix is stored in memory. Using `int8` {-1, +1} projection significantly reduces footprint (e.g., ~30 MB for standard config vs ~245 MB previously).
