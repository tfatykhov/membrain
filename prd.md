# Product Requirements Document (PRD): Neuromorphic Memory Bridge (PoC)

**Version:** 1.0

**Status:** Draft

**Author:** Senior Neuromorphic Engineer

**Target Hardware:** Standard CPU (Simulating Intel Loihi 2)

**Frameworks:** Intel Lava, Nengo, gRPC (A2A Protocol)

## 1. Executive Summary

The **Neuromorphic Memory Bridge** is a specialized sub-agent designed to offload long-term context retention from Large Language Models (LLMs). Unlike vector databases (RAG), which perform static retrieval, this system uses a Spiking Neural Network (SNN) to form dynamic, plastic associations between concepts.

This Proof of Concept (PoC) will simulate the behavior of the Intel Loihi neuromorphic chip using standard CPU hardware. It will expose a standardized **Agent-to-Agent (A2A)** interface via gRPC, allowing any LLM agent to "plug in" a synthetic hippocampus for associative recall and continuous learning.

## 2. System Architecture

The system follows a **Microservice Architecture** encapsulated in a Docker container.

* **Frontend (The Interface):** A gRPC Server implementing the A2A Memory Protocol.
* **Middleware (The Encoder):** A "Dense-to-Sparse" encoder implementing the **FlyHash** algorithm.
* **Backend (The Brain):** A **Nengo** simulation running on the nengo-loihi emulator backend.
* **Orchestrator:** A **Lava** Process that manages the lifecycle of the Nengo simulation (Run, Pause, Reset).

### 2.1 Data Flow

1. **Input:** Main Agent sends a 1536-d float vector (OpenAI Embedding).
2. **Transduction:** Middleware converts this to a 20,000-d binary sparse spike train (FlyHash).
3. **Ingestion:** Spikes are injected into the Nengo Network.
4. **Dynamics:**
   * *Write:* Synaptic weights are updated via Voja (learning rule) and STDP.
   * *Read:* The network settles into an attractor state (Associative Recall).
5. **Output:** The active neuron indices are mapped back to a Context ID (UUID) and returned to the Agent.

---

## 3. Functional Requirements & Implementation Steps

### Feature 1: The A2A Interface (gRPC)

**Objective:** Define a strict contract for Agent-to-Memory communication.

**Step 1.1: Define Protocol Buffers (memory_a2a.proto)**

We will define a service with two core methods: Remember (Write) and Recall (Read).

```protobuf
syntax = "proto3";

package memory_bridge;

service MemoryUnit {
  // Agent sends a context vector to be stored
  rpc Remember (MemoryPacket) returns (Ack) {}

  // Agent sends a query vector to trigger association
  rpc Recall (QueryPacket) returns (ContextResponse) {}

  // Trigger sleep phase for consolidation
  rpc Consolidate (SleepSignal) returns (Ack) {}
}

message MemoryPacket {
  string context_id = 1;      // UUID of the text chunk
  repeated float vector = 2;  // Dense embedding (e.g., Ada-002)
  float importance = 3;       // 0.0 to 1.0 (Modulates learning rate)
}

message QueryPacket {
  repeated float vector = 1;
  float threshold = 2;        // Similarity threshold
}

message ContextResponse {
  repeated string context_ids = 1; // IDs of recalled memories
  float confidence = 2;
}

message SleepSignal {
  int32 duration_ms = 1;
}

message Ack {
  bool success = 1;
  string message = 2;
}
```

**Step 1.2: Generate Python Stubs**

Run the compiler to generate the Python interface code:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. memory_a2a.proto
```

---

### Feature 2: The Encoder (FlyHash)

**Objective:** Convert dense LLM embeddings into sparse binary spikes suitable for the SNN.

**Step 2.1: Implement FlyHash Class**

This utilizes a random projection matrix followed by a Winner-Take-All (WTA) inhibition.

* **Inputs:** Dense Vector `x` (Dimension `d`).
* **Parameters:** Projection Ratio (`r`), Active Bits (`k`).
* **Logic:**
  1. Initialize a static random binary matrix `M` of shape `(d, d*r)`.
  2. Compute `y = M^T * x`.
  3. Identify indices of the top `k` values in `y`.
  4. Return a binary vector where top `k` indices are `1`, others `0`.

*Why this matters:* SNNs on Loihi are energy efficient only if activations are sparse. FlyHash guarantees fixed sparsity.

---

### Feature 3: The Neuromorphic Core (Nengo-Loihi)

**Objective:** Build the SNN that stores and retrieves patterns. We use **Nengo** for its high-level "Associative Memory" primitives, running on the **NengoLoihi** emulator to enforce hardware constraints (integer weights, discretized spikes).

**Step 3.1: Define the Nengo Network**

We will use `nengo.networks.AssociativeMemory`.

```python
import nengo
import nengo_loihi
import numpy as np

class BiCameralMemory:
    def __init__(self, n_neurons=1000, dimensions=1536):
        self.model = nengo.Network(label="Hippocampus")

        with self.model:
            # 1. Input Node: Receives spike trains from FlyHash
            self.input_node = nengo.Node(np.zeros(dimensions))

            # 2. Memory Ensemble: The core storage
            # We use Voja (Vector Oja) learning for building associations dynamically
            # and PES (Prescribed Error Sensitivity) for error correction.
            self.memory = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=dimensions,
                neuron_type=nengo.LoihiSpikingRectifiedLinear() # Hardware-aware neuron
            )

            # 3. Learning Connection (Plasticity)
            # This connection learns to map Input -> Memory State
            conn = nengo.Connection(
                self.input_node,
                self.memory,
                learning_rule_type=nengo.Voja(learning_rate=1e-2),
                synapse=0.01
            )

            # 4. Output Probe
            self.output_probe = nengo.Probe(self.memory, synapse=0.01)

    def build_sim(self):
        # target='sim' creates a CPU simulation that respects Loihi constraints
        return nengo_loihi.Simulator(self.model, target='sim', dt=0.001)
```

**Step 3.2: Implement the Recall Logic**

When the gRPC Recall is hit:

1. Pause the simulation.
2. Inject the Query Spike Train into `input_node`.
3. Run simulation for 20 timesteps (20ms).
4. Read `output_probe`.
5. If the network settles (attractor state), decoding the active neurons identifies the stored Memory ID.

---

### Feature 4: Integration with Lava (Process Management)

**Objective:** Ensure the system is future-proof for pure Loihi deployment.

**Step 4.1: Wrap Nengo in a Lava Process**

While Nengo handles the math, we wrap the entire simulation controller in a **Lava Process**. This strictly enforcing the *Communicating Sequential Processes (CSP)* pattern.

* Create a class `HippocampusProcess` inheriting from `lava.magma.core.process.process.AbstractProcess`.
* Define `InPort` (for vectors) and `OutPort` (for IDs).
* This wrapper doesn't change the logic now, but it means later we can swap the Python backend for a C backend running on the Loihi chip without changing the A2A API.

---

## 4. Development Workflow

### Phase 1: Environment Setup

1. **Docker Base:** Use `python:3.9-slim`.
2. **Dependencies:**

```text
nengo==3.2.0
nengo-loihi==1.1.0
lava-nc==0.5.0
grpcio==1.50.0
numpy==1.24.0
```

### Phase 2: The Encoder Test

* **Action:** Create a unit test `test_flyhash.py`.
* **Input:** Two semantically similar vectors (e.g., "King", "Queen") and one different ("Apple").
* **Verify:** The Hamming distance between the FlyHash binaries of "King" and "Queen" must be significantly lower than "King" and "Apple".

### Phase 3: The Associative Memory Loop

* **Action:** Write the main `server.py` loop.
* **Logic:**
  * Initialize `BiCameralMemory`.
  * Start gRPC server.
  * On `Remember(A)`: Run SNN for 50ms with Input A + Plasticity ON.
  * On `Recall(A')` (noisy A): Run SNN for 20ms with Input A' + Plasticity OFF.
  * **Verify:** Does the SNN output vector match the original A vector?

### Phase 4: Containerization

* **Action:** Build the Dockerfile.
* **Constraint:** Ensure nengo-loihi is configured to use the CPU backend (`target='sim'`) via an environment variable `LOIHI_BACKEND=SIM`.

## 5. Success Metrics (The "Proof")

Since we lack the hardware to prove energy efficiency directly, we will use **Proxy Metrics** generated by the nengo-loihi emulator. The PoC is successful if it reports:

1. **SynOp Count:** The system must log the number of synaptic operations per query.
   * *Target:* Show that SynOps scale linearly with active neurons (sparse), not total neurons (dense).
2. **Sparsity Rate:** The system must log the % of neurons active per timestep.
   * *Target:* >90% sparsity (meaning <10% neurons fire).
3. **Pattern Completion:**
   * *Target:* 100% retrieval accuracy when the query vector has up to 20% noise added.

## 6. Migration Path to Hardware

To move from this PoC to physical Loihi hardware in the future:

1. Change `target='sim'` to `target='loihi'` in the Nengo Simulator config.
2. Install `nxsdk` (Intel's proprietary driver) in the container.
3. No other code changes are required due to the architectural isomorphism enforced by this PRD.
