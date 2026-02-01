# Product Vision: MemoryUnit

**The Bi-Cameral Memory Co-Processor for Autonomous Agents**

**Status:** Strategic Draft  
**Target Audience:** Enterprise AI Architects, Agent Framework Developers  
**Core Tech:** Spiking Neural Networks (SNN), FlyHash, gRPC (A2A Protocol)

---

## 1. Product Thesis: The "Living" Index

Current AI agents suffer from the **"Goldfish Effect"**—they possess high reasoning capacity (via LLMs) but lack narrative continuity. The industry-standard solution, Retrieval-Augmented Generation (RAG), treats memory as a static library: store text, search by similarity, retrieve text.

**MemoryUnit is a paradigm shift.** It is a drop-in, stateful "memory co-processor" that treats memory as a dynamic biological process. It does not just store data; it *metabolises* experience into associations.

### We are selling:

1. **Associative Recall:** The ability to retrieve information based on temporal context and partial cues ("what was I doing before this?"), not just semantic similarity.

2. **Temporal Chaining:** A system that learns sequences via Spike-Timing-Dependent Plasticity (STDP), enabling agents to predict next steps in a workflow without explicit instruction.

3. **Automatic Hygiene:** A biological approach to data retention where memories naturally decay unless reinforced, solving the "context clutter" problem of infinite vector stores.

4. **Stability via "Sleep":** An offline consolidation phase that protects critical policies from catastrophic forgetting.

### The Wedge:

We do not compete with Vector Databases (Pinecone, Milvus) on storage density. We compete with Agent Memory Layers (Letta, Zep) by offering **dynamics**—associations, sequences, replay, and forgetting—that static indexes cannot replicate.

---

## 2. Product Hypotheses & Validation Strategy

We frame our product claims as testable hypotheses.

### H1: Associative Recall > Similarity Search

- **Belief:** In agentic workflows, the most relevant context is often situationally linked (e.g., "When user complains about X, check Policy Y") rather than semantically similar. Vector search fails here because "complaint" and "policy" are semantically distant.

- **Mechanism:** Our Recall endpoint triggers "attractor states" in the SNN. If "Complaint" and "Policy" have co-occurred previously, activating one triggers the reverberation of the other.

- **Ship:** `Recall(query_vector, threshold)` returning a list of `memory_ids` derived from the network's settled state.

### H2: Temporal Binding Yields "Next-Step" Continuity

- **Belief:** Agents fail multi-step tasks because they lose the thread of execution. A static database doesn't know that Step 2 follows Step 1.

- **Mechanism:** We utilize STDP (Spike-Timing-Dependent Plasticity). If the agent writes Context A followed immediately by Context B, the synapse A→B is strengthened. Future occurrences of A will pre-activate B.

- **Ship:** `Remember(write_vector, metadata)` with an internal "temporal binding" flag that reinforces links to the immediately preceding write operation.

### H3: Forgetting is a Feature (The "Lean Memory" Advantage)

- **Belief:** Infinite memory is a liability. It increases retrieval latency and noise. Most agent interactions (e.g., "Thanks", "Ok") are trivial and should fade.

- **Mechanism:** Synaptic weight decay. Without reinforcement (re-access or explicit importance flagging), connections degrade over time, naturally pruning the index.

- **Ship:** A configurable `decay_rate` and an explicit `importance` scalar (0.0 - 1.0) in the Remember API.

### H4: "Sleep" Prevents Catastrophic Overwrites

- **Belief:** Continuous learning usually destroys old knowledge (catastrophic forgetting).

- **Mechanism:** A `Consolidate` mode that mimics biological sleep. It cuts off sensory input and injects noise to trigger "replay" of strong memories, interleaving old and new patterns to find a stable joint representation.

- **Ship:** `Consolidate(duration_ms)` endpoint to be called during agent idle time (e.g., nightly).

---

## 3. Architecture Refinement: The Split-Brain Design

To make this "product-real," we must separate the experimental neuromorphic index from the critical data storage.

### Component A: The Payload Store (Reliable)

- **Tech:** Standard SQL/NoSQL (e.g., PostgreSQL with pgvector).
- **Role:** Stores the actual text, files, tool outputs, and timestamps.
- **Security:** Encrypted at rest. PII redaction happens before data hits this store.
- **Key:** `memory_id` (UUID).

### Component B: The Associative Index (Neuromorphic)

- **Tech:** SNN (Nengo/Lava) running on CPU (Phase 1) or Loihi (Phase 2).
- **Role:** Stores only the `memory_id` encoded as a sparse pattern.
- **Function:** It maps `Current Context Vector → Relevant memory_ids`.
- **Privacy:** The SNN never sees raw text, only hashed vectors (FlyHash) and UUIDs. This is a massive privacy advantage for enterprise deployment.

### Critical Engineering Guardrails

1. **Write-Gating:** A pre-processing pipeline that scores "Salience." Only interactions with a salience score > threshold are committed to the SNN. This prevents "trash" from altering the synaptic weights.

2. **Stability Controls:** Implementation of homeostatic plasticity (if neurons fire too much, increase threshold) to prevent "epileptic" runaway association where everything connects to everything.

---

## 4. Competitive Landscape & Positioning

| Product | Core Philosophy | Our Positioning vs. Them |
|---------|----------------|--------------------------|
| **Letta** (formerly MemGPT) | OS for Agents: Manages context window like RAM with explicit paging. | Letta handles the Context Window; we handle the Long-Term Substrate. We are the hard drive to Letta's RAM. Our sleep-based consolidation offers better long-term stability than Letta's explicit archival. |
| **Mem0** | Personalisation: "ChatGPT Memory" for developers. Optimises for user preferences. | Mem0 is excellent for simple facts ("User likes golf"). MemoryUnit is for complex workflows. We remember sequences and causal relationships ("If user asks for refund, check warranty first"), not just static preferences. |
| **Zep** | Knowledge Graph: Extracts entities and builds a graph structure. | Zep is structured and explicit. MemoryUnit is implicit and fluid. We handle "fuzzy" associations that don't fit neatly into a graph schema. We are faster (O(1) retrieval) and more noise-tolerant. |

### The "Blue Ocean" Pitch:

All competitors treat memory as **Information Retrieval**. We treat memory as **State Reconstruction**. We don't just find the document; we reconstruct the cognitive state required to solve the problem.

---

## 5. Development Roadmap

### Phase 0: Baseline Harness (Metrics)

- **Goal:** Quantify "better."
- **Build:** A test agent (using LangChain/LangGraph) with a swappable memory backend.
- **Benchmarks:**
  - Partial Cue Retrieval: Can it find the memory given only 20% of the context?
  - Sequence Prediction: Can it predict the next tool call without a prompt?

### Phase 1: API & Payload Store (The "Skeleton")

- **Goal:** A usable microservice.
- **Build:** gRPC server implementation of the A2A protocol.
- **Deliverable:** MemoryUnit container that accepts Remember/Recall calls, stores data in Postgres, and returns simple vector-search results (Mocking the SNN).

### Phase 2: The Encoder Bridge (FlyHash)

- **Goal:** Connect the vector world to the spike world.
- **Build:** Python implementation of FlyHash.
- **Deliverable:** A module that reliably converts 1536-d OpenAI embeddings into 20,000-d sparse binary vectors (SDRs).

### Phase 3: SNN MVP (Simulation)

- **Goal:** Prove Associative Recall.
- **Build:** Nengo network with `nengo.networks.AssociativeMemory`.
- **Validation:** Verify that inputting a noisy version of Vector A retrieves the clean ID for Vector A.

### Phase 4: Dynamics (Forgetting & Salience)

- **Goal:** The "Lean Memory" differentiator.
- **Build:** Implement synaptic decay rules and importance-scalar modulation.
- **Validation:** Run a "lifetime" simulation (10,000 interactions) and verify the index size/performance remains stable (H3).

### Phase 5: Consolidation (Sleep)

- **Goal:** Solve Catastrophic Forgetting.
- **Build:** The `Consolidate` endpoint trigger.
- **Logic:** Offline replay of high-salience memories mixed with random noise to stabilize weights.

### Phase 6: Hardware Acceleration (Deployment)

- **Goal:** Scale and Efficiency.
- **Action:** Swap the Nengo CPU backend for nengo-loihi.
- **Value:** Drop power consumption from ~100W (GPU) to ~1W (Loihi) for "Always-On" memory monitoring.

---

## References

1. [Associative Memory — NengoSPA docs](https://www.nengo.ai/nengo-spa/examples/associative-memory.html)
2. [Sleep prevents catastrophic forgetting in spiking neural networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010628)
3. [ASP: Learning to Forget with Adaptive Synaptic Plasticity in SNNs](https://ieeexplore.ieee.org/ielaam/5503868/8330765/8094937-aam.pdf)
4. [Sleep forms joint synaptic weight representation - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC9674146/)
5. [Brain-inspired algorithm mitigating catastrophic forgetting - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC10456855/)
6. [Mitigating Catastrophic Forgetting through Threshold Modulation](https://openreview.net/pdf?id=15SoThZmtU)
7. [The power of bit pair expansion - HTM Forum](https://discourse.numenta.org/t/the-power-of-bit-pair-expansion/10082)
8. [arXiv:2303.08109 - SNN Catastrophic Forgetting](https://arxiv.org/pdf/2303.08109)
9. [Loihi 2 Technology Brief - Intel](https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf)
10. [Benchmarking Neuromorphic Hardware Energy - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC9201569/)
