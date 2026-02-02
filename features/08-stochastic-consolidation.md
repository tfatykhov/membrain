# Feature 08 — Stochastic Consolidation & Attractor Settling

## Status: 🔴 Not Started — CRITICAL (Core Patent Claim Enabler)

## Parent Module: Neuromorphic Core

---

## 1. Overview

The `consolidate()` method acts as the system's "sleep" or "settling" cycle. Previously defined as a passive state holding the current activation (No Input), it is now redefined as an **active process**.

The function must inject controlled **Gaussian White Noise** into the hidden layers of the network. This noise forces the system to exit "spurious" (unstable) states and settle into the nearest "basin of attraction" (a stable, learned memory pattern). This mimics biological neural consolidation and provides the mathematical proof for our **"Attractor Dynamics"** claim.

---

## 2. Technical Requirements

### 2.1 Functional Changes

- **Old Logic (Deprecated):** `consolidate()` → Freeze weights, maintain current state S_t.
- **New Logic (Active):** `consolidate(noise_sigma, iterations)` →
  1. Add noise to current state vector: `S' = S_t + N(0, σ)`
  2. Iterate the recurrent network dynamics without external sensory input.
  3. Allow the network energy function E to minimize until `ΔS < ε` (convergence).

### 2.2 Mathematical Model (For Patent Description)

The consolidation step must implement the following state update equation, distinct from the standard inference step:

```
h_{t+1} = φ(W_rec · h_t + ξ(t))
```

Where:
- `h_t` is the hidden state vector.
- `W_rec` is the recurrent weight matrix (the "long-term memory").
- `ξ(t)` is the injected White Noise (the "temperature" or "shaking" factor).
- The system settles when `h_{t+1} ≈ h_t` (The Attractor State).

---

## 3. Implementation Details

### 3.1 Interface Definition

The function signature must be updated to accept noise parameters.

```python
def consolidate(self, noise_scale=0.05, max_steps=50, convergence_threshold=1e-4):
    """
    Triggers attractor dynamics to stabilize the network state.
    
    Args:
        noise_scale (float): Standard deviation of Gaussian noise injected.
                             Essential for escaping local minima.
        max_steps (int): Maximum recurrent cycles allowed for settling.
        convergence_threshold (float): Difference in state vector magnitude 
                                       to consider 'settled'.
    
    Returns:
        Vector: The finalized 'clean' memory (The Attractor).
    """
    pass  # Implementation below
```

### 3.2 Pseudo-Code Logic

This logic must be implemented to replace the current placeholder.

```python
def consolidate(self, noise_scale, max_steps, convergence_threshold):
    # 1. Start with current working memory (short-term state)
    current_state = self.state_vector
    
    # 2. Inject White Noise (The "Kick")
    # This prevents the system from getting stuck in "half-formed" thoughts
    noise = np.random.normal(0, noise_scale, current_state.shape)
    perturbed_state = current_state + noise
    
    # 3. Iterative Settling (Attractor Dynamics)
    for step in range(max_steps):
        previous_state = perturbed_state
        
        # Recurrent update: State relies ONLY on internal weights, not external input
        # This is the definition of "Consolidation" vs "Perception"
        perturbed_state = self.activation_function(
            np.dot(self.weights_recurrent, previous_state)
        )
        
        # 4. Check for Convergence (Did we find the memory?)
        diff = np.linalg.norm(perturbed_state - previous_state)
        if diff < convergence_threshold:
            self.log_event(f"Settled into attractor at step {step}")
            break
            
    self.state_vector = perturbed_state
    return self.state_vector
```

---

## 4. Files / Modules

| File | Action |
|------|--------|
| `src/membrain/core.py` | **Update** — Implement new consolidate() |
| `src/membrain/config.py` | **Update** — Add noise_scale, max_steps, convergence_threshold |
| `src/membrain/server.py` | **Update** — Pass consolidation params via gRPC |
| `protos/memory_a2a.proto` | **Update** — Add params to SleepSignal |
| `tests/test_consolidation.py` | **Create** — Attractor settling tests |

---

## 5. Acceptance Criteria (Test Cases)

### Test Case 5.1 (Noise Resilience)
- **Given** an input that is 80% similar to a stored pattern (a "noisy" memory).
- **When** `consolidate()` is called with `noise_scale > 0`.
- **Then** the output state must be >99% similar to the stored pattern (The system "fixed" the memory).

### Test Case 5.2 (Spurious State Rejection)
- **Given** a random noise input.
- **When** `consolidate()` is called.
- **Then** the system should settle to the nearest valid pattern or zero (it should not Hallucinate indefinitely).

### Test Case 5.3 (Convergence)
- **Given** a stored pattern.
- **When** `consolidate()` is called.
- **Then** the system should converge within `max_steps` iterations.

---

## 6. Why This Matters

This feature is **CRITICAL** for the patent claim. The stochastic consolidation process provides:

1. **Biological plausibility** — Mimics hippocampal consolidation during sleep
2. **Mathematical foundation** — Proves attractor dynamics with energy minimization
3. **Noise robustness** — System can recover from corrupted/partial inputs
4. **Differentiator** — This is what makes Membrain more than "vector search with extra steps"

Without this, Membrain cannot claim true "neuromorphic memory" behavior.
