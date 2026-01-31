# **Product Requirements Document (PRD): Neuromorphic Memory Bridge (PoC)**

**Version:** 1.0

**Status:** Draft

**Author:** Senior Neuromorphic Engineer

**Target Hardware:** Standard CPU (Simulating Intel Loihi 2\)

**Frameworks:** Intel Lava, Nengo, gRPC (A2A Protocol)

## **1\. Executive Summary**

The **Neuromorphic Memory Bridge** is a specialized sub-agent designed to offload long-term context retention from Large Language Models (LLMs). Unlike vector databases (RAG), which perform static retrieval, this system uses a Spiking Neural Network (SNN) to form dynamic, plastic associations between concepts.

This Proof of Concept (PoC) will simulate the behavior of the Intel Loihi neuromorphic chip using standard CPU hardware. It will expose a standardized **Agent-to-Agent (A2A)** interface via gRPC, allowing any LLM agent to "plug in" a synthetic hippocampus for associative recall and continuous learning.

## **2\. System Architecture**

The system follows a **Microservice Architecture** encapsulated in a Docker container.

* **Frontend (The Interface):** A gRPC Server implementing the A2A Memory Protocol.  
* **Middleware (The Encoder):** A "Dense-to-Sparse" encoder implementing the **FlyHash** algorithm.  
* **Backend (The Brain):** A **Nengo** simulation running on the nengo-loihi emulator backend.  
* **Orchestrator:** A **Lava** Process that manages the lifecycle of the Nengo simulation (Run, Pause, Reset).

### **2.1 Data Flow**

1. **Input:** Main Agent sends a 1536-d float vector (OpenAI Embedding).  
2. **Transduction:** Middleware converts this to a 20,000-d binary sparse spike train (FlyHash).  
3. **Ingestion:** Spikes are injected into the Nengo Network.  
4. **Dynamics:**  
   * *Write:* Synaptic weights are updated via Voja (learning rule) and STDP.  
   * *Read:* The network settles into an attractor state (Associative Recall).  
5. **Output:** The active neuron indices are mapped back to a Context ID (UUID) and returned to the Agent.

## ---

**3\. Functional Requirements & Implementation Steps**

### **Feature 1: The A2A Interface (gRPC)**

**Objective:** Define a strict contract for Agent-to-Memory communication.

**Step 1.1: Define Protocol Buffers (memory\_a2a.proto)**

We will define a service with two core methods: Remember (Write) and Recall (Read).

Protocol Buffers

syntax \= "proto3";

package memory\_bridge;

service MemoryUnit {  
  // Agent sends a context vector to be stored  
  rpc Remember (MemoryPacket) returns (Ack) {}

  // Agent sends a query vector to trigger association  
  rpc Recall (QueryPacket) returns (ContextResponse) {}  

  // Trigger sleep phase for consolidation  
  rpc Consolidate (SleepSignal) returns (Ack) {}  
}

message MemoryPacket {  
  string context\_id \= 1;      // UUID of the text chunk  
  repeated float vector \= 2;  // Dense embedding (e.g., Ada-002)  
  float importance \= 3;       // 0.0 to 1.0 (Modulates learning rate)  
}

message QueryPacket {  
  repeated float vector \= 1;  
  float threshold \= 2;        // Similarity threshold  
}

message ContextResponse {  
  repeated string context\_ids \= 1; // IDs of recalled memories  
  float confidence \= 2;  
}

message SleepSignal {  
  int32 duration\_ms \= 1;  
}

message Ack {  
  bool success \= 1;  
  string message \= 2;  
}

**Step 1.2: Generate Python Stubs**

Run the compiler to generate the Python interface code:

python \-m grpc\_tools.protoc \-I. \--python\_out=. \--grpc\_python\_out=. memory\_a2a.proto

### ---

**Feature 2: The Encoder (FlyHash)**

**Objective:** Convert dense LLM embeddings into sparse binary spikes suitable for the SNN.

**Step 2.1: Implement FlyHash Class**

This utilizes a random projection matrix followed by a Winner-Take-All (WTA) inhibition.

* **Inputs:** Dense Vector ![][image1] (Dimension ![][image2]).  
* **Parameters:** Projection Ratio (![][image3]), Active Bits (![][image4]).  
* **Logic:**  
  1. Initialize a static random binary matrix ![][image5] of shape ![][image6].  
  2. Compute ![][image7].  
  3. Identify indices of the top ![][image8] values in ![][image9].  
  4. Return a binary vector where top ![][image8] indices are ![][image10], others ![][image11].

*Why this matters:* SNNs on Loihi are energy efficient only if activations are sparse. FlyHash guarantees fixed sparsity.

### ---

**Feature 3: The Neuromorphic Core (Nengo-Loihi)**

**Objective:** Build the SNN that stores and retrieves patterns. We use **Nengo** for its high-level "Associative Memory" primitives, running on the **NengoLoihi** emulator to enforce hardware constraints (integer weights, discretized spikes).

**Step 3.1: Define the Nengo Network**

We will use nengo.networks.AssociativeMemory.

Python

import nengo  
import nengo\_loihi  
import numpy as np

class BiCameralMemory:  
    def \_\_init\_\_(self, n\_neurons=1000, dimensions=1536):  
        self.model \= nengo.Network(label="Hippocampus")  

        with self.model:  
            \# 1\. Input Node: Receives spike trains from FlyHash  
            self.input\_node \= nengo.Node(np.zeros(dimensions))  
              
            \# 2\. Memory Ensemble: The core storage  
            \# We use Voja (Vector Oja) learning for building associations dynamically  
            \# and PES (Prescribed Error Sensitivity) for error correction.  
            self.memory \= nengo.Ensemble(  
                n\_neurons=n\_neurons,  
                dimensions=dimensions,  
                neuron\_type=nengo.LoihiSpikingRectifiedLinear() \# Hardware-aware neuron  
            )  
              
            \# 3\. Learning Connection (Plasticity)  
            \# This connection learns to map Input \-\> Memory State  
            conn \= nengo.Connection(  
                self.input\_node,   
                self.memory,   
                learning\_rule\_type=nengo.Voja(learning\_rate=1e-2),  
                synapse=0.01  
            )  
              
            \# 4\. Output Probe  
            self.output\_probe \= nengo.Probe(self.memory, synapse=0.01)

    def build\_sim(self):  
        \# target='sim' creates a CPU simulation that respects Loihi constraints  
        return nengo\_loihi.Simulator(self.model, target='sim', dt=0.001)

**Step 3.2: Implement the Recall Logic**

When the gRPC Recall is hit:

1. Pause the simulation.  
2. Inject the Query Spike Train into input\_node.  
3. Run simulation for 20 timesteps (20ms).  
4. Read output\_probe.  
5. If the network settles (attractor state), decoding the active neurons identifies the stored Memory ID.

### ---

**Feature 4: Integration with Lava (Process Management)**

**Objective:** Ensure the system is future-proof for pure Loihi deployment.

**Step 4.1: Wrap Nengo in a Lava Process**

While Nengo handles the math, we wrap the entire simulation controller in a **Lava Process**. This strictly enforcing the *Communicating Sequential Processes (CSP)* pattern.

* Create a class HippocampusProcess inheriting from lava.magma.core.process.process.AbstractProcess.  
* Define InPort (for vectors) and OutPort (for IDs).  
* This wrapper doesn't change the logic now, but it means later we can swap the Python backend for a C backend running on the Loihi chip without changing the A2A API.

## ---

**4\. Development Workflow**

### **Phase 1: Environment Setup**

1. **Docker Base:** Use python:3.9-slim.  
2. **Dependencies:**  
   nengo==3.2.0  
   nengo-loihi==1.1.0  
   lava-nc==0.5.0  
   grpcio==1.50.0  
   numpy==1.24.0

### **Phase 2: The Encoder Test**

* **Action:** Create a unit test test\_flyhash.py.  
* **Input:** Two semantically similar vectors (e.g., "King", "Queen") and one different ("Apple").  
* **Verify:** The Hamming distance between the FlyHash binaries of "King" and "Queen" must be significantly lower than "King" and "Apple".

### **Phase 3: The Associative Memory Loop**

* **Action:** Write the main server.py loop.  
* **Logic:**  
  * Initialize BiCameralMemory.  
  * Start gRPC server.  
  * On Remember(A): Run SNN for 50ms with Input A \+ Plasticity ON.  
  * On Recall(A') (noisy A): Run SNN for 20ms with Input A' \+ Plasticity OFF.  
  * **Verify:** Does the SNN output vector match the original A vector?

### **Phase 4: Containerization**

* **Action:** Build the Dockerfile.  
* **Constraint:** Ensure nengo-loihi is configured to use the CPU backend (target='sim') via an environment variable LOIHI\_BACKEND=SIM.

## **5\. Success Metrics (The "Proof")**

Since we lack the hardware to prove energy efficiency directly, we will use **Proxy Metrics** generated by the nengo-loihi emulator. The PoC is successful if it reports:

1. **SynOp Count:** The system must log the number of synaptic operations per query.  
   * *Target:* Show that SynOps scale linearly with active neurons (sparse), not total neurons (dense).  
2. **Sparsity Rate:** The system must log the % of neurons active per timestep.  
   * *Target:* \>90% sparsity (meaning \<10% neurons fire).  
3. **Pattern Completion:**  
   * *Target:* 100% retrieval accuracy when the query vector has up to 20% noise added.

## **6\. Migration Path to Hardware**

To move from this PoC to physical Loihi hardware in the future:

1. Change target='sim' to target='loihi' in the Nengo Simulator config.  
2. Install nxsdk (Intel's proprietary driver) in the container.  
3. No other code changes are required due to the architectural isomorphism enforced by this PRD.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAAA30lEQVR4Xu2SMQ4BQRSGf6GSiKBSioYDEAlOoKFEqLUaiSgUInEDohKVRqVwBgVn0DsE/9vZYffJZtWyX/IVM//Mm8m8ASIsfTpVZnwrgAWduZlY98fAiD7pnqZUJiTpmrZoTGUObZgCZ5pTmdChS5rQgaUBU+BKSyoTDrSgJ73IJtl8pxV/5JzaU3NfZOkJ5hYDz3ye7jzjQOSRNjAF5LUFOVlefGgXhTGHKbByx026pen3ihC6+HRiQscIaFkQthMXmP+gP1MoZXqjD1pV2U/YThx18CtxWqNFHUT8NS8ROCImoRcdXgAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFEAAAAZCAYAAABJhMI3AAADPElEQVR4Xu2Xy6tNURzHvx55y1uSwb2JvCIDIo8oj1K3FBl4x0SRR0SSjERJykBCXgOKCAOPJKJ0DRjJxOTM/RH8Pn57nbPuOnvve6R7T932p76d9tprr7PWd/9+v7W2VFFRUVFRRIfppOmO6XWma6bTkbpMo7P+7Wa6aXjaaIxU/hzz2oaaJpommQYl91ImmBabppoGJ/fqjDetNu0y1Uz7TWszrTOdM/00fc36twsWO8PUbZqT3IOVpl+m8/IXz7xfmBbEnYytpvemS/I+H0wrlG/mXNNbeb8Hpls9bzdzwHTRNCS9kcGfX1V+FPQlIWpGySOHl1lkIqZh4CHTfDVHzgjTb3nWjcnarmdtV0zDsrZ5ps+mM9k1/x36FcLgpDDRWAQT/yRP/3bRm4kstAxM6s1EguSyPCuXZH1goelIdN0EdeadaVF6I4KJ/1B5n77mf00E6ltcJ1/KTSTTgLH5jzemKaFTK6yS1xPqYxGYV1PPt9PftGLiNjVSurNHj3xYN1kYInOz3NT7ptnymvnFdEz5m9RfWsl3iu5Z02OVGz1TjU2pFbGhlY2XUmYim8C96Jo5HzRtj9oCHfL/32K6LU/VAJsIXtRMy6L2jVk74zbRIa91ZbsvYU3Yn1DBIP1EmYl5EJ1P5BtTERjP6YN0Zm3BxDSdyUTKGaWvieAwUVbEennYd6Y3+pl/NZHo/Jb9FrFXvv6P8vVxSuH6pjxLA6FW5pYztnEeIl3zoFbckJvYziiEIhNppyQ9TNrpx9qISODIQ0TFRx/u0QdRD0NNZLw8E8NYdcbKCygPEZEpmLZTbiDFujeOqzGhVlRTwZstoMjE2IiY5fJNYVZ2HTKqq95DWqPGsxgYzErrf4jqMFadHfKHeYPjsjaMmywfsNu0Qc2H1nbAvNgtWQjGx3PiQH7KtDtqm2Z6Jj84B/jiOarGoRoeyY09rEamcVqhTobxGJ9M5QBeBzd5Q2lkBDHAXfn3aLsh+l6peY4o3gwpOxfU+OQjAPiCSXlueio/OO8zfZcHE0YFMJOdmxe2R14fMZDIHvAQnZQlvvlJv9iYAH1IWfrQtyxQyM5NpqUq71dRUVFRUVFRMRD4AzdusGkzJTCrAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGoAAAAZCAYAAADZl7v4AAADlElEQVR4Xu2Yy+tNURTHl/czeSTP9EMJkSQDeYTE1MAj8hr8BkoGHinEL8ojkmcSSikZSInyGBgZGTCQTEzuzMAfwfrcdTb7rnvOvefeuvf+ZH/qW/futc9jr7X3WnsfkUQikUgk/g36VCdUj1RvVadV4+MODmwDYn2fqo6rZtT06C7jVNN8Y8RY1bxMI52tmywX8y0+w3foqupkJmKwQTUqXOCZqFqn2qH6pfogFrw8hqh2q76K9eXGq1Vj4k5dBMe/EZs4efBeH1UXMjG2Qkd0mOli/jqk+qTanv1HW1Q3VT/EAtiQA6rzYgFYU2v6A5HfKPagd87WTVjVk1RDxd43L1BLVF98o1gbtl4wTHVZddAbMlgIZCjE7zqYmddUR8UGvrfWXIULCdJWsT7Xa809oyhQYSwe2oocFcAfzdIkKZWJ0gqTVa9Ua70hgkXyTCzT1TFVrEbtEhsIudSzWOzlz0hxMHtBXqBw4r3M5qHtrmq0N0TgrCeq2d6QsUl1XzXFG5qwTPVNNdMbInj2eynos1J1RaxTWC3xjGLgZ8WiTLR5GA8dDOQFCgdSu4oCVThjI6i9BMtDZrkjrQcJmNw8v9FqpV7l7hO4iCARLKJINKk/rDKYK7bCKMJhxZFnybdlYKMSimYZzbfLStNOoLCVdfQL1VKxNLdPbDW2A/58raq49piQCfxCqUJwHojdKKwYNgsLxYJDkAgWECAGSp0aLHQ6UASJYF1S3ZDGR5dGsBAqYsEqAp/j+23eABS2i2IrhLzNjGEwpEECEgeFh3xWLYraek2nAzVcrC6T7skO7cIGhmc32oT1i/VhR1sHuyNSWoBB05m2c2LLMVCRcvm9m+QFqtlmAls8rkaQURApmZq1vsZajngBFG3C+GjAjvC7NwAve1tqNwah4B1WLYjagXZmFwW1LBWx68rqWPWq8nDNgG8Ucy42D21MzjJwYOaLAasKOLQ+ltbGD6H2F23CuD9n1J+qPc5WLY6c1okin1fCw0MujYsmNoLGYRF7qy/aKSi4OJ764VcI/9mdxZ+3+M15sdnXCeoQ9YhUlAf3pY6U9cMpsfdk0xaCjv9nqfarXqpWSM79+sS2gPFMfhjZnoudm0aIfY/ys57Ir8r694Kw6vMUwwYJJ4Ray+8ytWmzaqfkOC5jguqIFJ+zAvgIX/l3DOJT3C35G7z/GnZSnE04pM5xtkQikUgkEolEIpEow29dicwMx8kgPAAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAZCAYAAABnweOlAAAC30lEQVR4Xu2Yy+tNURTHl0fe8n6XrmciGRl5dJVioigpwkBEMfAokqQIMaAolFeMREwYkMkvogzMzEzMDPwRfD+ts1lnd+51Ju79+XU+9e1399rr3nv22muvve7PrKGhwZkgTZGG5xOBYdK43DhUOSs9ka5JN7O5BAHZaO5TyQrponQ608Hg05JOZvNLw3wvGC+NzY1idHjNYp9LU4vxZmmLld83STohvbMua5gmrZf2SD+k49IGaWXwmWEeiAHpsLTGqh/wX8LzfZfOmT8LG/lWmhd85kpnwniidFR6Kr2RXkuvpE/StuDXEYJxXRoVbCnNHkqLg70fEJTH5gE5IM03f77IWnO/3Ba5J83JbJVQcO5IhzI7acYZJXX7DYslS7qxyspBGVHYEqyTU1CLlvReWh1s7ATnsVsF7yV1gkItOW9/Mohjn+oLtt3SyGL8VzZJP80/JB2ZzyWP/pOCss/8CPF3VsnD+SAtN1/8qWDnQnkQxl2hGHFWCcpd8+zYUYwXBr+6UIApxKRpXVHU0+52oi3dCmMWfVVaF2xV8LnHpL3Fa56PMZtO4a1kmbkDQWgVNqo41Xp/MR6skD03rHwt5xA0LhAaOrhkHlzGs63DtbzVPCDc74kU3XjnD0ba5pvHse8Etw3HKfHVytnFZVLKUgbcLgSFvxGODvbtmb0fLJKeWfn4ANctWU62V8ER25XZcv8rlrX7k82/jMWTMRGuM+z0KCn16rDAvFvkvXVFuz2GN3eAY4IfuxzZKb20ztlMR5sfrTwotCI0sL85Yv5ldHwz40TBgPk8BXh6eaqnsCnUBW7FBEfio3ldyGHnCTRFPOeblVuPC+YJYEvMW918x1rJ07xdzudjt9trqBuPzK/jy9IX6X7Jw6Ek0I9wHVf1JNRJftfhx3y7NPsfwiLoqciYllU3ltwmt83/dVAF8y/MM4lf01WBG3KwyG71CfChDvGLuaGhoaGhoU/8AligfcnonwFNAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAZCAYAAACl8achAAACYElEQVR4Xu2Wy6vNURTHv97vvFIeKQYkIgN5FEUpUoqJQkIpjG4MSMpzQCSPJCUpycRICgMzpQz8Cf4CfwTfz117s3/7nGNw+91zbtxvfTrn7rXPvmuvvdbaWxrXuP5NHTWXKqabmeZUF9u++JlWV+MXzbJkG1VNMOvNB/PTvDZ70vhks8kcS7arZpdZPPxLaZE5YD6a82abmZZsfREOZcdqLVfYDtcGa6ciygNRdvq+mVqMTzIXko2Il5pr7pmV1XjfdELh2FNFLmdtNA+T7UwxjthEvZG+KuftM/1xms+7ZkOylalDdIky0R6YtiscoyAXmiUKpyg2VJ/CNUWhDlRU/g/zzawxQ2ZvYS+dJrr9ymO6V+5mHcJRHM5O31SzfeH0O7PAHFSPRVrWLPNc0QhoCB3KThPt3WZd0/w7dXJhjgmRx/mCIZfrSDL+RZEibGpMaI55qYh0mctZZep003HF6XxW3JrordliDpnNaZwL6qtZZW6Zs2Zimv/CzFYEjG62QrFGr/85XGBE8XH6XgtnTqvzBLJwgiudjRMAxCZxkjcKn6w7z6xVrMP1vzTNRXkzOL7V7FDUUe5gHSLRmdjrwUNLnFEPFsKJG4o3SFZ2grVvF+OIDTC/DMJ7NR3kMqtv6FZFBN8o2hPCmUeKDkTH4ZhLUUPlbcq8y2p2iSeKm3rURGp8Mg8UOVx2GGykXq1X5og5Z65XNvTX1GhDHDWR4pYk6rm4EGNTir+zmDNf3WsIkVJd+3NbKlNjpCKlTporigjvb5rbFcX7XdHfeVyNVBTcHUUus9bA3zb/t34BEJVlAnP/Yh8AAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAZCAYAAACCXybJAAADO0lEQVR4Xu2X26vMURTHl0vuIlJuD0MRUlKIXOpI8ehBQtFRFA9ejlsSJ5TkkkJSlPKiiJRyeVBKKQ8kSR7n3R/B+sz6bbN/e/b+/X4zx6hhvvXtNHv99t7rvtcR6eP/xuiMvYqxyinhYhHYcFw5KhT0EDD4vFS0AYNPKE+HAsUk5QHlFeWrjA/Fvj2V8aByvtvQAWpi998XO5+ziyKGbFia+hxTzslk05R7pILhA8rnElcch6xSblM+VV4V+95xv/Kt8ofyiG1pG9OVm5Q7lZ+U78QckQJGfVX+FNNnvXKiJ8cRy7zfLdihfK9cGAoCrFB+U84NBRnYX+WcIgwqD4sZsyEv+g2ivFn5UflaOSsvbmCtmFP4G8Ud5XXluFAQYLeYMqR7DKxz1vZQUBHcf025TuyevXlxA6TsLrE7+CalN5nzWHlBOSaQNYDHBsPFAGy8JHZRCs7oQ6GgIogYNU0mcU+sv5CyJ5VnJO0YgHNoaC8kngmNWsS7RWAjB9SDdR9TlQ+UQ6GgIlYrL4tFLhZFnHpWrO8QRUqNkksBhyS/eSPpOnXgAKdIChvFHIjy7QLjMNjtRSe/XheIRX589htdyLxo6mbgrLoksuGlcma46AGFMLYsnVzKzQhkVYDT70rTSCJJ2S3JfmMwhgMM5Z6y3sFezmjRmemrzGgUwvN1SUeRb4gMke4EZMlFaUbutuQ7uG8gjuFZW+qtxZA0GpQZTb1jTKopEOV9YkqibCegD/A6OAyLnccaEfZfDBxPJtChi4BTcE7U6A/KReGiB7pxUQ3RUXmfnylnB7KquCn5huN6CMMO3doH+lBKZRMXWeIc1wKiuDVczED6Y9B35UpvHc/z8N9SHlVO9mQOE8Qi/0W5PJA5cP48sWmQocYZ4poQ+904imyx2PSHvMxonJOc7KjF8JlxCuOpGF26l73J1GJd4tNVTUwp/9x7Yg5E9kSaoyRzf0yH1FPrZoYb0uz4OTC1VKmRToFjYkZ3EzUxhzJiR7FG+VkK5tQR4pwkUqyLwFgyuHD+oAE9Um4JBSME51KHfxPUut8LCkH90IH/JLpVMkUYkDZfkZqk/4vqBZDOsX9U+ujjX8cvX6acdmXZHmMAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHYAAAAZCAYAAADkBdqeAAAEMElEQVR4Xu2Z2atPURTHl3nOFJn73USGyIMxQ5QidcOLQuoqZYiEkGROZMiYFDIkD0RJGYonpTzwIHnxcsujP4L1sc7ut3/rnnv9fvfe3zmXzqe+3X57nWGfvfZea+19RQoKCgoKCgqyYLrqtOqQ01bVQNWOFBuazM0ZMVN1Uirff0w1K7qmm2qTu2ZzZG8PG6Xld/dV9VdtSbGtstv+jE3cflA1NrFlxgjVMrFB+anam/yeoeqpmq1aqXqmuqNarlqo6sfNGTFKrE/7Vc2qXaolqiHRNTiWPv9SPVI1qiZG9loJz3sl5WeuSNrDuDBm2Jhk9I9+AmO6RvVa9VW1QNUnsWUODr2k6u0NyhyxAS1VNmdOSWwgF7n2QINqnG/sIDgtOM8zXsy23huUpWKrlUmQG4SWm2IzMI0msQ/gujwZLtaPtH6yIi74xk4gONZP+h6qA4nN92ew6qLYRMuVkuq92Mr08DF8FB+QN0ws+kF08ZAurvnGTqBJ7J1M/Hhik9+vJLZtUTvgaO/sXCB30EFyg2eM6p1Y/s0bVgn9pJAi1wVGq+5JfQq6kEdvSdmx/D0vVtT5MM0qZbWyanNlkOqBWAcpALwuJ7Yb4Ya/QEFDYeOf05ZqKXLoywvVsOQ3918Xq+Brgeq2mvxHPuedFFGkAiYRjguLwK/m41Ldc+vOFNUn1WdvkHLuTQs3eUFfiCBEErgvlduealis+q566g0pUNESrRgjxmq3WNgPxI5llTZEtnpCVR6q9FQoy+ncE2+Qcu5tlvT8mwffxCbhVLHQzH671Y9rBXI039zs2tMIEz849pRUbl3iCLJWau9LexggtvWkeGMMWkAnjkg5b3mY2czWN5Kef/Pgo1h/CZH0b2iluSoIqUSg1d6QQnAs48AengOdmBCmQzHVJSAfslLpHCvXE2a2L/XbIux5ua9a7ePGKgkHBjvFaoN6wyQI7yS3+hVJ+wexcIzjuwQMDh17rBrpbN1VX1Q/xDbb/oPy4qxYn8mPWUSRUFyyYuPcGojDdBrUAqxyUhqFHjxXzVOtU81N2jnkIBpNUp1RbRfzAdwVKw7xAVV6SewZLd45TSxP+ZVTEovft1NsqCtwWKzwIQxnQSggqbzTDmhwWFt5foJYNGRyMEmAiYAj2Z7xl+cSPfELz+EoMhSHEByOc+eLfTt5PYuJnRlsjfi41KKhDvAe3tfaIT65/m/n5tQv8aFKcBTPJgLF4GS/T38plU6kPqglPRbUCeoYtiaAw66KVdZU0oTUGHJ6fGrFdUSpeCJzptAU/S7IibdiBz3k1LhyJkQT5j0PVRtUe1QnnA3+uzD8L8IKZcVxGkUeDQUR0NYr+h3gGrZxaTkdCN9ZpaKCVsCZIQy3FybHZtVRsZXaWGkuyBoKrnNi+1/+YdBeKJJ4DrmVZ3WJs+iCgoKO8BugGdOf/PnhIAAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAABEUlEQVR4XmNgGAWkAkcgbgLiSjRsCZXnAuIUNDkzqBwKUGZAGPYfiAOB2A6IBaDy7EAcAMTnoLgcSQ4DgGydzQAxCB3MAGIvIGZEl8AGFID4MBCfRRM3BmJNNDG8wIEB4prVUD4LEMcAcRtMAbGgiAFiUC0Q8wFxJxBPAWIOZEWEgAIDxFuggCwEYm0gns8AMbgEoYwwcGCAaHrNAHERCIRAxXZC+UQBmLcagJgJKibFADEEZDhRgBeIFzFADHJHEgdFNchbIHFQ0iAIFBgg4XMSiFVRpRh0gPgSEFugiWMFoNgB2Qqi0WMIlATyGCBpC5SeYN5GATDbQIYg436ovAIDxKXIcjuAWAIqPwpGAdUAADQhNXUjPoBPAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAA6ElEQVR4Xu2RIQ7CQBBFPwmCEEg4QhMMguAQGFIUAgOWY+BIwOAQhAPgsByAA6AxKCQ3gfnMhu5Om6aA7U9ems6bbmd3gTLfZiRshKVh4Lytk55zQdpIFnsKM2EotJz33VmYeC6VunCANtvwoxPUxaFKJxIuwtXUmQ60Th+FKp0Y+sejqTNTqOPEnDw3C2jz2gpojY49uYmgY9+EOfRwfe7Osy83MZJtNUP1zk/bqhjH90Lb4gSchM1j4xhe/dfXzmu2Ya3Q+Wyhf+SzZlxD2AkroWrcJ13oLXERn73QFx4ZLusMy5T5Oy8BFTnNANCajgAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAASUlEQVR4XmNgGJqAEYhlgJgJXQIZKAJxDxBvA2JhNDkwUAbiVUB8DohfA/F2BhwKYUADiM8yjCrEAainECQIkvyPBXMhqRthAAAiIRpsDSBSQQAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAtklEQVR4XmNgGHqAB4h1gVgCiFnQ5OCAD4i3AXE1EPcB8WEgtkRRAQTsQNwLxIJQPiMQewPxSbgKqGAtED9AFgQCLiCeAcQKMAFeIF4ExGdhAkigHohtYBxhIN7OgFthLIyjwABxOEGFGgwQRQQVEm01Ls+AQqMJiB1gAsxA3AHEr2ECUAALHpDT4MCVAVOhABCvZoBEKwpQAOJpQOwOxBlAfA6IY5AVIANDBohCeyDmR5MbPgAAbvIdC6L9XfwAAAAASUVORK5CYII=>
