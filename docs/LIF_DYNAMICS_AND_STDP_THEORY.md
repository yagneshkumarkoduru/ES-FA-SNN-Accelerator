# Theoretical Foundations: LIF Neuronal Dynamics, STDP Synaptic Plasticity, and Spike-Driven FlashAttention

**Author:** [Yagnesh Kumar Koduru](https://github.com/yagneshkumarkoduru)  
**Affiliation:** Esthien Labs  
**Domain:** Theoretical Neuromorphic Computing, Computational Neuroscience, VLSI Microarchitecture  

---

## 1. Continuous-Time to Discrete-Time LIF Neuronal Dynamics

### 1.1 Biophysical Membrane ODE
The continuous-time passive membrane dynamics of a Leaky Integrate-and-Fire (LIF) neuron $i$ subject to input synaptic currents is governed by the RC-circuit differential equation:

$$\tau_m \frac{d V_i(t)}{dt} = -\left(V_i(t) - V_{\text{rest}}\right) + R_m I_{\text{syn}, i}(t) + R_m I_{\text{ext}}(t)$$

Where:
- $V_i(t) \in \mathbb{R}$: Intracellular membrane potential at time $t$.
- $V_{\text{rest}} \in \mathbb{R}$: Resting membrane potential (normalized to $0\text{ mV}$ in digital representations).
- $\tau_m = R_m C_m$: Passive membrane time constant ($\sim 10\text{ ms} - 20\text{ ms}$).
- $I_{\text{syn}, i}(t) = \sum_{j=1}^{N_{\text{pre}}} W_{ij} \sum_{k} \delta(t - t_j^k)$: Total synaptic input current from presynaptic spikes at timestamps $\{t_j^k\}$.

### 1.2 Euler-Maruyama Discretization & Decay Factor
Integrating over a discrete sampling interval $\Delta t = t[n] - t[n-1]$ yields the discrete-time update equation:

$$V_i[n] = V_i[n-1] e^{-\Delta t / \tau_m} + \sum_{j=1}^{N_{\text{in}}} W_{ij} S_j[n]$$

Defining the discrete membrane decay factor:
$$\beta \triangleq e^{-\Delta t / \tau_m} \in (0, 1)$$

In synthesizable hardware (Tier 1 RTL) and bit-exact fixed-point C99 simulation (Tier 2), multiplication by non-integer $\beta$ is eliminated through an arithmetic shift-right approximation:

$$\beta V \approx V - \left(V \gg k_{\text{leak}}\right) = V \left(1 - 2^{-k_{\text{leak}}}\right)$$

Setting $k_{\text{leak}} = 3$ yields an effective decay factor:
$$\beta_{\text{eff}} = 1 - 2^{-3} = 0.875 \quad \implies \quad \tau_{\text{eff}} = -\frac{\Delta t}{\ln(0.875)} \approx 7.49 \cdot \Delta t$$

### 1.3 Threshold Crossing, Firing, and Hard Reset
The spike emission event $S_i[n] \in \{0, 1\}$ is defined by the Heaviside step operator $\Theta(\cdot)$:

$$S_i[n] = \Theta\left(V_i[n] - V_{\text{th}}\right) = \begin{cases} 1, & \text{if } V_i[n] \ge V_{\text{th}} \\ 0, & \text{if } V_i[n] < V_{\text{th}} \end{cases}$$

Upon firing ($S_i[n] = 1$), the membrane potential undergoes a **hard zero reset**:

$$V_i[n] \leftarrow V_i[n] \cdot \left(1 - S_i[n]\right) + V_{\text{reset}} \cdot S_i[n]$$

---

## 2. Gradient Backpropagation via Fast-Sigmoid Surrogate

Because $\frac{\partial \Theta(x)}{\partial x} = \delta(x)$ vanishes almost everywhere and is non-differentiable at $x = 0$, backpropagation through time (BPTT) requires continuous surrogate gradient estimators.

We formulate the **fast-sigmoid surrogate derivative**:

$$\sigma_k(x) = \frac{x}{1 + k |x|}$$

$$\frac{\partial S}{\partial V} \approx \sigma_k'(V - V_{\text{th}}) = \frac{1}{\left(1 + k |V - V_{\text{th}}|\right)^2}$$

Where $k = 25.0$ defines the curvature steepness. The total error gradient with respect to synaptic weight $W_{ij}$ over a temporal horizon of length $T$ satisfies:

$$\frac{\partial \mathcal{L}}{\partial W_{ij}} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial S_i[t]} \frac{\partial S_i[t]}{\partial V_i[t]} \frac{\partial V_i[t]}{\partial W_{ij}} + \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial V_i[t+1]} \frac{\partial V_i[t+1]}{\partial V_i[t]} \frac{\partial V_i[t]}{\partial W_{ij}}$$

Where:
$$\frac{\partial V_i[t+1]}{\partial V_i[t]} = \beta \left(1 - S_i[t]\right) - V_{\text{th}} \frac{\partial S_i[t]}{\partial V_i[t]}$$

---

## 3. Biophysical Spike-Timing-Dependent Plasticity (STDP)

### 3.1 Continuous-Time Asymmetric Hebbian Formulation
Synaptic weight modification $\Delta W_{ij}$ depends on the precise temporal offset $\Delta t = t_i^{\text{post}} - t_j^{\text{pre}}$ between pre- and post-synaptic spike events:

$$\Delta W_{ij}(\Delta t) = \begin{cases} A_+ \exp\left(-\frac{\Delta t}{\tau_+}\right), & \text{if } \Delta t > 0 \quad (\text{Long-Term Potentiation, LTP}) \\ -A_- \exp\left(\frac{\Delta t}{\tau_-}\right), & \text{if } \Delta t < 0 \quad (\text{Long-Term Depression, LTD}) \end{cases}$$

Where $A_+, A_- > 0$ denote maximal synaptic modification amplitudes and $\tau_+, \tau_-$ are plasticity time constants.

### 3.2 Online Presynaptic and Postsynaptic Synaptic Traces
To compute STDP locally without storing complete spike history buffers, each neuron maintains continuous decaying synaptic traces $x_j(t)$ and $y_i(t)$:

$$\frac{d x_j(t)}{dt} = -\frac{x_j(t)}{\tau_+} + \sum_{k} \delta(t - t_j^k)$$

$$\frac{d y_i(t)}{dt} = -\frac{y_i(t)}{\tau_-} + \sum_{m} \delta(t - t_i^m)$$

At each pre-synaptic spike event at $t_j^k$:
$$W_{ij} \leftarrow W_{ij} - A_- \cdot y_i\left(t_j^k\right)$$

At each post-synaptic spike event at $t_i^m$:
$$W_{ij} \leftarrow W_{ij} + A_+ \cdot x_j\left(t_i^m\right)$$

### 3.3 Hardware-Synthesizable Bounded Rectification
In the synthesizable Verilog core (`implementations/v1_synthesizable_rtl_verilog/stdp_weight_updater.v`), synaptic weights are constrained within signed INT8 limits $[-128, +127]$:

$$W_{ij}[t] = \text{clip}\left(W_{ij}[t-1] + \Delta W_{ij}, -128, +127\right)$$

Ensuring that runaway LTP does not induce pathological bursting or destabilize recurrent firing dynamics.

---

## 4. Spike-Driven FlashAttention (SD-FlashAttention)

### 4.1 Rejection of Quadratic Softmax Attention
In standard transformer architectures, Self-Attention computes:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

This requires:
1. $O(N^2 d_k)$ floating-point multiplications for $\mathbf{Q} \mathbf{K}^T$.
2. Off-chip memory write-back of the $N \times N$ attention matrix $\mathbf{S}$.
3. Transcendental exponential evaluation $\exp(S_{ij})$ and normalization across the sequence length $N$.

### 4.2 Sparse Ternary Attention Algebra
Spike-Driven FlashAttention reformulates attention through ternary spike matrices:

$$\mathbf{S}_Q, \mathbf{S}_K \in \{-1, 0, +1\}^{N \times d_k}$$

The attention score $\mathbf{A}[i, j]$ is computed via sparse accumulation:

$$\mathbf{A}[i, j] = \sum_{d=1}^{d_k} S_Q[i, d] \cdot S_K[j, d]$$

Because $S_Q, S_K$ are ternary, the product $S_Q[i, d] \cdot S_K[j, d] \in \{-1, 0, +1\}$ requires **zero hardware multipliers**:

$$S_Q \cdot S_K = \begin{cases} +1, & (S_Q = 1 \land S_K = 1) \lor (S_Q = -1 \land S_K = -1) \\ -1, & (S_Q = 1 \land S_K = -1) \lor (S_Q = -1 \land S_K = 1) \\ 0, & S_Q = 0 \lor S_K = 0 \end{cases}$$

### 4.3 Computational Complexity & Memory Bandwidth Bounds

| Architecture | Computational Complexity | Multiplier Operations | Off-Chip SRAM Access |
| :--- | :---: | :---: | :---: |
| **Standard Softmax FlashAttention** | $\mathcal{O}(N^2 d_k)$ | Dense FP16 / BF16 MACs | $\mathcal{O}(N \cdot d_k)$ |
| **Spike-Driven FlashAttention** | $\mathcal{O}\left((1 - s_Q)(1 - s_K) N^2 d_k\right)$ | **0 Multipliers** (Add/Sub Only) | $\mathcal{O}\left(N_{\text{active}} \cdot d_k\right)$ |

When event sparsity satisfies $s_Q, s_K \ge 0.85$, the number of active synaptic additions is reduced by:

$$\text{Operation Reduction} = 1 - (1 - 0.85)^2 = 1 - 0.0225 = \mathbf{97.75\%}$$

---

## 5. VLSI Energy Scaling & Roofline Analysis

### 5.1 Dynamic Toggle Energy
The dynamic power consumption of digital CMOS gates is given by:

$$P_{\text{dyn}} = \alpha \cdot C_{\text{load}} \cdot V_{\text{dd}}^2 \cdot f_{\text{clk}}$$

Where $\alpha$ is the gate switching activity factor. In synchronous dense accelerators (e.g., systolic arrays), $\alpha \approx 0.7 - 0.9$ because all MAC units toggle every clock cycle. In **ES-FA**, clock gating and sparse event scheduling constrain $\alpha \le 0.08 - 0.15$:

$$\frac{P_{\text{ES-FA}}}{P_{\text{Systolic}}} = \frac{\alpha_{\text{sparse}}}{\alpha_{\text{dense}}} \approx \frac{0.10}{0.80} = \mathbf{0.125} \quad (8\times \text{ power reduction})$$

### 5.2 Synaptic Operation (SOP) vs. Floating-Point MAC Energy
At 28nm TSMC CMOS technology node ($V_{\text{dd}} = 0.9\text{ V}$):

| Operation | Datapath Width | Energy per Operation | Relative Cost |
| :--- | :---: | :---: | :---: |
| **FP32 Multiply-Accumulate (MAC)** | 32-bit | $4.60\text{ pJ}$ | $10.9\times$ |
| **FP16 Multiply-Accumulate (MAC)** | 16-bit | $1.10\text{ pJ}$ | $2.6\times$ |
| **INT8 MAC (Standard TPU/NPU)** | 8-bit | $0.20\text{ pJ}$ | $0.48\times$ |
| **ES-FA Synaptic Operation (SOP)** | INT8 Add/Sub + Shift Leak | **$0.038\text{ pJ}$** | **$0.09\times$** |
