# Mathematical Foundations of QPCA

## Table of Contents
1. [Classical PCA Mathematics](#classical-pca-mathematics)
2. [Quantum State Representation](#quantum-state-representation)  
3. [Quantum PCA Algorithm](#quantum-pca-algorithm)
4. [Complexity Analysis](#complexity-analysis)
5. [Theoretical Guarantees](#theoretical-guarantees)

## Classical PCA Mathematics

### Problem Statement
Given a dataset X ∈ ℝ^(n×d) with n samples and d features, find the k-dimensional subspace that best approximates the data in the least-squares sense.

### Mathematical Formulation

#### Covariance Matrix
```
C = (1/n) X^T X
```
where X is assumed to be centered (mean = 0).

#### Eigendecomposition
```
C = V Λ V^T
```
where:
- V = [v₁, v₂, ..., vₐ] contains eigenvectors (principal components)
- Λ = diag(λ₁, λ₂, ..., λₐ) contains eigenvalues
- λ₁ ≥ λ₂ ≥ ... ≥ λₐ ≥ 0

#### Dimensionality Reduction
The projection onto the first k principal components is:
```
Y = X V_k
```
where V_k = [v₁, v₂, ..., vₖ] ∈ ℝ^(d×k).

#### Reconstruction
```
X̂ = Y V_k^T = X V_k V_k^T
```

#### Reconstruction Error
```
E = ||X - X̂||_F² = Σᵢ₌ₖ₊₁ᵈ λᵢ
```

### Optimization Perspective

PCA can be formulated as the optimization problem:
```
min_{V_k} ||X - X V_k V_k^T||_F²
subject to: V_k^T V_k = I_k
```

Solution: V_k contains the first k eigenvectors of X^T X.

## Quantum State Representation

### Quantum State Encoding

A classical vector x ∈ ℝᵈ is encoded as a quantum state:
```
|x⟩ = (1/||x||) Σⱼ₌₁ᵈ xⱼ |j⟩
```

where {|j⟩} forms an orthonormal computational basis.

### Density Matrix

For a dataset {x₁, x₂, ..., xₙ}, the quantum covariance matrix is:
```
ρ = (1/n) Σᵢ₌₁ⁿ |xᵢ⟩⟨xᵢ|
```

### Properties

1. **Hermitian**: ρ = ρ†
2. **Positive semidefinite**: ⟨ψ|ρ|ψ⟩ ≥ 0 for all |ψ⟩
3. **Unit trace**: Tr(ρ) = 1 (after normalization)

## Quantum PCA Algorithm

### Algorithm Overview

1. **State Preparation**: Prepare quantum states |xᵢ⟩ for each data point
2. **Density Matrix Construction**: Form ρ = (1/n) Σᵢ |xᵢ⟩⟨xᵢ|
3. **Quantum Phase Estimation**: Extract eigenvalues and eigenvectors
4. **Measurement**: Obtain classical information about principal components

### Detailed Steps

#### Step 1: Quantum State Preparation
For each data vector xᵢ ∈ ℝᵈ:
```
|xᵢ⟩ = (1/||xᵢ||) Σⱼ₌₁ᵈ xᵢⱼ |j⟩
```

**Circuit Complexity**: O(d) gates per state (using amplitude loading)

#### Step 2: Density Matrix Construction
The density matrix representing the dataset:
```
ρ = (1/n) Σᵢ₌₁ⁿ |xᵢ⟩⟨xᵢ| ∈ ℂᵈˣᵈ
```

**Eigendecomposition**:
```
ρ = Σⱼ₌₁ᵈ λⱼ |vⱼ⟩⟨vⱼ|
```

#### Step 3: Quantum Phase Estimation (QPE)

**QPE Circuit**:
```
|0⟩^⊗m ⊗ |ψ⟩ → |λ̃⟩ ⊗ |v_λ⟩
```

where:
- m = O(log(1/ε)) qubits for precision ε
- |ψ⟩ is the input state
- |λ̃⟩ encodes the eigenvalue estimate
- |v_λ⟩ is the corresponding eigenvector

**Unitary Operation**:
For eigenvalue estimation, we need a unitary U such that:
```
U |vⱼ⟩ = e^{2πi λⱼ t} |vⱼ⟩
```

This can be achieved using:
```
U = Σⱼ e^{2πi λⱼ t} |vⱼ⟩⟨vⱼ|
```

#### Step 4: Principal Component Extraction

After QPE, we obtain:
- Eigenvalues λⱼ with precision ε
- Access to eigenstates |vⱼ⟩

**Amplitude Amplification** can be used to increase the probability of measuring states corresponding to large eigenvalues.

### Quantum Phase Estimation Details

#### Standard QPE

For a unitary U with eigenvalue e^{2πiφ}:

1. **Initialization**: |0⟩^⊗m ⊗ |ψ⟩
2. **Hadamard gates**: Apply H^⊗m to ancilla qubits
3. **Controlled unitaries**: Apply controlled-U^{2^j} operations
4. **Inverse QFT**: Apply QFT† to ancilla qubits
5. **Measurement**: Measure ancilla to get φ̃ ≈ φ

**Success probability**: High for well-separated eigenvalues

#### Variational QPE (for NISQ devices)

Use a parameterized quantum circuit:
```
|ψ(θ)⟩ = U(θ) |0⟩
```

Optimize θ to minimize:
```
E(θ) = ⟨ψ(θ)| H |ψ(θ)⟩
```

where H encodes the problem Hamiltonian.

## Complexity Analysis

### Classical PCA Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Covariance computation | O(nd²) | O(d²) |
| Eigendecomposition | O(d³) | O(d²) |
| Projection | O(ndk) | O(nk) |
| **Total** | **O(nd² + d³)** | **O(d²)** |

### Quantum PCA Complexity

#### Theoretical Bounds (Fault-Tolerant)

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| State preparation | O(nd polylog(d)) | O(log d) |
| QPE | O(polylog(nd)) | O(log d) |
| **Total** | **O(polylog(nd))** | **O(log d)** |

#### Conditions for Speedup

1. **Efficient quantum data access**: Data must be available in quantum form
2. **Low-rank structure**: ρ has rapidly decaying eigenvalues
3. **Quantum output**: Application needs only quantum state output

#### NISQ Implementation Complexity

For variational approaches:
- **Time**: O(P × M × T) where P = parameters, M = measurements, T = circuit depth
- **Space**: O(n_qubits) quantum memory

### Speedup Analysis

**Polynomial speedup**: For data with specific structure
**Exponential speedup**: Possible for quantum data or with quantum subroutines

## Theoretical Guarantees

### Approximation Quality

#### Classical PCA
For rank-k approximation with eigenvalues λ₁ ≥ ... ≥ λₐ:

**Eckart-Young Theorem**:
```
||X - X̂||₂ = λₖ₊₁
||X - X̂||_F² = Σᵢ₌ₖ₊₁ᵈ λᵢ
```

#### Quantum PCA
With precision ε in QPE:

**Eigenvalue approximation**:
```
|λ̃ⱼ - λⱼ| ≤ ε
```

**State fidelity**:
```
|⟨vⱼ|ṽⱼ⟩|² ≥ 1 - O(ε)
```

### Error Propagation

#### Quantum Noise Effects

1. **Decoherence**: T₁, T₂ times affect fidelity
2. **Gate errors**: Each gate introduces error ~10⁻³ (current NISQ)
3. **Measurement errors**: Finite sampling introduces statistical noise

#### Error Bounds

For ε-approximate QPE with polynomial-time classical post-processing:
```
||ρ̃ - ρ||₁ ≤ ε
```

where ρ̃ is the reconstructed density matrix.

### Sample Complexity

#### Classical PCA
For (ε,δ)-approximation:
- **Samples needed**: O(k/ε²) for rank-k matrix
- **Confidence**: 1-δ

#### Quantum PCA
For quantum algorithms with quantum data access:
- **Query complexity**: O(polylog(n,d)/ε²)
- **Depends on**: Matrix condition number, desired precision

## Implementation Considerations

### Quantum Circuit Design

#### State Preparation Circuits
```
Data encoding: |0⟩ → |x⟩ = Σⱼ αⱼ |j⟩
```

Methods:
1. **Amplitude loading**: O(d) gates
2. **QRAM access**: O(log d) time, requires quantum memory
3. **Variational encoding**: Parameterized circuits

#### Eigenvalue Estimation Circuits

**Phase kickback mechanism**:
```
|φ⟩ ⊗ |vⱼ⟩ → e^{iφλⱼ} |φ⟩ ⊗ |vⱼ⟩
```

**Controlled evolution**:
```
|j⟩ ⊗ |ψ⟩ → |j⟩ ⊗ U^{2^j} |ψ⟩
```

### Practical Algorithms

#### Variational Quantum Eigensolver (VQE) approach
```
min_θ ⟨ψ(θ)| H |ψ(θ)⟩
```

where H is the Hamiltonian encoding the PCA problem.

#### Quantum Approximate Optimization Algorithm (QAOA)
For discrete optimization formulations of PCA.

## References and Further Reading

### Foundational Papers
1. Lloyd, S., Mohseni, M., & Rebentrost, P. (2013). Quantum principal component analysis. *Nature Physics*, 10(9), 631-633.
2. Kerenidis, I., & Prakash, A. (2017). Quantum recommendation systems. *arXiv preprint arXiv:1603.08675*.

### Recent Developments
3. Bravo-Prieto, C., et al. (2019). Variational quantum linear solver. *arXiv preprint arXiv:1909.05820*.
4. Huang, H. Y., et al. (2021). Information-theoretic bounds on quantum advantage in machine learning. *Physical Review Letters*, 126(19), 190505.

### Classical References
5. Jolliffe, I. T. (2002). Principal component analysis. Springer.
6. Trefethen, L. N., & Bau III, D. (1997). Numerical linear algebra. SIAM.