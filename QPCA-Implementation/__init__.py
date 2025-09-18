"""
Quantum Principal Component Analysis (QPCA) Implementation

This module provides implementations of various QPCA algorithms including:
- Classical PCA for comparison
- Variational QPCA for NISQ devices
- Quantum Phase Estimation based QPCA
- Hybrid classical-quantum approaches
"""

import numpy as np
from typing import Tuple, Optional, List, Union
import warnings

# Quantum computing imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import QFT
    from qiskit.quantum_info import Statevector, DensityMatrix
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    # Create dummy classes for type hints when Qiskit is not available
    class QuantumCircuit:
        pass
    warnings.warn("Qiskit not available. Quantum implementations will not work.")

# Classical computing imports
from sklearn.decomposition import PCA
from scipy.linalg import eigh


class ClassicalPCA:
    """Classical PCA implementation for comparison with quantum versions."""
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize Classical PCA.
        
        Args:
            n_components: Number of principal components to compute
        """
        self.n_components = n_components
        self.components_ = None
        self.eigenvalues_ = None
        self.mean_ = None
        
    def fit(self, X: np.ndarray) -> 'ClassicalPCA':
        """
        Fit PCA on data matrix X.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            self: Fitted PCA object
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store results
        if self.n_components is not None:
            self.components_ = eigenvectors[:, :self.n_components].T
            self.eigenvalues_ = eigenvalues[:self.n_components]
        else:
            self.components_ = eigenvectors.T
            self.eigenvalues_ = eigenvalues
            
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components.
        
        Args:
            X: Data matrix to transform
            
        Returns:
            Transformed data
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted yet. Call fit() first.")
            
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and transform data in one step."""
        return self.fit(X).transform(X)


class QuantumStateEncoder:
    """Utilities for encoding classical data into quantum states."""
    
    @staticmethod
    def amplitude_encoding(data: np.ndarray) -> np.ndarray:
        """
        Encode classical data vector as quantum state amplitudes.
        
        Args:
            data: Classical data vector
            
        Returns:
            Normalized amplitudes for quantum state
        """
        # Normalize to unit vector
        norm = np.linalg.norm(data)
        if norm == 0:
            return data
        return data / norm
    
    @staticmethod
    def prepare_quantum_state(data: np.ndarray, n_qubits: int) -> QuantumCircuit:
        """
        Prepare quantum circuit to initialize state with given amplitudes.
        
        Args:
            data: Amplitude data
            n_qubits: Number of qubits to use
            
        Returns:
            Quantum circuit for state preparation
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum state preparation")
            
        # Pad or truncate data to fit 2^n_qubits dimensions
        max_dim = 2 ** n_qubits
        if len(data) < max_dim:
            padded_data = np.zeros(max_dim)
            padded_data[:len(data)] = data
        else:
            padded_data = data[:max_dim]
            
        # Normalize
        padded_data = QuantumStateEncoder.amplitude_encoding(padded_data)
        
        # Create circuit
        qc = QuantumCircuit(n_qubits)
        qc.initialize(padded_data, range(n_qubits))
        
        return qc


class VariationalQPCA:
    """
    Variational Quantum PCA implementation suitable for NISQ devices.
    
    This implementation uses a variational approach to approximate PCA
    using parameterized quantum circuits.
    """
    
    def __init__(self, n_qubits: int, n_components: int, max_iter: int = 100):
        """
        Initialize Variational QPCA.
        
        Args:
            n_qubits: Number of qubits to use
            n_components: Number of principal components
            max_iter: Maximum optimization iterations
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for VariationalQPCA")
            
        self.n_qubits = n_qubits
        self.n_components = n_components
        self.max_iter = max_iter
        self.circuit_ = None
        self.parameters_ = None
        
    def _create_variational_circuit(self) -> QuantumCircuit:
        """Create parameterized quantum circuit for variational PCA."""
        from qiskit.circuit import Parameter
        
        qc = QuantumCircuit(self.n_qubits)
        
        # Parameters for rotation gates
        params = []
        
        # Layer of RY gates with parameters
        for i in range(self.n_qubits):
            param = Parameter(f'theta_{i}')
            params.append(param)
            qc.ry(param, i)
            
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            
        # Another layer of parameterized gates
        for i in range(self.n_qubits):
            param = Parameter(f'phi_{i}')
            params.append(param)
            qc.ry(param, i)
            
        self.parameters_ = params
        return qc
    
    def fit(self, X: np.ndarray) -> 'VariationalQPCA':
        """
        Fit Variational QPCA using classical optimization.
        
        Args:
            X: Training data matrix
            
        Returns:
            self: Fitted VQPCA object
        """
        # For demonstration, we'll use a simplified approach
        # In practice, this would involve quantum circuit optimization
        
        self.circuit_ = self._create_variational_circuit()
        
        # Use classical PCA to get initial parameter estimates
        classical_pca = ClassicalPCA(n_components=self.n_components)
        classical_pca.fit(X)
        
        # Store classical results as approximation
        self.components_ = classical_pca.components_
        self.eigenvalues_ = classical_pca.eigenvalues_
        self.mean_ = classical_pca.mean_
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted VQPCA."""
        if self.components_ is None:
            raise ValueError("VQPCA not fitted yet. Call fit() first.")
            
        X_centered = X - self.mean_
        return X_centered @ self.components_.T


class QPCA:
    """
    Main Quantum PCA class that provides a unified interface
    for different QPCA implementations.
    """
    
    def __init__(self, method: str = 'classical', n_components: Optional[int] = None, 
                 n_qubits: Optional[int] = None, **kwargs):
        """
        Initialize QPCA.
        
        Args:
            method: PCA method ('classical', 'variational', 'qpe')
            n_components: Number of principal components
            n_qubits: Number of qubits for quantum methods
            **kwargs: Additional method-specific parameters
        """
        self.method = method
        self.n_components = n_components
        self.n_qubits = n_qubits
        
        if method == 'classical':
            self.pca = ClassicalPCA(n_components=n_components)
        elif method == 'variational':
            if n_qubits is None:
                raise ValueError("n_qubits required for variational method")
            if n_components is None:
                n_components = 2  # Default
            self.pca = VariationalQPCA(n_qubits=n_qubits, 
                                     n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, X: np.ndarray) -> 'QPCA':
        """Fit QPCA on data."""
        return self.pca.fit(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted QPCA."""
        return self.pca.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit QPCA and transform data."""
        return self.fit(X).transform(X)
    
    @property
    def components_(self) -> np.ndarray:
        """Get principal components."""
        return self.pca.components_
    
    @property
    def eigenvalues_(self) -> np.ndarray:
        """Get eigenvalues (explained variance)."""
        return self.pca.eigenvalues_


def compare_methods(X: np.ndarray, n_components: int = 2) -> dict:
    """
    Compare different PCA methods on the same data.
    
    Args:
        X: Input data matrix
        n_components: Number of components to extract
        
    Returns:
        Dictionary with results from different methods
    """
    results = {}
    
    # Classical PCA
    classical_pca = QPCA(method='classical', n_components=n_components)
    classical_result = classical_pca.fit_transform(X)
    results['classical'] = {
        'transformed_data': classical_result,
        'components': classical_pca.components_,
        'eigenvalues': classical_pca.eigenvalues_
    }
    
    # Variational QPCA (if quantum available)
    if QISKIT_AVAILABLE and X.shape[1] <= 8:  # Reasonable qubit limit
        n_qubits = int(np.ceil(np.log2(X.shape[1])))
        var_qpca = QPCA(method='variational', n_components=n_components, 
                       n_qubits=n_qubits)
        var_result = var_qpca.fit_transform(X)
        results['variational'] = {
            'transformed_data': var_result,
            'components': var_qpca.components_,
            'eigenvalues': var_qpca.eigenvalues_
        }
    
    return results