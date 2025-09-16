"""
Basic tests for QPCA implementation
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qpca import QuantumPCA, ClassicalPCA, generate_test_data


def test_classical_pca_basic():
    """Test basic Classical PCA functionality"""
    X, _ = generate_test_data(n_samples=100, n_features=5, random_state=42)
    
    pca = ClassicalPCA(n_components=3)
    pca.fit(X)
    
    # Test transformation
    X_transformed = pca.transform(X)
    assert X_transformed.shape == (100, 3)
    
    # Test inverse transformation
    X_reconstructed = pca.inverse_transform(X_transformed)
    assert X_reconstructed.shape == X.shape
    
    # Test explained variance
    explained_var = pca.explained_variance_ratio()
    assert len(explained_var) == 3
    assert np.all(explained_var >= 0)
    assert np.all(explained_var <= 1)


def test_quantum_pca_basic():
    """Test basic Quantum PCA functionality"""
    X, _ = generate_test_data(n_samples=100, n_features=5, random_state=42)
    
    qpca = QuantumPCA(n_components=3)
    qpca.fit(X)
    
    # Test transformation
    X_transformed = qpca.transform(X)
    assert X_transformed.shape == (100, 3)
    
    # Test inverse transformation
    X_reconstructed = qpca.inverse_transform(X_transformed)
    assert X_reconstructed.shape == X.shape
    
    # Test explained variance
    explained_var = qpca.explained_variance_ratio()
    assert len(explained_var) == 3
    assert np.all(explained_var >= 0)
    assert np.all(explained_var <= 1)


def test_qpca_vs_classical_similarity():
    """Test that QPCA produces similar results to Classical PCA"""
    X, _ = generate_test_data(n_samples=200, n_features=6, random_state=123)
    
    # Fit both methods
    classical_pca = ClassicalPCA(n_components=3)
    quantum_pca = QuantumPCA(n_components=3)
    
    classical_pca.fit(X)
    quantum_pca.fit(X)
    
    # Compare explained variance ratios
    classical_var = classical_pca.explained_variance_ratio()
    quantum_var = quantum_pca.explained_variance_ratio()
    
    # Should be similar (within reasonable tolerance)
    for i in range(3):
        assert abs(classical_var[i] - quantum_var[i]) < 0.1, \
            f"Component {i}: Classical={classical_var[i]:.4f}, Quantum={quantum_var[i]:.4f}"
    
    # Compare reconstruction errors
    classical_error = classical_pca.reconstruction_error(X)
    quantum_error = quantum_pca.reconstruction_error(X)
    
    relative_error = abs(quantum_error - classical_error) / classical_error
    assert relative_error < 0.5, f"Reconstruction errors too different: {relative_error:.4f}"


def test_auto_component_selection():
    """Test automatic component selection"""
    X, _ = generate_test_data(n_samples=150, n_features=8, random_state=456)
    
    # Test with automatic component selection
    qpca = QuantumPCA()  # No n_components specified
    qpca.fit(X)
    
    # Should select some number of components automatically
    assert qpca.n_components > 0
    assert qpca.n_components <= X.shape[1]
    
    # Should explain at least 95% of variance
    total_explained = np.sum(qpca.explained_variance_ratio())
    assert total_explained >= 0.95


def test_data_scaling():
    """Test that data scaling works correctly"""
    # Create data with different scales
    X = np.random.randn(100, 4)
    X[:, 0] *= 1000  # Scale first feature
    X[:, 1] *= 0.001  # Scale second feature
    
    qpca = QuantumPCA(n_components=2)
    qpca.fit(X)
    
    # Should still work despite different scales
    X_transformed = qpca.transform(X)
    assert X_transformed.shape == (100, 2)
    
    # Check that scaler was applied
    assert qpca.scaler is not None


def test_quantum_advantage_metric():
    """Test quantum advantage metric calculation"""
    X, _ = generate_test_data(n_samples=100, n_features=5, random_state=789)
    
    classical_pca = ClassicalPCA(n_components=3)
    quantum_pca = QuantumPCA(n_components=3)
    
    classical_pca.fit(X)
    quantum_pca.fit(X)
    
    # Test quantum advantage metric
    metrics = quantum_pca.quantum_advantage_metric(classical_pca)
    
    assert 'quantum_components' in metrics
    assert 'classical_components' in metrics
    assert 'quantum_explained_variance' in metrics
    assert 'classical_explained_variance' in metrics
    assert 'eigenvalue_similarity' in metrics
    
    # Check values are reasonable
    assert metrics['quantum_components'] == 3
    assert metrics['classical_components'] == 3
    assert 0 <= metrics['quantum_explained_variance'] <= 1
    assert 0 <= metrics['classical_explained_variance'] <= 1


if __name__ == "__main__":
    # Run tests manually
    test_classical_pca_basic()
    print("✓ Classical PCA basic test passed")
    
    test_quantum_pca_basic()
    print("✓ Quantum PCA basic test passed")
    
    test_qpca_vs_classical_similarity()
    print("✓ QPCA vs Classical similarity test passed")
    
    test_auto_component_selection()
    print("✓ Auto component selection test passed")
    
    test_data_scaling()
    print("✓ Data scaling test passed")
    
    test_quantum_advantage_metric()
    print("✓ Quantum advantage metric test passed")
    
    print("\nAll tests passed! ✓")