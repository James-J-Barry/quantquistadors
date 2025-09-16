#!/usr/bin/env python3
"""
Quick QPCA Validation Script

This script provides a quick validation that the QPCA algorithm works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from qpca import QuantumPCA, ClassicalPCA, generate_test_data, visualize_results

def quick_qpca_validation():
    """Quick validation of QPCA algorithm"""
    print("Quick QPCA Validation")
    print("=" * 40)
    
    # Generate test data
    print("Generating test data...")
    X, y = generate_test_data(n_samples=200, n_features=6, random_state=42)
    
    # Initialize both methods
    classical_pca = ClassicalPCA(n_components=3)
    quantum_pca = QuantumPCA(n_components=3)
    
    # Fit both methods
    print("Fitting Classical and Quantum PCA...")
    classical_pca.fit(X)
    quantum_pca.fit(X)
    
    # Compare results
    classical_var = classical_pca.explained_variance_ratio()
    quantum_var = quantum_pca.explained_variance_ratio()
    
    print("\nExplained Variance Comparison:")
    for i in range(3):
        print(f"PC{i+1}: Classical={classical_var[i]:.4f}, Quantum={quantum_var[i]:.4f}")
    
    # Check similarity
    similarity = np.corrcoef(classical_var, quantum_var)[0, 1]
    print(f"\nVariance Correlation: {similarity:.4f}")
    
    if similarity > 0.95:
        print("✓ SUCCESS: QPCA algorithm is working correctly!")
    else:
        print("⚠ WARNING: Results differ significantly")
    
    return classical_pca, quantum_pca, X, y

if __name__ == "__main__":
    classical_pca, quantum_pca, X, y = quick_qpca_validation()
    
    print("\nGenerating comparison plot...")
    fig = visualize_results(classical_pca, quantum_pca, X, y)
    fig.savefig('/tmp/qpca_quick_validation.png', dpi=150, bbox_inches='tight')
    print("Plot saved to /tmp/qpca_quick_validation.png")
    
    # Show reconstruction errors
    classical_error = classical_pca.reconstruction_error(X)
    quantum_error = quantum_pca.reconstruction_error(X)
    print(f"\nReconstruction Errors:")
    print(f"Classical: {classical_error:.6f}")
    print(f"Quantum:   {quantum_error:.6f}")
    print(f"Difference: {abs(quantum_error - classical_error):.6f}")
    
    plt.show()