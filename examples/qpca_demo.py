#!/usr/bin/env python3
"""
Quantum PCA Demonstration Script

This script demonstrates the Quantum Principal Component Analysis (QPCA) algorithm
and compares it with classical PCA to validate its effectiveness.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from qpca import QuantumPCA, ClassicalPCA, generate_test_data, visualize_results
from qpca.data_utils import (
    generate_correlated_data, 
    generate_high_dimensional_data,
    plot_eigenvalue_spectrum,
    generate_performance_report
)


def demonstrate_qpca_basic():
    """Basic demonstration of QPCA vs Classical PCA"""
    print("=" * 60)
    print("QUANTUM PCA DEMONSTRATION - Basic Test")
    print("=" * 60)
    
    # Generate test data
    print("Generating test data...")
    X, y = generate_test_data(n_samples=500, n_features=10, random_state=42)
    print(f"Data shape: {X.shape}")
    
    # Initialize both PCA methods
    n_components = 3
    classical_pca = ClassicalPCA(n_components=n_components)
    quantum_pca = QuantumPCA(n_components=n_components)
    
    # Fit and time both methods
    print("\nFitting Classical PCA...")
    start_time = time.time()
    classical_pca.fit(X)
    classical_time = time.time() - start_time
    print(f"Classical PCA fitting time: {classical_time:.4f} seconds")
    
    print("\nFitting Quantum PCA...")
    start_time = time.time()
    quantum_pca.fit(X)
    quantum_time = time.time() - start_time
    print(f"Quantum PCA fitting time: {quantum_time:.4f} seconds")
    
    # Compare results
    print("\n" + "="*40)
    print("RESULTS COMPARISON")
    print("="*40)
    
    # Explained variance ratios
    classical_var = classical_pca.explained_variance_ratio()
    quantum_var = quantum_pca.explained_variance_ratio()
    
    print("\nExplained Variance Ratios:")
    print("Component\tClassical\tQuantum\t\tDifference")
    print("-" * 50)
    for i in range(min(len(classical_var), len(quantum_var))):
        diff = abs(quantum_var[i] - classical_var[i])
        print(f"PC{i+1}\t\t{classical_var[i]:.4f}\t\t{quantum_var[i]:.4f}\t\t{diff:.4f}")
    
    # Reconstruction errors
    classical_error = classical_pca.reconstruction_error(X)
    quantum_error = quantum_pca.reconstruction_error(X)
    
    print(f"\nReconstruction Errors:")
    print(f"Classical PCA: {classical_error:.6f}")
    print(f"Quantum PCA:   {quantum_error:.6f}")
    print(f"Relative difference: {abs(quantum_error - classical_error) / classical_error * 100:.2f}%")
    
    # Generate performance report
    report = generate_performance_report(
        classical_pca, quantum_pca, X, 
        classical_time, quantum_time
    )
    
    print(f"\nComponent Similarity:")
    print(f"Mean similarity: {report['component_similarity']['mean_similarity']:.4f}")
    print(f"Min similarity:  {report['component_similarity']['min_similarity']:.4f}")
    print(f"Max similarity:  {report['component_similarity']['max_similarity']:.4f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    fig1 = visualize_results(classical_pca, quantum_pca, X, y, " - Basic Test")
    fig1.savefig('/tmp/qpca_basic_comparison.png', dpi=150, bbox_inches='tight')
    
    fig2 = plot_eigenvalue_spectrum(classical_pca, quantum_pca, " - Basic Test")
    fig2.savefig('/tmp/qpca_eigenvalue_spectrum.png', dpi=150, bbox_inches='tight')
    
    print("Visualizations saved to /tmp/")
    plt.show()
    
    return report


def demonstrate_qpca_correlated_data():
    """Demonstrate QPCA on data with known correlation structure"""
    print("\n" + "=" * 60)
    print("QUANTUM PCA DEMONSTRATION - Correlated Data")
    print("=" * 60)
    
    # Generate correlated data
    print("Generating correlated test data...")
    X, true_components = generate_correlated_data(
        n_samples=400, n_features=8, correlation_strength=0.7, random_state=123
    )
    print(f"Data shape: {X.shape}")
    
    # Initialize PCA methods
    classical_pca = ClassicalPCA(n_components=4)
    quantum_pca = QuantumPCA(n_components=4)
    
    # Fit both methods
    print("Fitting both PCA methods...")
    classical_pca.fit(X)
    quantum_pca.fit(X)
    
    # Compare with true components
    classical_components = classical_pca.get_components()
    quantum_components = quantum_pca.get_components()
    
    print("\nComparing recovered components with true structure:")
    
    # Calculate similarity with true components
    true_similarities_classical = []
    true_similarities_quantum = []
    
    for i in range(min(4, true_components.shape[1])):
        # Classical similarity
        classical_sim = abs(np.dot(classical_components[i], true_components[:, i]))
        true_similarities_classical.append(classical_sim)
        
        # Quantum similarity  
        quantum_sim = abs(np.dot(quantum_components[i], true_components[:, i]))
        true_similarities_quantum.append(quantum_sim)
    
    print("Component\tClassical vs True\tQuantum vs True")
    print("-" * 50)
    for i in range(len(true_similarities_classical)):
        print(f"PC{i+1}\t\t{true_similarities_classical[i]:.4f}\t\t\t{true_similarities_quantum[i]:.4f}")
    
    # Visualize results
    fig = visualize_results(classical_pca, quantum_pca, X, title_suffix=" - Correlated Data")
    fig.savefig('/tmp/qpca_correlated_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'classical_true_similarity': np.mean(true_similarities_classical),
        'quantum_true_similarity': np.mean(true_similarities_quantum)
    }


def demonstrate_qpca_high_dimensional():
    """Demonstrate QPCA on high-dimensional data with low intrinsic dimensionality"""
    print("\n" + "=" * 60)
    print("QUANTUM PCA DEMONSTRATION - High-Dimensional Data")
    print("=" * 60)
    
    # Generate high-dimensional data
    print("Generating high-dimensional test data...")
    X, true_embedding = generate_high_dimensional_data(
        n_samples=300, n_features=25, intrinsic_dim=3, random_state=456
    )
    print(f"Data shape: {X.shape}")
    print(f"True intrinsic dimensionality: {true_embedding.shape[1]}")
    
    # Initialize PCA methods with automatic component selection
    classical_pca = ClassicalPCA()  # Will select components automatically
    quantum_pca = QuantumPCA()     # Will select components automatically
    
    # Fit both methods
    print("Fitting both PCA methods with automatic component selection...")
    classical_pca.fit(X)
    quantum_pca.fit(X)
    
    print(f"Classical PCA selected components: {classical_pca.pca.n_components_}")
    print(f"Quantum PCA selected components: {quantum_pca.n_components}")
    
    # Check if they captured the intrinsic dimensionality correctly
    classical_var_95 = np.argmax(np.cumsum(classical_pca.explained_variance_ratio()) >= 0.95) + 1
    quantum_var_95 = np.argmax(np.cumsum(quantum_pca.explained_variance_ratio()) >= 0.95) + 1
    
    print(f"\nComponents needed for 95% variance:")
    print(f"Classical PCA: {classical_var_95}")
    print(f"Quantum PCA:   {quantum_var_95}")
    print(f"True intrinsic dim: {true_embedding.shape[1]}")
    
    # Evaluate dimensionality reduction quality
    classical_transformed = classical_pca.transform(X)
    quantum_transformed = quantum_pca.transform(X)
    
    # Calculate reconstruction errors
    classical_error = classical_pca.reconstruction_error(X)
    quantum_error = quantum_pca.reconstruction_error(X)
    
    print(f"\nReconstruction Errors (full data):")
    print(f"Classical PCA: {classical_error:.6f}")
    print(f"Quantum PCA:   {quantum_error:.6f}")
    
    # Visualize results
    fig = visualize_results(classical_pca, quantum_pca, X, title_suffix=" - High-Dim Data")
    fig.savefig('/tmp/qpca_highdim_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'classical_components_95': classical_var_95,
        'quantum_components_95': quantum_var_95,
        'true_intrinsic_dim': true_embedding.shape[1],
        'reconstruction_error_classical': classical_error,
        'reconstruction_error_quantum': quantum_error
    }


def run_qpca_validation_suite():
    """Run comprehensive validation suite for QPCA"""
    print("\n" + "=" * 80)
    print("QUANTUM PCA VALIDATION SUITE")
    print("=" * 80)
    
    print("Running comprehensive validation of QPCA algorithm...")
    print("This will test the algorithm on multiple datasets and scenarios.\n")
    
    results = {}
    
    # Test 1: Basic functionality
    print("TEST 1: Basic QPCA functionality")
    results['basic_test'] = demonstrate_qpca_basic()
    
    # Test 2: Correlated data
    print("\nTEST 2: Correlated data structure")
    results['correlated_test'] = demonstrate_qpca_correlated_data()
    
    # Test 3: High-dimensional data
    print("\nTEST 3: High-dimensional data")
    results['highdim_test'] = demonstrate_qpca_high_dimensional()
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print("\n1. ALGORITHM CORRECTNESS:")
    basic_similarity = results['basic_test']['component_similarity']['mean_similarity']
    print(f"   - Mean component similarity with classical PCA: {basic_similarity:.4f}")
    if basic_similarity > 0.9:
        print("   ✓ EXCELLENT: Components very similar to classical PCA")
    elif basic_similarity > 0.7:
        print("   ✓ GOOD: Components reasonably similar to classical PCA")
    else:
        print("   ⚠ WARNING: Components differ significantly from classical PCA")
    
    print("\n2. STRUCTURE RECOVERY:")
    classical_true_sim = results['correlated_test']['classical_true_similarity']
    quantum_true_sim = results['correlated_test']['quantum_true_similarity']
    print(f"   - Classical PCA vs true structure: {classical_true_sim:.4f}")
    print(f"   - Quantum PCA vs true structure:   {quantum_true_sim:.4f}")
    if abs(quantum_true_sim - classical_true_sim) < 0.1:
        print("   ✓ EXCELLENT: Quantum PCA recovers structure as well as classical")
    else:
        print("   ⚠ Note: Some difference in structure recovery")
    
    print("\n3. DIMENSIONALITY REDUCTION:")
    true_dim = results['highdim_test']['true_intrinsic_dim']
    classical_dim = results['highdim_test']['classical_components_95']
    quantum_dim = results['highdim_test']['quantum_components_95']
    print(f"   - True intrinsic dimensionality: {true_dim}")
    print(f"   - Classical PCA identified:     {classical_dim}")
    print(f"   - Quantum PCA identified:       {quantum_dim}")
    if abs(quantum_dim - true_dim) <= abs(classical_dim - true_dim):
        print("   ✓ EXCELLENT: Quantum PCA dimensionality estimate as good as classical")
    else:
        print("   ⚠ Note: Classical PCA slightly better at dimensionality estimation")
    
    print("\n4. RECONSTRUCTION QUALITY:")
    basic_rel_error = results['basic_test']['reconstruction_error']['relative_difference']
    print(f"   - Relative reconstruction error difference: {basic_rel_error:.4f}")
    if basic_rel_error < 0.1:
        print("   ✓ EXCELLENT: Reconstruction quality very similar to classical")
    elif basic_rel_error < 0.3:
        print("   ✓ GOOD: Reconstruction quality reasonably similar to classical")
    else:
        print("   ⚠ WARNING: Reconstruction quality differs significantly")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    
    if (basic_similarity > 0.8 and basic_rel_error < 0.2 and 
        abs(quantum_true_sim - classical_true_sim) < 0.15):
        print("✓ SUCCESS: QPCA algorithm is working correctly!")
        print("  The quantum implementation produces results very similar to classical PCA")
        print("  while demonstrating the core quantum PCA concepts.")
    else:
        print("⚠ PARTIAL SUCCESS: QPCA algorithm is functional but shows some differences")
        print("  from classical PCA. This may be expected due to quantum implementation details.")
    
    print("\nThe QPCA implementation successfully demonstrates:")
    print("• Quantum-inspired eigenvalue decomposition")
    print("• Principal component extraction")
    print("• Data transformation and reconstruction")
    print("• Comparison metrics with classical methods")
    
    return results


if __name__ == "__main__":
    # Set up matplotlib for better display
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # Run the complete validation suite
    validation_results = run_qpca_validation_suite()
    
    print(f"\nAll visualizations have been saved to /tmp/")
    print("Demonstration complete!")