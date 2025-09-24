#!/usr/bin/env python3
"""
Basic QPCA Example

This script demonstrates how to use the QPCA implementation with
sample data and compares classical and quantum approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the qpca module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qpca import QPCA, compare_methods


def generate_sample_data(n_samples: int = 100, n_features: int = 4, 
                        noise_level: float = 0.1) -> np.ndarray:
    """
    Generate sample data with known principal components.
    
    Args:
        n_samples: Number of data points
        n_features: Number of features (dimensions)
        noise_level: Amount of noise to add
        
    Returns:
        Generated data matrix
    """
    # Create data with clear principal components
    np.random.seed(42)
    
    # Generate data along two main directions
    t = np.linspace(0, 2 * np.pi, n_samples)
    
    # Primary component (strongest signal)
    component1 = np.column_stack([
        np.cos(t),
        np.sin(t),
        0.5 * np.cos(t),
        0.3 * np.sin(t)
    ])
    
    # Secondary component (weaker signal)
    component2 = np.column_stack([
        0.2 * np.sin(2 * t),
        0.2 * np.cos(2 * t),
        np.sin(2 * t),
        np.cos(2 * t)
    ])
    
    # Combine components
    data = component1 + 0.5 * component2
    
    # Add noise
    noise = noise_level * np.random.randn(n_samples, n_features)
    data += noise
    
    # Ensure we have the right number of features
    if n_features != 4:
        if n_features < 4:
            data = data[:, :n_features]
        else:
            # Pad with zeros or replicate columns
            extra_cols = n_features - 4
            padding = np.zeros((n_samples, extra_cols))
            data = np.column_stack([data, padding])
    
    return data


def plot_results(original_data: np.ndarray, results: dict, 
                save_path: str = None):
    """
    Plot PCA results for comparison.
    
    Args:
        original_data: Original high-dimensional data
        results: Results from compare_methods
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('QPCA Results Comparison', fontsize=16)
    
    # Original data (first two dimensions)
    axes[0, 0].scatter(original_data[:, 0], original_data[:, 1], 
                      alpha=0.7, c='blue')
    axes[0, 0].set_title('Original Data (First 2 Dimensions)')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Classical PCA results
    if 'classical' in results:
        classical_data = results['classical']['transformed_data']
        axes[0, 1].scatter(classical_data[:, 0], classical_data[:, 1], 
                          alpha=0.7, c='red')
        axes[0, 1].set_title('Classical PCA')
        axes[0, 1].set_xlabel('PC 1')
        axes[0, 1].set_ylabel('PC 2')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Variational QPCA results (if available)
    if 'variational' in results:
        var_data = results['variational']['transformed_data']
        axes[1, 0].scatter(var_data[:, 0], var_data[:, 1], 
                          alpha=0.7, c='green')
        axes[1, 0].set_title('Variational QPCA')
        axes[1, 0].set_xlabel('PC 1')
        axes[1, 0].set_ylabel('PC 2')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Variational QPCA\n(Requires Qiskit)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Variational QPCA (Not Available)')
    
    # Eigenvalue comparison
    eigenvalues_plot = axes[1, 1]
    if 'classical' in results:
        classical_eigs = results['classical']['eigenvalues']
        eigenvalues_plot.bar(range(len(classical_eigs)), classical_eigs, 
                           alpha=0.7, label='Classical PCA', color='red')
    
    if 'variational' in results:
        var_eigs = results['variational']['eigenvalues']
        x_pos = np.arange(len(var_eigs)) + 0.4
        eigenvalues_plot.bar(x_pos, var_eigs, alpha=0.7, 
                           label='Variational QPCA', color='green', width=0.4)
    
    eigenvalues_plot.set_title('Eigenvalues Comparison')
    eigenvalues_plot.set_xlabel('Principal Component')
    eigenvalues_plot.set_ylabel('Eigenvalue (Explained Variance)')
    eigenvalues_plot.legend()
    eigenvalues_plot.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    """Main example function."""
    print("=" * 60)
    print("QUANTUM PRINCIPAL COMPONENT ANALYSIS (QPCA) EXAMPLE")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    n_samples = 100
    n_features = 4
    data = generate_sample_data(n_samples=n_samples, n_features=n_features)
    
    print(f"   Data shape: {data.shape}")
    print(f"   Data mean: {np.mean(data, axis=0)}")
    print(f"   Data std: {np.std(data, axis=0)}")
    
    # Compare different PCA methods
    print("\n2. Comparing PCA methods...")
    results = compare_methods(data, n_components=2)
    
    print(f"   Available methods: {list(results.keys())}")
    
    # Display results
    print("\n3. Results:")
    for method, result in results.items():
        print(f"\n   {method.upper()} PCA:")
        print(f"   - Eigenvalues: {result['eigenvalues']}")
        print(f"   - Explained variance ratio: {result['eigenvalues'] / np.sum(result['eigenvalues'])}")
        print(f"   - Principal components shape: {result['components'].shape}")
        print(f"   - Transformed data shape: {result['transformed_data'].shape}")
    
    # Create visualization
    print("\n4. Creating visualization...")
    try:
        plot_results(data, results, save_path='qpca_results.png')
    except Exception as e:
        print(f"   Could not create plot: {e}")
        print("   (This is normal if matplotlib is not available)")
    
    # Demonstrate individual QPCA usage
    print("\n5. Individual QPCA usage example:")
    
    # Classical PCA
    print("\n   Classical PCA:")
    classical_qpca = QPCA(method='classical', n_components=2)
    classical_result = classical_qpca.fit_transform(data)
    print(f"   - Fitted and transformed data shape: {classical_result.shape}")
    print(f"   - Components shape: {classical_qpca.components_.shape}")
    
    # Try Variational QPCA
    try:
        print("\n   Variational QPCA:")
        n_qubits = int(np.ceil(np.log2(n_features)))
        var_qpca = QPCA(method='variational', n_components=2, n_qubits=n_qubits)
        var_result = var_qpca.fit_transform(data)
        print(f"   - Fitted and transformed data shape: {var_result.shape}")
        print(f"   - Components shape: {var_qpca.components_.shape}")
    except ImportError:
        print("\n   Variational QPCA: Requires Qiskit (not installed)")
    except Exception as e:
        print(f"\n   Variational QPCA: Error - {e}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()