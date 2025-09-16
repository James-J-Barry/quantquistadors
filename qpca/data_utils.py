"""
Data utilities for generating test datasets and visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler

try:
    import seaborn as sns
except ImportError:
    sns = None


def generate_test_data(n_samples=1000, n_features=10, n_informative=5, 
                      n_redundant=2, noise=0.1, random_state=42):
    """
    Generate synthetic test data for PCA analysis
    
    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        noise: Amount of noise to add
        random_state: Random seed for reproducibility
        
    Returns:
        X: Feature matrix
        y: Target labels (for visualization)
    """
    # Ensure valid feature counts
    n_informative = min(n_informative, n_features - 1)
    n_redundant = min(n_redundant, n_features - n_informative)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        flip_y=noise,  # Use flip_y instead of noise parameter
        random_state=random_state
    )
    
    return X, y


def generate_correlated_data(n_samples=1000, n_features=6, correlation_strength=0.8, 
                           random_state=42):
    """
    Generate data with known correlation structure for PCA testing
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        correlation_strength: Strength of correlation between features
        random_state: Random seed
        
    Returns:
        X: Correlated feature matrix
        true_components: Known principal components
    """
    np.random.seed(random_state)
    
    # Generate base uncorrelated components
    n_components = min(3, n_features)
    base_components = np.random.randn(n_samples, n_components)
    
    # Create correlation matrix
    corr_matrix = np.eye(n_features)
    for i in range(n_features):
        for j in range(i+1, n_features):
            corr_matrix[i, j] = correlation_strength ** abs(i - j)
            corr_matrix[j, i] = corr_matrix[i, j]
    
    # Generate correlated features
    X = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=corr_matrix,
        size=n_samples
    )
    
    # True principal components (for validation)
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    idx = np.argsort(eigenvals)[::-1]
    true_components = eigenvecs[:, idx]
    
    return X, true_components


def generate_high_dimensional_data(n_samples=500, n_features=50, intrinsic_dim=3,
                                 noise_level=0.1, random_state=42):
    """
    Generate high-dimensional data with low intrinsic dimensionality
    
    Args:
        n_samples: Number of samples
        n_features: Number of features (high-dimensional)
        intrinsic_dim: True intrinsic dimensionality
        noise_level: Amount of noise to add
        random_state: Random seed
        
    Returns:
        X: High-dimensional data matrix
        true_embedding: True low-dimensional embedding
    """
    np.random.seed(random_state)
    
    # Generate low-dimensional embedding
    true_embedding = np.random.randn(n_samples, intrinsic_dim)
    
    # Random projection to high dimensions
    projection_matrix = np.random.randn(intrinsic_dim, n_features)
    
    # Project to high dimensions
    X = true_embedding @ projection_matrix
    
    # Add noise
    X += noise_level * np.random.randn(n_samples, n_features)
    
    return X, true_embedding


def visualize_results(classical_pca, quantum_pca, X, y=None, title_suffix=""):
    """
    Visualize and compare Classical vs Quantum PCA results
    
    Args:
        classical_pca: Fitted classical PCA model
        quantum_pca: Fitted quantum PCA model  
        X: Original data
        y: Optional labels for coloring
        title_suffix: Additional text for plot titles
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Transform data with both methods
    X_classical = classical_pca.transform(X)
    X_quantum = quantum_pca.transform(X)
    
    # Plot 1: Original data (first 2 features)
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
    axes[0, 0].set_title(f'Original Data (First 2 Features){title_suffix}')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    
    # Plot 2: Classical PCA projection
    axes[0, 1].scatter(X_classical[:, 0], X_classical[:, 1], c=y, cmap='viridis', alpha=0.6)
    axes[0, 1].set_title(f'Classical PCA Projection{title_suffix}')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    
    # Plot 3: Quantum PCA projection  
    axes[0, 2].scatter(X_quantum[:, 0], X_quantum[:, 1], c=y, cmap='viridis', alpha=0.6)
    axes[0, 2].set_title(f'Quantum PCA Projection{title_suffix}')
    axes[0, 2].set_xlabel('PC1')
    axes[0, 2].set_ylabel('PC2')
    
    # Plot 4: Explained variance comparison
    classical_var = classical_pca.explained_variance_ratio()
    quantum_var = quantum_pca.explained_variance_ratio()
    
    x_pos = np.arange(min(len(classical_var), len(quantum_var)))
    width = 0.35
    
    axes[1, 0].bar(x_pos - width/2, classical_var[:len(x_pos)], width, 
                   label='Classical PCA', alpha=0.8)
    axes[1, 0].bar(x_pos + width/2, quantum_var[:len(x_pos)], width,
                   label='Quantum PCA', alpha=0.8)
    axes[1, 0].set_title(f'Explained Variance Ratio{title_suffix}')
    axes[1, 0].set_xlabel('Principal Component')
    axes[1, 0].set_ylabel('Explained Variance Ratio')
    axes[1, 0].legend()
    
    # Plot 5: Cumulative explained variance
    classical_cumvar = np.cumsum(classical_var)
    quantum_cumvar = np.cumsum(quantum_var)
    
    axes[1, 1].plot(classical_cumvar, 'o-', label='Classical PCA', linewidth=2)
    axes[1, 1].plot(quantum_cumvar, 's-', label='Quantum PCA', linewidth=2)
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Variance')
    axes[1, 1].set_title(f'Cumulative Explained Variance{title_suffix}')
    axes[1, 1].set_xlabel('Number of Components')
    axes[1, 1].set_ylabel('Cumulative Explained Variance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Reconstruction error comparison
    classical_error = classical_pca.reconstruction_error(X)
    quantum_error = quantum_pca.reconstruction_error(X)
    
    methods = ['Classical PCA', 'Quantum PCA']
    errors = [classical_error, quantum_error]
    colors = ['blue', 'red']
    
    bars = axes[1, 2].bar(methods, errors, color=colors, alpha=0.7)
    axes[1, 2].set_title(f'Reconstruction Error Comparison{title_suffix}')
    axes[1, 2].set_ylabel('Mean Squared Error')
    
    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{error:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def plot_eigenvalue_spectrum(classical_pca, quantum_pca, title_suffix=""):
    """
    Plot eigenvalue spectrum comparison
    
    Args:
        classical_pca: Fitted classical PCA model
        quantum_pca: Fitted quantum PCA model
        title_suffix: Additional text for plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Classical eigenvalues
    classical_eigenvals = classical_pca.pca.explained_variance_
    ax1.plot(classical_eigenvals, 'o-', linewidth=2, markersize=8)
    ax1.set_title(f'Classical PCA Eigenvalue Spectrum{title_suffix}')
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Quantum eigenvalues
    quantum_eigenvals = quantum_pca.eigenvalues[:len(classical_eigenvals)]
    ax2.plot(quantum_eigenvals, 's-', color='red', linewidth=2, markersize=8)
    ax2.set_title(f'Quantum PCA Eigenvalue Spectrum{title_suffix}')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig


def compare_component_similarity(classical_pca, quantum_pca):
    """
    Compare similarity between classical and quantum principal components
    
    Args:
        classical_pca: Fitted classical PCA model
        quantum_pca: Fitted quantum PCA model
        
    Returns:
        Dictionary with similarity metrics
    """
    classical_components = classical_pca.get_components()
    quantum_components = quantum_pca.get_components()
    
    n_components = min(classical_components.shape[0], quantum_components.shape[0])
    
    # Calculate cosine similarity between corresponding components
    similarities = []
    for i in range(n_components):
        # Cosine similarity
        dot_product = np.dot(classical_components[i], quantum_components[i])
        norm_product = (np.linalg.norm(classical_components[i]) * 
                       np.linalg.norm(quantum_components[i]))
        similarity = abs(dot_product / norm_product)  # abs for direction invariance
        similarities.append(similarity)
    
    return {
        'component_similarities': similarities,
        'mean_similarity': np.mean(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities)
    }


def generate_performance_report(classical_pca, quantum_pca, X, 
                              execution_time_classical=None, 
                              execution_time_quantum=None):
    """
    Generate comprehensive performance comparison report
    
    Args:
        classical_pca: Fitted classical PCA model
        quantum_pca: Fitted quantum PCA model
        X: Original data
        execution_time_classical: Time taken for classical PCA
        execution_time_quantum: Time taken for quantum PCA
        
    Returns:
        Dictionary with performance metrics
    """
    # Calculate reconstruction errors
    classical_error = classical_pca.reconstruction_error(X)
    quantum_error = quantum_pca.reconstruction_error(X)
    
    # Component similarity
    similarity_metrics = compare_component_similarity(classical_pca, quantum_pca)
    
    # Explained variance
    classical_var = np.sum(classical_pca.explained_variance_ratio())
    quantum_var = np.sum(quantum_pca.explained_variance_ratio())
    
    # Quantum advantage metrics
    quantum_metrics = quantum_pca.quantum_advantage_metric(classical_pca)
    
    report = {
        'reconstruction_error': {
            'classical': classical_error,
            'quantum': quantum_error,
            'relative_difference': abs(quantum_error - classical_error) / classical_error
        },
        'explained_variance': {
            'classical': classical_var,
            'quantum': quantum_var,
            'difference': quantum_var - classical_var
        },
        'component_similarity': similarity_metrics,
        'quantum_advantage': quantum_metrics,
        'execution_time': {
            'classical': execution_time_classical,
            'quantum': execution_time_quantum,
            'speedup': (execution_time_classical / execution_time_quantum 
                       if execution_time_classical and execution_time_quantum else None)
        }
    }
    
    return report