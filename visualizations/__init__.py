"""
Visualization tools for QPCA results and analysis.

This module provides functions to visualize:
- Principal components
- Eigenvalue spectra  
- Data projections
- Comparison between classical and quantum methods
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")


def plot_principal_components(components: np.ndarray, 
                            feature_names: Optional[List[str]] = None,
                            title: str = "Principal Components",
                            save_path: Optional[str] = None) -> None:
    """
    Plot principal components as loading vectors.
    
    Args:
        components: Principal components matrix (n_components x n_features)
        feature_names: Names of features
        title: Plot title
        save_path: Path to save plot
    """
    n_components, n_features = components.shape
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    fig, axes = plt.subplots(1, min(n_components, 3), figsize=(4*min(n_components, 3), 4))
    if n_components == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i >= n_components:
            break
            
        # Plot loading vectors
        ax.bar(range(n_features), components[i], alpha=0.7)
        ax.set_title(f'Principal Component {i+1}')
        ax.set_xlabel('Features')
        ax.set_ylabel('Loading')
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feature_names, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_eigenvalue_spectrum(eigenvalues: np.ndarray, 
                           method_name: str = "PCA",
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> None:
    """
    Plot eigenvalue spectrum showing explained variance.
    
    Args:
        eigenvalues: Array of eigenvalues
        method_name: Name of the method
        title: Plot title
        save_path: Path to save plot
    """
    if title is None:
        title = f"{method_name} Eigenvalue Spectrum"
    
    # Calculate explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Individual eigenvalues
    ax1.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Eigenvalues')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    ax2.plot(range(1, len(eigenvalues) + 1), cumulative_variance, 'bo-', alpha=0.7)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add horizontal lines for common thresholds
    for threshold in [0.8, 0.9, 0.95]:
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, 
                   label=f'{threshold*100:.0f}%')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_data_projection(original_data: np.ndarray, 
                        projected_data: np.ndarray,
                        method_name: str = "PCA",
                        labels: Optional[np.ndarray] = None,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> None:
    """
    Plot data before and after PCA projection.
    
    Args:
        original_data: Original high-dimensional data
        projected_data: Data projected onto principal components
        method_name: Name of the method used
        labels: Optional labels for coloring points
        title: Plot title
        save_path: Path to save plot
    """
    if title is None:
        title = f"Data Projection using {method_name}"
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original data (first two dimensions)
    if labels is not None:
        scatter1 = axes[0].scatter(original_data[:, 0], original_data[:, 1], 
                                 c=labels, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter1, ax=axes[0])
    else:
        axes[0].scatter(original_data[:, 0], original_data[:, 1], alpha=0.7)
    
    axes[0].set_title('Original Data (First 2 Dimensions)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)
    
    # Projected data
    if projected_data.shape[1] >= 2:
        if labels is not None:
            scatter2 = axes[1].scatter(projected_data[:, 0], projected_data[:, 1], 
                                     c=labels, alpha=0.7, cmap='viridis')
            plt.colorbar(scatter2, ax=axes[1])
        else:
            axes[1].scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.7)
        
        axes[1].set_xlabel('PC 1')
        axes[1].set_ylabel('PC 2')
    else:
        # 1D projection
        if labels is not None:
            scatter2 = axes[1].scatter(projected_data[:, 0], np.zeros_like(projected_data[:, 0]), 
                                     c=labels, alpha=0.7, cmap='viridis')
            plt.colorbar(scatter2, ax=axes[1])
        else:
            axes[1].scatter(projected_data[:, 0], np.zeros_like(projected_data[:, 0]), alpha=0.7)
        
        axes[1].set_xlabel('PC 1')
        axes[1].set_ylabel('0')
    
    axes[1].set_title(f'Projected Data ({method_name})')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_method_comparison(results: Dict[str, Dict[str, Any]],
                          original_data: np.ndarray,
                          title: str = "PCA Methods Comparison",
                          save_path: Optional[str] = None) -> None:
    """
    Compare results from different PCA methods.
    
    Args:
        results: Dictionary with results from different methods
        original_data: Original data for reference
        title: Plot title
        save_path: Path to save plot
    """
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4*(n_methods+1), 8))
    
    if n_methods == 1:
        axes = axes.reshape(2, -1)
    
    # Plot original data
    axes[0, 0].scatter(original_data[:, 0], original_data[:, 1], alpha=0.7, c='gray')
    axes[0, 0].set_title('Original Data\n(First 2 Dimensions)')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot eigenvalues comparison
    method_names = list(results.keys())
    eigenvalues_data = [results[method]['eigenvalues'] for method in method_names]
    max_components = max(len(eigs) for eigs in eigenvalues_data)
    
    x_pos = np.arange(max_components)
    width = 0.8 / len(method_names)
    
    for i, (method, eigs) in enumerate(zip(method_names, eigenvalues_data)):
        offset = (i - len(method_names)/2 + 0.5) * width
        axes[1, 0].bar(x_pos[:len(eigs)] + offset, eigs, width, 
                      alpha=0.7, label=method.capitalize())
    
    axes[1, 0].set_title('Eigenvalues Comparison')
    axes[1, 0].set_xlabel('Principal Component')
    axes[1, 0].set_ylabel('Eigenvalue')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot each method's results
    colors = plt.cm.Set1(np.linspace(0, 1, n_methods))
    
    for i, (method, result) in enumerate(results.items()):
        col_idx = i + 1
        
        # Projected data
        projected_data = result['transformed_data']
        axes[0, col_idx].scatter(projected_data[:, 0], projected_data[:, 1], 
                               alpha=0.7, color=colors[i])
        axes[0, col_idx].set_title(f'{method.capitalize()} PCA\nProjected Data')
        axes[0, col_idx].set_xlabel('PC 1')
        axes[0, col_idx].set_ylabel('PC 2')
        axes[0, col_idx].grid(True, alpha=0.3)
        
        # Principal components
        components = result['components']
        n_features = components.shape[1]
        feature_names = [f'F{j+1}' for j in range(n_features)]
        
        # Plot first principal component
        axes[1, col_idx].bar(range(n_features), components[0], 
                           alpha=0.7, color=colors[i])
        axes[1, col_idx].set_title(f'{method.capitalize()} PCA\n1st Principal Component')
        axes[1, col_idx].set_xlabel('Features')
        axes[1, col_idx].set_ylabel('Loading')
        axes[1, col_idx].set_xticks(range(n_features))
        axes[1, col_idx].set_xticklabels(feature_names)
        axes[1, col_idx].grid(True, alpha=0.3)
        axes[1, col_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_reconstruction_error(original_data: np.ndarray,
                            reconstructed_data: np.ndarray,
                            method_name: str = "PCA",
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> None:
    """
    Plot reconstruction error analysis.
    
    Args:
        original_data: Original data
        reconstructed_data: Reconstructed data from PCA
        method_name: Name of method used
        title: Plot title
        save_path: Path to save plot
    """
    if title is None:
        title = f"Reconstruction Error Analysis - {method_name}"
    
    # Calculate errors
    error = original_data - reconstructed_data
    mse_per_sample = np.mean(error**2, axis=1)
    mse_per_feature = np.mean(error**2, axis=0)
    total_mse = np.mean(error**2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Error per sample
    axes[0, 0].plot(mse_per_sample, 'b-', alpha=0.7)
    axes[0, 0].set_title('MSE per Sample')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=total_mse, color='red', linestyle='--', 
                      label=f'Overall MSE: {total_mse:.4f}')
    axes[0, 0].legend()
    
    # Error per feature
    n_features = len(mse_per_feature)
    axes[0, 1].bar(range(n_features), mse_per_feature, alpha=0.7)
    axes[0, 1].set_title('MSE per Feature')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Mean Squared Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1, 0].hist(error.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Error Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Original vs Reconstructed (first feature)
    axes[1, 1].scatter(original_data[:, 0], reconstructed_data[:, 0], alpha=0.7)
    axes[1, 1].plot([original_data[:, 0].min(), original_data[:, 0].max()],
                   [original_data[:, 0].min(), original_data[:, 0].max()],
                   'r--', alpha=0.5, label='Perfect Reconstruction')
    axes[1, 1].set_title('Original vs Reconstructed (Feature 1)')
    axes[1, 1].set_xlabel('Original')
    axes[1, 1].set_ylabel('Reconstructed')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_qpca_dashboard(results: Dict[str, Dict[str, Any]],
                         original_data: np.ndarray,
                         save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive dashboard comparing QPCA methods.
    
    Args:
        results: Results from different PCA methods
        original_data: Original data
        save_path: Path to save dashboard
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots with custom layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Original data
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(original_data[:, 0], original_data[:, 1], alpha=0.7, c='gray')
    ax1.set_title('Original Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)
    
    # Method projections
    method_names = list(results.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(method_names)))
    
    for i, (method, result) in enumerate(results.items()):
        # Projected data
        ax = fig.add_subplot(gs[0, i+1])
        projected_data = result['transformed_data']
        ax.scatter(projected_data[:, 0], projected_data[:, 1], 
                  alpha=0.7, color=colors[i])
        ax.set_title(f'{method.capitalize()} PCA')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.grid(True, alpha=0.3)
    
    # Eigenvalues comparison
    ax_eig = fig.add_subplot(gs[1, :2])
    eigenvalues_data = [results[method]['eigenvalues'] for method in method_names]
    max_components = max(len(eigs) for eigs in eigenvalues_data)
    
    x_pos = np.arange(max_components)
    width = 0.8 / len(method_names)
    
    for i, (method, eigs) in enumerate(zip(method_names, eigenvalues_data)):
        offset = (i - len(method_names)/2 + 0.5) * width
        ax_eig.bar(x_pos[:len(eigs)] + offset, eigs, width, 
                  alpha=0.7, label=method.capitalize(), color=colors[i])
    
    ax_eig.set_title('Eigenvalues Comparison')
    ax_eig.set_xlabel('Principal Component')
    ax_eig.set_ylabel('Eigenvalue')
    ax_eig.legend()
    ax_eig.grid(True, alpha=0.3)
    
    # Explained variance ratios
    ax_var = fig.add_subplot(gs[1, 2:])
    for i, (method, result) in enumerate(results.items()):
        eigenvalues = result['eigenvalues']
        explained_var_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_var = np.cumsum(explained_var_ratio)
        ax_var.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                   'o-', alpha=0.7, label=method.capitalize(), color=colors[i])
    
    ax_var.set_title('Cumulative Explained Variance')
    ax_var.set_xlabel('Number of Components')
    ax_var.set_ylabel('Cumulative Variance Ratio')
    ax_var.legend()
    ax_var.grid(True, alpha=0.3)
    ax_var.set_ylim(0, 1)
    
    # Principal components comparison
    for i, (method, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[2, i])
        components = result['components']
        n_features = components.shape[1]
        
        # Plot first principal component
        ax.bar(range(n_features), components[0], alpha=0.7, color=colors[i])
        ax.set_title(f'{method.capitalize()}\n1st Component')
        ax.set_xlabel('Features')
        ax.set_ylabel('Loading')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.suptitle('QPCA Methods Dashboard', fontsize=18)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()