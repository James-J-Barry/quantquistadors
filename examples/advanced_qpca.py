#!/usr/bin/env python3
"""
Advanced QPCA Example with Real Data

This example demonstrates QPCA on real datasets and compares
different approaches for dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import load_digits, load_breast_cancer, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Add the qpca module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qpca import QPCA, compare_methods
from visualizations import plot_method_comparison, plot_eigenvalue_spectrum, create_qpca_dashboard


def analyze_digits_dataset():
    """Analyze the digits dataset with QPCA."""
    print("\n" + "="*60)
    print("DIGITS DATASET ANALYSIS")
    print("="*60)
    
    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Features: {X.shape[1]} (8x8 pixel images)")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply QPCA with different numbers of components
    n_components_list = [2, 10, 20, 30]
    
    print(f"\nApplying PCA with different numbers of components...")
    
    results = {}
    for n_comp in n_components_list:
        qpca = QPCA(method='classical', n_components=n_comp)
        X_transformed = qpca.fit_transform(X_scaled)
        
        # Calculate explained variance ratio
        explained_var_ratio = qpca.eigenvalues_ / np.sum(qpca.eigenvalues_)
        cumulative_var = np.sum(explained_var_ratio)
        
        results[n_comp] = {
            'transformed_data': X_transformed,
            'explained_variance_ratio': explained_var_ratio,
            'cumulative_variance': cumulative_var,
            'components': qpca.components_
        }
        
        print(f"  {n_comp} components: {cumulative_var:.1%} variance explained")
    
    # Visualize results for 2D case
    X_2d = results[2]['transformed_data']
    
    plt.figure(figsize=(12, 4))
    
    # Original images (show first few digits)
    plt.subplot(1, 3, 1)
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(digits.images[i], cmap='gray')
        plt.title(f'Digit: {y[i]}')
        plt.axis('off')
    
    plt.figure(figsize=(12, 4))
    
    # 2D PCA projection
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Digits in 2D PCA Space')
    
    # Explained variance plot
    plt.subplot(1, 3, 2)
    full_pca = QPCA(method='classical')
    full_pca.fit(X_scaled)
    explained_var = full_pca.eigenvalues_ / np.sum(full_pca.eigenvalues_)
    cumulative_var = np.cumsum(explained_var)
    
    plt.plot(range(1, min(21, len(cumulative_var) + 1)), 
             cumulative_var[:20], 'bo-', alpha=0.7)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Classification performance vs components
    plt.subplot(1, 3, 3)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    accuracies = []
    for n_comp in [1, 2, 5, 10, 15, 20, 30, 40, 50]:
        if n_comp <= X.shape[1]:
            # Apply PCA
            qpca = QPCA(method='classical', n_components=n_comp)
            X_train_pca = qpca.fit_transform(X_train)
            X_test_pca = qpca.transform(X_test)
            
            # Train classifier
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(X_train_pca, y_train)
            
            # Evaluate
            y_pred = classifier.predict(X_test_pca)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
    
    plt.plot(range(1, len(accuracies) + 1), accuracies, 'go-', alpha=0.7)
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Classification Accuracy')
    plt.title('Classification Performance vs Components')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('digits_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nDigits analysis plot saved to 'digits_analysis.png'")
    
    return results


def analyze_breast_cancer_dataset():
    """Analyze the breast cancer dataset with QPCA."""
    print("\n" + "="*60)
    print("BREAST CANCER DATASET ANALYSIS")
    print("="*60)
    
    # Load data
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Features: {cancer.feature_names[:5]}... (showing first 5)")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compare methods
    print(f"\nComparing PCA methods...")
    results = compare_methods(X_scaled, n_components=2)
    
    # Analyze feature importance in principal components
    classical_result = results['classical']
    components = classical_result['components']
    
    print(f"\nTop features in each principal component:")
    for i, component in enumerate(components):
        # Get indices of top features (by absolute value)
        top_indices = np.argsort(np.abs(component))[-5:][::-1]
        print(f"\nPC{i+1} (explains {classical_result['eigenvalues'][i]:.2f} variance):")
        for idx in top_indices:
            print(f"  {cancer.feature_names[idx]}: {component[idx]:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # 2D projection colored by class
    plt.subplot(1, 3, 1)
    transformed_data = classical_result['transformed_data']
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=colors, alpha=0.7)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Breast Cancer Data in PCA Space')
    
    # Add legend
    plt.scatter([], [], c='red', alpha=0.7, label='Malignant')
    plt.scatter([], [], c='blue', alpha=0.7, label='Benign')
    plt.legend()
    
    # Feature loadings for PC1
    plt.subplot(1, 3, 2)
    feature_loadings = components[0]
    sorted_indices = np.argsort(np.abs(feature_loadings))[-10:]
    
    plt.barh(range(len(sorted_indices)), feature_loadings[sorted_indices])
    plt.yticks(range(len(sorted_indices)), 
               [cancer.feature_names[i] for i in sorted_indices])
    plt.xlabel('Loading Value')
    plt.title('Top 10 Features in PC1')
    plt.grid(True, alpha=0.3)
    
    # Eigenvalue spectrum
    plt.subplot(1, 3, 3)
    plot_eigenvalue_spectrum(classical_result['eigenvalues'], 
                           method_name="Breast Cancer", title=None)
    
    plt.tight_layout()
    plt.savefig('breast_cancer_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nBreast cancer analysis plot saved to 'breast_cancer_analysis.png'")
    
    return results


def analyze_swiss_roll_dataset():
    """Analyze the Swiss roll dataset to demonstrate PCA limitations."""
    print("\n" + "="*60)
    print("SWISS ROLL DATASET ANALYSIS (PCA LIMITATIONS)")
    print("="*60)
    
    # Generate Swiss roll data
    n_samples = 1500
    X_3d, color = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    
    print(f"Dataset shape: {X_3d.shape}")
    print(f"This demonstrates when PCA might not be optimal")
    print(f"(Swiss roll is a nonlinear manifold)")
    
    # Apply PCA
    qpca = QPCA(method='classical', n_components=2)
    X_pca = qpca.fit_transform(X_3d)
    
    print(f"\nExplained variance ratio: {qpca.eigenvalues_ / np.sum(qpca.eigenvalues_)}")
    
    # Visualize
    fig = plt.figure(figsize=(15, 5))
    
    # Original 3D data
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=color, cmap=plt.cm.Spectral)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('X3')
    ax1.set_title('Original Swiss Roll (3D)')
    
    # PCA projection (loses nonlinear structure)
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    ax2.set_title('PCA Projection (2D)')
    
    # Show principal components in 3D space
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=color, cmap=plt.cm.Spectral, alpha=0.3)
    
    # Plot principal component directions
    mean_point = np.mean(X_3d, axis=0)
    pc1_direction = qpca.components_[0] * 3 * np.sqrt(qpca.eigenvalues_[0])
    pc2_direction = qpca.components_[1] * 3 * np.sqrt(qpca.eigenvalues_[1])
    
    ax3.plot([mean_point[0] - pc1_direction[0], mean_point[0] + pc1_direction[0]],
             [mean_point[1] - pc1_direction[1], mean_point[1] + pc1_direction[1]],
             [mean_point[2] - pc1_direction[2], mean_point[2] + pc1_direction[2]], 
             'r-', linewidth=3, label='PC1')
    
    ax3.plot([mean_point[0] - pc2_direction[0], mean_point[0] + pc2_direction[0]],
             [mean_point[1] - pc2_direction[1], mean_point[1] + pc2_direction[1]],
             [mean_point[2] - pc2_direction[2], mean_point[2] + pc2_direction[2]], 
             'g-', linewidth=3, label='PC2')
    
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.set_zlabel('X3')
    ax3.set_title('Principal Components in 3D')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('swiss_roll_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSwiss roll analysis plot saved to 'swiss_roll_analysis.png'")
    
    print(f"\nNote: PCA preserves linear relationships but may miss")
    print(f"nonlinear manifold structure. For this data, techniques like")
    print(f"t-SNE, UMAP, or kernel PCA might be more appropriate.")


def performance_comparison():
    """Compare computational performance of different methods."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    import time
    
    # Test different data sizes
    sizes = [100, 500, 1000, 2000]
    n_features = 50
    n_components = 10
    
    print(f"Testing classical PCA performance...")
    print(f"Features: {n_features}, Components: {n_components}")
    print(f"{'Samples':<10} {'Time (s)':<10} {'Memory (MB)':<12}")
    print("-" * 35)
    
    for n_samples in sizes:
        # Generate random data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        # Measure time
        start_time = time.time()
        qpca = QPCA(method='classical', n_components=n_components)
        X_transformed = qpca.fit_transform(X)
        end_time = time.time()
        
        # Estimate memory usage (rough approximation)
        memory_mb = (X.nbytes + X_transformed.nbytes + qpca.components_.nbytes) / 1024 / 1024
        
        print(f"{n_samples:<10} {end_time - start_time:<10.4f} {memory_mb:<12.2f}")


def main():
    """Main function to run all analyses."""
    print("ADVANCED QPCA ANALYSIS")
    print("=" * 60)
    print("This example demonstrates QPCA on real datasets")
    print("and explores various aspects of dimensionality reduction.")
    
    try:
        # Run different analyses
        digits_results = analyze_digits_dataset()
        cancer_results = analyze_breast_cancer_dataset()
        analyze_swiss_roll_dataset()
        performance_comparison()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- digits_analysis.png: Handwritten digits analysis")
        print("- breast_cancer_analysis.png: Cancer dataset analysis")
        print("- swiss_roll_analysis.png: Nonlinear manifold example")
        print("\nKey insights:")
        print("1. PCA effectively reduces dimensionality while preserving variance")
        print("2. Feature interpretability is preserved in principal components")
        print("3. PCA works best for linear relationships in data")
        print("4. Number of components should balance variance explained vs complexity")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("This might be due to missing dependencies (sklearn, matplotlib)")
        print("Install with: pip install scikit-learn matplotlib")


if __name__ == "__main__":
    main()