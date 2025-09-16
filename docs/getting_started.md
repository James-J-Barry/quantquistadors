# Getting Started with QPCA

This guide will help you get started with the Quantum Principal Component Analysis (QPCA) implementation in this repository.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/James-J-Barry/quantquistadors.git
cd quantquistadors
```

2. Install basic dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For quantum implementations, install Qiskit:
```bash
pip install qiskit qiskit-aer
```

## Quick Start

### Basic Classical PCA

```python
import numpy as np
from qpca import QPCA

# Generate sample data
data = np.random.randn(100, 5)

# Create and fit PCA
pca = QPCA(method='classical', n_components=2)
transformed_data = pca.fit_transform(data)

print(f"Original shape: {data.shape}")
print(f"Transformed shape: {transformed_data.shape}")
print(f"Explained variance: {pca.eigenvalues_}")
```

### Comparing Methods

```python
from qpca import compare_methods

# Compare different PCA implementations
results = compare_methods(data, n_components=2)

for method, result in results.items():
    print(f"{method.upper()}:")
    print(f"  Eigenvalues: {result['eigenvalues']}")
    print(f"  Explained variance ratio: {result['eigenvalues'] / sum(result['eigenvalues'])}")
```

### Variational QPCA (with Qiskit)

```python
# Only available if Qiskit is installed
qpca_quantum = QPCA(method='variational', n_components=2, n_qubits=3)
quantum_result = qpca_quantum.fit_transform(data)
```

## Examples

### Example 1: Basic Usage
See `examples/basic_qpca.py` for a complete example that:
- Generates synthetic data with known structure
- Applies different PCA methods
- Visualizes the results
- Compares classical and quantum approaches

### Example 2: Real Data Analysis
```python
from sklearn.datasets import load_iris
from qpca import QPCA
import matplotlib.pyplot as plt

# Load real dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply QPCA
qpca = QPCA(method='classical', n_components=2)
X_transformed = qpca.fit_transform(X)

# Visualize results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Data (First 2 Features)')

plt.subplot(1, 2, 2)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Transformed Data')

plt.tight_layout()
plt.show()

print(f"Explained variance ratio: {qpca.eigenvalues_ / sum(qpca.eigenvalues_)}")
```

## Understanding the Results

### Principal Components
The principal components are the directions of maximum variance in your data:
```python
print("Principal Components:")
for i, component in enumerate(qpca.components_):
    print(f"PC{i+1}: {component}")
```

### Eigenvalues
Eigenvalues represent the amount of variance explained by each component:
```python
total_variance = sum(qpca.eigenvalues_)
for i, eigenvalue in enumerate(qpca.eigenvalues_):
    variance_ratio = eigenvalue / total_variance
    print(f"PC{i+1} explains {variance_ratio:.1%} of variance")
```

### Choosing Number of Components
```python
import matplotlib.pyplot as plt

# Fit PCA with all components
pca_full = QPCA(method='classical')
pca_full.fit(data)

# Plot explained variance
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca_full.eigenvalues_) + 1), pca_full.eigenvalues_)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues')

plt.subplot(1, 2, 2)
cumulative_var = np.cumsum(pca_full.eigenvalues_) / sum(pca_full.eigenvalues_)
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()

plt.tight_layout()
plt.show()
```

## Visualization

The repository includes comprehensive visualization tools:

```python
from visualizations import plot_method_comparison, plot_eigenvalue_spectrum

# Compare different methods visually
results = compare_methods(data, n_components=2)
plot_method_comparison(results, data)

# Analyze eigenvalue spectrum
plot_eigenvalue_spectrum(qpca.eigenvalues_, method_name="Classical PCA")
```

## Best Practices

### Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler

# Standardize features (recommended for PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA to scaled data
qpca = QPCA(method='classical', n_components=2)
result = qpca.fit_transform(data_scaled)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Use PCA for dimensionality reduction in ML pipeline
pca = QPCA(method='classical', n_components=10)
X_reduced = pca.fit_transform(X)

# Evaluate on downstream task
classifier = LogisticRegression()
scores = cross_val_score(classifier, X_reduced, y, cv=5)
print(f"Classification accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Common Issues and Solutions

### Issue: "Qiskit not available"
**Solution**: Install Qiskit for quantum implementations:
```bash
pip install qiskit qiskit-aer
```

### Issue: Poor reconstruction quality
**Solution**: 
- Increase number of components
- Check if data is properly centered/scaled
- Verify data doesn't have too much noise

### Issue: Slow performance
**Solutions**:
- Use fewer components for initial exploration
- Consider data sampling for very large datasets
- Use classical methods for quick prototyping

## Next Steps

1. **Explore Advanced Features**: Look into `docs/qpca_theory.md` for mathematical details
2. **Custom Implementations**: Extend the codebase for specific quantum hardware
3. **Real Applications**: Apply QPCA to your own datasets
4. **Contribute**: Help improve the implementations and documentation

## Support

For questions or issues:
1. Check the documentation in `docs/`
2. Review the examples in `examples/`
3. Run the tests in `tests/` to verify your installation
4. Open an issue on GitHub for bugs or feature requests