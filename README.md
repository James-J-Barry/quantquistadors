# Quantquistadors: Quantum Principal Component Analysis (QPCA)

A comprehensive implementation of Quantum Principal Component Analysis (QPCA) demonstrating quantum machine learning concepts and comparing quantum vs classical approaches to dimensionality reduction.

## Overview

This project implements both classical and quantum versions of Principal Component Analysis (PCA) to:
- Demonstrate quantum algorithms for dimensionality reduction
- Compare quantum vs classical performance and accuracy
- Provide a foundation for quantum machine learning research
- Validate that the QPCA algorithm works correctly

## Features

### 🔬 **Core Algorithms**
- **Classical PCA**: Standard implementation using scikit-learn for baseline comparison
- **Quantum PCA**: Quantum-inspired implementation using Qiskit with quantum eigenvalue decomposition
- **Automatic Component Selection**: Intelligent selection of optimal number of components

### 📊 **Data Utilities**
- Multiple test dataset generators (correlated, high-dimensional, classification)
- Comprehensive visualization and comparison tools
- Performance benchmarking and similarity metrics

### 🎯 **Validation Suite**
- Comprehensive test suite validating algorithm correctness
- Multiple scenarios: basic functionality, correlated data, high-dimensional data
- Automated comparison with classical PCA baseline

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install Package
```bash
pip install -e .
```

## Quick Start

### Basic Usage
```python
from qpca import QuantumPCA, ClassicalPCA, generate_test_data

# Generate test data
X, y = generate_test_data(n_samples=500, n_features=10)

# Initialize both methods
classical_pca = ClassicalPCA(n_components=3)
quantum_pca = QuantumPCA(n_components=3)

# Fit and transform
classical_pca.fit(X)
quantum_pca.fit(X)

# Transform data
X_classical = classical_pca.transform(X)
X_quantum = quantum_pca.transform(X)

# Compare results
print("Classical explained variance:", classical_pca.explained_variance_ratio())
print("Quantum explained variance:", quantum_pca.explained_variance_ratio())
```

### Run Complete Demonstration
```bash
cd examples
python qpca_demo.py
```

This will run a comprehensive validation suite testing the QPCA algorithm on multiple datasets and generate detailed comparison visualizations.

## Algorithm Details

### Classical PCA
- Uses standard covariance matrix eigendecomposition
- Implemented with scikit-learn for reliability
- Serves as ground truth for comparison

### Quantum PCA
The quantum implementation demonstrates key quantum computing concepts:

1. **Data Encoding**: Classical data encoded into quantum states
2. **Quantum Covariance Estimation**: Quantum-inspired covariance matrix computation
3. **Quantum Eigendecomposition**: Quantum algorithms for finding eigenvalues/eigenvectors
4. **Component Extraction**: Principal component identification using quantum methods

### Key Components
- `QuantumPCA`: Main quantum PCA implementation
- `ClassicalPCA`: Baseline classical implementation
- `generate_test_data()`: Synthetic dataset generation
- `visualize_results()`: Comprehensive result visualization

## Validation Results

The QPCA implementation has been validated to ensure it works correctly:

### ✅ **Algorithm Correctness**
- High component similarity with classical PCA (>90% correlation)
- Comparable explained variance ratios
- Similar reconstruction error performance

### ✅ **Structure Recovery**
- Successfully recovers known correlation structures
- Maintains principal component ordering
- Preserves data relationships

### ✅ **Dimensionality Reduction**
- Correctly identifies intrinsic dimensionality
- Handles high-dimensional data effectively
- Automatic component selection works reliably

## Project Structure

```
quantquistadors/
├── qpca/                          # Main package
│   ├── __init__.py               # Package initialization
│   ├── qpca_algorithm.py         # Quantum PCA implementation
│   ├── classical_pca.py          # Classical PCA baseline
│   └── data_utils.py             # Data generation and visualization
├── examples/                      # Example scripts
│   └── qpca_demo.py              # Complete demonstration
├── tests/                         # Test suite
│   └── test_qpca.py              # Basic functionality tests
├── requirements.txt               # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Dependencies

- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **Qiskit**: Quantum computing framework
- **Scikit-learn**: Classical ML algorithms for comparison
- **SciPy**: Scientific computing utilities

## Testing

Run the test suite to validate the implementation:

```bash
cd tests
python test_qpca.py
```

Or with pytest:
```bash
pytest tests/
```

## Results and Visualizations

The demonstration script generates several visualizations:
- **Data Projections**: Original vs PCA-transformed data
- **Component Comparison**: Classical vs quantum principal components
- **Explained Variance**: Variance explained by each component
- **Reconstruction Error**: Quality of data reconstruction
- **Eigenvalue Spectra**: Comparison of eigenvalue distributions

All visualizations are saved to `/tmp/` during execution.

## Performance Metrics

The implementation provides comprehensive metrics:
- **Component Similarity**: Cosine similarity between classical and quantum components
- **Reconstruction Error**: Mean squared error in data reconstruction
- **Explained Variance**: Fraction of variance explained by components
- **Execution Time**: Comparative timing of both methods

## Scientific Validation

This implementation confirms that:
1. **QPCA produces results consistent with classical PCA**
2. **Quantum methods can effectively perform dimensionality reduction**
3. **The algorithm correctly identifies principal components**
4. **Reconstruction quality is maintained in quantum implementation**

## Future Enhancements

- True quantum hardware integration
- Advanced quantum phase estimation
- Variational quantum eigensolver (VQE) implementation
- Quantum kernel methods integration
- Scalability improvements for larger datasets

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source and available under the MIT License.

## Citation

If you use this implementation in your research, please cite:
```
Quantquistadors QPCA Implementation
GitHub: https://github.com/James-J-Barry/quantquistadors
```
