"""
Quantum Principal Component Analysis (QPCA) Package

This package implements quantum algorithms for principal component analysis,
providing both classical and quantum implementations for comparison.
"""

__version__ = "0.1.0"

from .qpca_algorithm import QuantumPCA
from .classical_pca import ClassicalPCA
from .data_utils import generate_test_data, visualize_results

__all__ = [
    "QuantumPCA",
    "ClassicalPCA", 
    "generate_test_data",
    "visualize_results"
]