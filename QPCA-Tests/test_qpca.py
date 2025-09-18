"""
Unit tests for QPCA implementations.

These tests verify the correctness of classical and quantum PCA algorithms.
"""

import unittest
import numpy as np
import sys
import os

# Add the qpca module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qpca import QPCA, ClassicalPCA, compare_methods


class TestClassicalPCA(unittest.TestCase):
    """Test cases for Classical PCA implementation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create simple test data with known structure
        self.n_samples = 50
        self.n_features = 4
        
        # Generate data with clear principal components
        t = np.linspace(0, 2 * np.pi, self.n_samples)
        self.test_data = np.column_stack([
            np.cos(t),
            np.sin(t),
            0.5 * np.cos(t),
            0.1 * np.random.randn(self.n_samples)
        ])
    
    def test_pca_initialization(self):
        """Test PCA initialization."""
        pca = ClassicalPCA(n_components=2)
        self.assertEqual(pca.n_components, 2)
        self.assertIsNone(pca.components_)
        self.assertIsNone(pca.eigenvalues_)
    
    def test_pca_fit(self):
        """Test PCA fitting."""
        pca = ClassicalPCA(n_components=2)
        pca.fit(self.test_data)
        
        # Check that components and eigenvalues are computed
        self.assertIsNotNone(pca.components_)
        self.assertIsNotNone(pca.eigenvalues_)
        
        # Check shapes
        self.assertEqual(pca.components_.shape, (2, self.n_features))
        self.assertEqual(pca.eigenvalues_.shape, (2,))
        
        # Check that eigenvalues are sorted in descending order
        self.assertTrue(np.all(pca.eigenvalues_[:-1] >= pca.eigenvalues_[1:]))
    
    def test_pca_transform(self):
        """Test PCA transformation."""
        pca = ClassicalPCA(n_components=2)
        pca.fit(self.test_data)
        
        transformed = pca.transform(self.test_data)
        
        # Check shape
        self.assertEqual(transformed.shape, (self.n_samples, 2))
        
        # Check that transformation preserves variance correctly
        # (The first PC should capture most variance)
        var_pc1 = np.var(transformed[:, 0])
        var_pc2 = np.var(transformed[:, 1])
        self.assertGreater(var_pc1, var_pc2)
    
    def test_pca_fit_transform(self):
        """Test combined fit and transform."""
        pca = ClassicalPCA(n_components=2)
        
        transformed1 = pca.fit_transform(self.test_data)
        transformed2 = pca.transform(self.test_data)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(transformed1, transformed2)
    
    def test_pca_reconstruction(self):
        """Test data reconstruction accuracy."""
        pca = ClassicalPCA(n_components=3)  # Use more components for better reconstruction
        transformed = pca.fit_transform(self.test_data)
        
        # Reconstruct data
        reconstructed = transformed @ pca.components_ + pca.mean_
        
        # Reconstruction error should be small for high-variance data
        mse = np.mean((self.test_data - reconstructed) ** 2)
        self.assertLess(mse, 0.1)  # Should be reasonably accurate


class TestQPCAInterface(unittest.TestCase):
    """Test cases for QPCA interface."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_data = np.random.randn(30, 3)
    
    def test_classical_method(self):
        """Test QPCA with classical method."""
        qpca = QPCA(method='classical', n_components=2)
        transformed = qpca.fit_transform(self.test_data)
        
        self.assertEqual(transformed.shape, (30, 2))
        self.assertEqual(qpca.components_.shape, (2, 3))
        self.assertEqual(qpca.eigenvalues_.shape, (2,))
    
    def test_invalid_method(self):
        """Test QPCA with invalid method."""
        with self.assertRaises(ValueError):
            QPCA(method='invalid_method')
    
    def test_variational_method_without_qubits(self):
        """Test variational method requires n_qubits."""
        with self.assertRaises(ValueError):
            QPCA(method='variational', n_components=2)


class TestComparisonFunction(unittest.TestCase):
    """Test cases for method comparison function."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_data = np.random.randn(25, 3)
    
    def test_compare_methods(self):
        """Test compare_methods function."""
        results = compare_methods(self.test_data, n_components=2)
        
        # Should always have classical results
        self.assertIn('classical', results)
        
        classical_result = results['classical']
        self.assertIn('transformed_data', classical_result)
        self.assertIn('components', classical_result)
        self.assertIn('eigenvalues', classical_result)
        
        # Check shapes
        self.assertEqual(classical_result['transformed_data'].shape, (25, 2))
        self.assertEqual(classical_result['components'].shape, (2, 3))
        self.assertEqual(classical_result['eigenvalues'].shape, (2,))


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation and edge cases."""
    
    def test_empty_data(self):
        """Test behavior with empty data."""
        pca = ClassicalPCA(n_components=1)
        empty_data = np.array([]).reshape(0, 3)
        
        with self.assertRaises((ValueError, np.linalg.LinAlgError)):
            pca.fit(empty_data)
    
    def test_single_sample(self):
        """Test behavior with single sample."""
        pca = ClassicalPCA(n_components=1)
        single_sample = np.array([[1, 2, 3]])
        
        # Should handle gracefully (though not very meaningful)
        try:
            pca.fit(single_sample)
            # If it succeeds, check basic properties
            self.assertIsNotNone(pca.components_)
        except (ValueError, np.linalg.LinAlgError):
            # Acceptable to fail with single sample
            pass
    
    def test_more_components_than_features(self):
        """Test requesting more components than features."""
        pca = ClassicalPCA(n_components=5)
        data = np.random.randn(10, 3)  # Only 3 features
        
        pca.fit(data)
        # Should automatically limit to number of features
        self.assertLessEqual(pca.components_.shape[0], 3)
    
    def test_transform_before_fit(self):
        """Test transform before fitting."""
        pca = ClassicalPCA(n_components=2)
        data = np.random.randn(10, 3)
        
        with self.assertRaises(ValueError):
            pca.transform(data)


class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of PCA."""
    
    def setUp(self):
        """Set up test data with known properties."""
        np.random.seed(42)
        
        # Create data with known covariance structure
        n_samples = 100
        # Generate uncorrelated data in transformed space
        z = np.random.randn(n_samples, 3)
        
        # Apply transformation to create correlations
        transform_matrix = np.array([
            [1, 0.5, 0.2],
            [0, 1, 0.3],
            [0, 0, 1]
        ])
        
        self.test_data = z @ transform_matrix.T
    
    def test_orthogonal_components(self):
        """Test that principal components are orthogonal."""
        pca = ClassicalPCA(n_components=3)
        pca.fit(self.test_data)
        
        # Components should be orthonormal
        dot_product = pca.components_ @ pca.components_.T
        
        # Should be approximately identity matrix
        identity = np.eye(3)
        np.testing.assert_array_almost_equal(dot_product, identity, decimal=10)
    
    def test_variance_explained(self):
        """Test that eigenvalues represent variance explained."""
        pca = ClassicalPCA(n_components=3)
        transformed = pca.fit_transform(self.test_data)
        
        # Variance of each principal component should match eigenvalues
        pc_variances = np.var(transformed, axis=0, ddof=1)
        
        # Should be approximately equal to eigenvalues
        np.testing.assert_array_almost_equal(pc_variances, pca.eigenvalues_, decimal=5)
    
    def test_total_variance_preservation(self):
        """Test that total variance is preserved."""
        # Center the data
        centered_data = self.test_data - np.mean(self.test_data, axis=0)
        original_total_var = np.sum(np.var(centered_data, axis=0, ddof=1))
        
        pca = ClassicalPCA()  # All components
        pca.fit(self.test_data)
        
        total_explained_var = np.sum(pca.eigenvalues_)
        
        # Total variance should be preserved
        self.assertAlmostEqual(original_total_var, total_explained_var, places=5)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestClassicalPCA,
        TestQPCAInterface,
        TestComparisonFunction,
        TestDataValidation,
        TestMathematicalProperties
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running QPCA Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    print("=" * 50)
    if success:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        exit(1)