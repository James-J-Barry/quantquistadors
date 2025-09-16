"""
Classical PCA implementation for baseline comparison with QPCA
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class ClassicalPCA:
    """Classical Principal Component Analysis implementation"""
    
    def __init__(self, n_components=None):
        """
        Initialize Classical PCA
        
        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, X):
        """
        Fit the PCA model to data
        
        Args:
            X: Input data matrix (n_samples, n_features)
        """
        # Standardize the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.pca.fit(X_scaled)
        self.fitted = True
        
    def transform(self, X):
        """
        Transform data to principal component space
        
        Args:
            X: Input data matrix
            
        Returns:
            Transformed data in PC space
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before transform")
            
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
        
    def fit_transform(self, X):
        """
        Fit model and transform data in one step
        
        Args:
            X: Input data matrix
            
        Returns:
            Transformed data in PC space
        """
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X_transformed):
        """
        Transform data back from PC space to original space
        
        Args:
            X_transformed: Data in PC space
            
        Returns:
            Data transformed back to original space
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before inverse transform")
            
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        return self.scaler.inverse_transform(X_reconstructed)
        
    def explained_variance_ratio(self):
        """Get the explained variance ratio for each component"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.pca.explained_variance_ratio_
        
    def get_components(self):
        """Get the principal components (eigenvectors)"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.pca.components_
        
    def reconstruction_error(self, X):
        """
        Calculate reconstruction error
        
        Args:
            X: Original data
            
        Returns:
            Mean squared reconstruction error
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed) ** 2)