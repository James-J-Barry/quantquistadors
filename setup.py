from setuptools import setup, find_packages

setup(
    name="quantquistadors",
    version="0.1.0",
    description="Quantum Principal Component Analysis implementation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "qiskit>=0.45.0",
        "qiskit-aer>=0.12.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "jupyter>=1.0.0",
    ],
)