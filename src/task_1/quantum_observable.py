import numpy as np
from typing import List, Tuple

class QuantumObservables:
    def __init__(self):
        # Pauli matrices
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
    
    def get_observable(self, observable_type: str, target: int, n_qubits: int) -> np.ndarray:
        """Construct full observable matrix for n-qubit system"""
        # Get the base operator
        base_op = getattr(self, observable_type)
        
        # Build full operator using kronecker products
        op = np.array([[1]], dtype=complex)
        for i in range(n_qubits):
            op = np.kron(op, base_op if i == target else self.I)
        return op
    
    def get_product_observable(self, observables: List[Tuple[str, int]], n_qubits: int) -> np.ndarray:
        """Construct product of observables, e.g., Z₁⊗Z₂"""
        op = np.eye(2**n_qubits, dtype=complex)
        for obs_type, target in observables:
            op = op @ self.get_observable(obs_type, target, n_qubits)
        return op
