import numpy as np
from typing import List, Tuple
from src.task_1.quantum_simulator import QuantumSimulator

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

def demonstrate_expectations():
    # Initialize our simulators
    simulator = QuantumSimulator()
    observables = QuantumObservables()
    
    # Example 1: Single qubit in superposition
    print("\nExample 1: Single qubit in superposition (H|0⟩)")
    n_qubits = 1
    circuit = [('H', 0)]
    state = simulator.simulate_circuit_matrix(n_qubits, circuit)
    
    # Measure different observables
    for obs_type in ['X', 'Y', 'Z']:
        observable = observables.get_observable(obs_type, 0, n_qubits)
        expect_val = simulator.expectation_value(state, observable)
        print(f"⟨{obs_type}⟩ = {expect_val.real:.3f}")
    
    # Example 2: Bell state
    print("\nExample 2: Bell state (CNOT(H⊗I)|00⟩)")
    n_qubits = 2
    circuit = [('H', 0), ('CNOT', 0, 1)]
    state = simulator.simulate_circuit_matrix(n_qubits, circuit)
    
    # Measure correlations
    zz_obs = observables.get_product_observable([('Z', 0), ('Z', 1)], n_qubits)
    xx_obs = observables.get_product_observable([('X', 0), ('X', 1)], n_qubits)
    yy_obs = observables.get_product_observable([('Y', 0), ('Y', 1)], n_qubits)
    
    print(f"⟨Z₁Z₂⟩ = {simulator.expectation_value(state, zz_obs).real:.3f}")
    print(f"⟨X₁X₂⟩ = {simulator.expectation_value(state, xx_obs).real:.3f}")
    print(f"⟨Y₁Y₂⟩ = {simulator.expectation_value(state, yy_obs).real:.3f}")
    
    # Example 3: GHZ state
    print("\nExample 3: GHZ state")
    n_qubits = 3
    circuit = [
        ('H', 0),
        ('CNOT', 0, 1),
        ('CNOT', 1, 2)
    ]
    state = simulator.simulate_circuit_matrix(n_qubits, circuit)
    
    # Measure various correlations
    zzz_obs = observables.get_product_observable([('Z', 0), ('Z', 1), ('Z', 2)], n_qubits)
    xxx_obs = observables.get_product_observable([('X', 0), ('X', 1), ('X', 2)], n_qubits)
    
    print(f"⟨Z₁Z₂Z₃⟩ = {simulator.expectation_value(state, zzz_obs).real:.3f}")
    print(f"⟨X₁X₂X₃⟩ = {simulator.expectation_value(state, xxx_obs).real:.3f}")
    
    # Individual Z measurements should be 0 for GHZ state
    for i in range(n_qubits):
        z_obs = observables.get_observable('Z', i, n_qubits)
        expect_val = simulator.expectation_value(state, z_obs)
        print(f"⟨Z_{i+1}⟩ = {expect_val.real:.3f}")