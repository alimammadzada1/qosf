import numpy as np
from typing import List, Tuple

class QuantumSimulator:
    def __init__(self):
        # Define basic single-qubit gates
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        # Define CNOT gate
        self.CNOT = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=complex)

    def initialize_state(self, n_qubits: int) -> np.ndarray:
        """Initialize state |0...0> with n qubits"""
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0
        return state

    def initialize_tensor_state(self, n_qubits: int) -> np.ndarray:
        """Initialize state |0...0> as tensor"""
        state = np.zeros([2] * n_qubits, dtype=complex)
        state[(0,) * n_qubits] = 1.0
        return state

    def apply_single_qubit_gate_matrix(self, state: np.ndarray, gate: np.ndarray, 
                                     target: int, n_qubits: int) -> np.ndarray:
        """Apply single qubit gate using matrix multiplication"""
        # Construct the full operator using kronecker products
        op = np.array([[1]], dtype=complex)
        for i in range(n_qubits):
            op = np.kron(op, gate if i == target else self.I)
        return op @ state

    def apply_single_qubit_gate_tensor(self, state: np.ndarray, gate: np.ndarray, 
                                     target: int) -> np.ndarray:
        """Apply single qubit gate using tensor multiplication"""
        n_qubits = len(state.shape)
        
        # Reshape gate into rank-4 tensor
        gate_tensor = gate.reshape(2, 1, 2, 1)
        
        # Contract gate with state
        state_reshape = np.moveaxis(state, target, 0)
        state_reshape = state_reshape.reshape(2, -1)
        result = gate @ state_reshape
        result = result.reshape([2] + [2] * (n_qubits - 1))
        
        return np.moveaxis(result, 0, target)

    def apply_cnot_matrix(self, state: np.ndarray, control: int, target: int, 
                         n_qubits: int) -> np.ndarray:
        """Apply CNOT gate using matrix multiplication"""
        # Initialize the operator
        dim = 2**n_qubits
        op = np.eye(dim, dtype=complex)
        
        # Apply CNOT operation
        for i in range(dim):
            # Convert to binary and check control bit
            bin_str = format(i, f'0{n_qubits}b')
            if bin_str[control] == '1':
                # Flip target bit
                new_bin = list(bin_str)
                new_bin[target] = '1' if bin_str[target] == '0' else '0'
                j = int(''.join(new_bin), 2)
                
                # Swap elements in operator
                op[i, i] = 0
                op[j, j] = 0
                op[i, j] = 1
                op[j, i] = 1
        
        return op @ state

    def apply_cnot_tensor(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate using tensor multiplication"""
        n_qubits = len(state.shape)
        
        # Move control and target qubits to first two positions
        axes = list(range(n_qubits))
        axes.remove(control)
        axes.remove(target)
        axes = [control, target] + axes
        
        # Reshape state
        state_temp = np.transpose(state, axes)
        shape_before = state_temp.shape
        state_temp = state_temp.reshape(4, -1)
        
        # Apply CNOT
        state_temp = self.CNOT @ state_temp
        state_temp = state_temp.reshape(shape_before)
        
        # Restore original qubit order
        inv_axes = [axes.index(i) for i in range(n_qubits)]
        return np.transpose(state_temp, inv_axes)

    def simulate_circuit_matrix(self, n_qubits: int, gates: List[Tuple]) -> np.ndarray:
        """Simulate quantum circuit using matrix multiplication"""
        state = self.initialize_state(n_qubits)
        
        for gate, *qubits in gates:
            if gate in ['X', 'H']:
                gate_matrix = getattr(self, gate)
                state = self.apply_single_qubit_gate_matrix(
                    state, gate_matrix, qubits[0], n_qubits)
            elif gate == 'CNOT':
                state = self.apply_cnot_matrix(
                    state, qubits[0], qubits[1], n_qubits)
        
        return state

    def simulate_circuit_tensor(self, n_qubits: int, gates: List[Tuple]) -> np.ndarray:
        """Simulate quantum circuit using tensor multiplication"""
        state = self.initialize_tensor_state(n_qubits)
        
        for gate, *qubits in gates:
            if gate in ['X', 'H']:
                gate_matrix = getattr(self, gate)
                state = self.apply_single_qubit_gate_tensor(
                    state, gate_matrix, qubits[0])
            elif gate == 'CNOT':
                state = self.apply_cnot_tensor(state, qubits[0], qubits[1])
        
        return state

    def measure_state(self, state: np.ndarray, shots: int = 1000) -> dict:
        """Sample from the final quantum state"""
        # Ensure state is 1D
        if len(state.shape) > 1:
            state = state.flatten()
        
        # Calculate probabilities
        probs = np.abs(state) ** 2
        probs /= np.sum(probs)  # Normalize
        
        # Sample from distribution
        n_qubits = int(np.log2(len(state)))
        outcomes = np.random.choice(2**n_qubits, size=shots, p=probs)
        
        # Convert to binary strings and count
        results = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{n_qubits}b')
            results[bitstring] = results.get(bitstring, 0) + 1
        
        return results

    def expectation_value(self, state: np.ndarray, operator: np.ndarray) -> complex:
        """Compute expectation value <Ψ|Op|Ψ>"""
        if len(state.shape) > 1:
            state = state.flatten()
        return np.vdot(state, operator @ state)