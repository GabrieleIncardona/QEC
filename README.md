# Distributed Surface Code Simulation with SquidASM
This repository implements a Distributed Surface Code architecture using the SquidASM framework. The project demonstrates how a large-scale error-correcting code can be partitioned across multiple quantum nodes, using entanglement to perform stabilizer measurements across physical boundaries.

## 1. Distributed Grid Architecture
In this implementation, the global Surface Code is not a single monolithic circuit. Instead, it is a tessellation of subgrids, where each node in a network operates a specific "tile" of the total code.
## Node-Subgrid Mapping
- **Nodal Autonomy**: Each node manages its own set of physical data qubits and ancillas.
- **Layout Manager**: A global manager assigns roles (pQ, zQ, xQ) to qubits based on their global coordinates, ensuring that the union of all nodes forms a valid Surface Code.
- **Boundary Intersections**: Stabilizers located at the edges of a node’s subgrid must interact with data qubits located in a neighbor's subgrid.

## 2. Distributed Operation Model
The core of the project is the Entanglement-Assisted CNOT protocol, which allows an ancilla in Node A to perform a parity check on a data qubit in Node B.
## The 5-Phase Cycle
1. **EPR Link Establishment:** Pairs of entangled qubits are generated and distributed between adjacent nodes. These EPR pairs serve as the "quantum bridges" between subgrids.
2. **Local Resource Initialization:** Physical qubits are allocated locally within each node.
3. **Cross-Node Parity Checks:** Ancilla qubits perform local gates and then interact with the shared EPR pairs. By measuring the EPR half, the "control" information is projected across the network boundary.
4. **Classical Communication Layer:** Nodes exchange measurement results via classical channels. This layer is synchronized to ensure that every node has the necessary information from its neighbors before proceeding.
5. **Distributed Gate Completion:** Based on received classical data, nodes apply local Pauli corrections to their data qubits, successfully completing the multi-node CNOT gates.

## 3. Features & Scalability
- **Seamless Boundary Logic:** The system automatically identifies if a neighbor exists. If a node is on the edge of the total grid, it correctly treats the stabilizer as a "physical boundary" (weight-2), whereas internal node boundaries are treated as "distributed links" (weight-4).
- **Asynchronous-Friendly Design:** By separating the quantum measurement phase from the classical correction phase, the implementation allows for efficient processing of classical bitstreams across the distributed network.
- **Modular Codebase:** The implementation is designed to scale with the code distance d. By increasing the block_size or the number of nodes, one can simulate larger Surface Codes without changing the underlying nodal logic.

## 4. Setup and Execution
The simulation requires the squidasm and netqasm libraries. It is configured to run on a simulated topology where nodes are connected in a 2D mesh, matching the logical structure of the Surface Code.
