# Distributed Surface Code Simulation

This project implements a distributed quantum surface code simulation using **SquidASM** and **NetSquid**. It simulates a planar surface code split across multiple network nodes (a cluster), which collaboratively perform syndrome measurements and rely on a central coordinator for global decoding.

## 📌 Features

* **Distributed Architecture**: Divides a global qubit grid into smaller subgrids managed by distinct cluster nodes.
* **TeleGate Protocol**: Implements a globally canonical border protocol to perform stabilizer measurements across node boundaries using EPR pairs.
* **Spacetime Decoding**: Performs multiple rounds of stabilizer measurements (default: 2) and uses the XORed syndrome to handle measurement errors.
* **SVD Compression**: Each node locally compresses its parity-check matrix and syndrome using Singular Value Decomposition (SVD) to reduce communication overhead, keeping a configurable energy threshold (default: 95%).
* **Global OSD Decoder**: A centralized `coordinator` node assembles the reduced local systems into a block-diagonal global matrix and performs Ordered Statistic Decoding (OSD) over GF(2).

## 🗂️ Project Structure

* **`main.py`**: The entry point of the simulation. It configures the complete-graph network topology, initializes the layout manager, sets up the cluster nodes and the coordinator, and runs the simulation.
* **`surface_code.py`**: Contains the `SurfaceLayout` class. It manages the mapping of the global grid into local subgrids, assigning roles to qubits (`pQ` for data, `xQ` for X-stabilizers, `zQ` for Z-stabilizers) in a checkerboard pattern.
* **`dis_surface_code.py`** (`ClusterNodeProgram`): The program running on each local node. It handles:
    * Local qubit allocation and noise injection.
    * Local CNOT operations for stabilizers.
    * Cross-node CNOTs via the TeleGate border protocol.
    * Building the local parity-check matrices ($H_X$, $H_Z$).
    * SVD dimensionality reduction.
    * Applying corrections received from the coordinator.
* **`coordinator.py`** (`CoordinatorProgram`): The centralized decoder. It handles:
    * Receiving SVD-compressed payloads from all active nodes.
    * Assembling a global block-diagonal system.
    * Running Gaussian elimination and OSD over GF(2) to find the most probable error vector.
    * Back-projecting the reduced error vector to the physical data qubits.
    * Sending local correction instructions back to the cluster nodes.
    * Aggregating the final logical-Z parity to check for logical failures.

## 🚀 How to Run

### Prerequisites
Ensure you have Python installed along with the required quantum simulation libraries and math packages:
* `squidasm`
* `netsquid-netbuilder`
* `numpy`

### Execution
Run the main script from your terminal:

```bash
python main.py