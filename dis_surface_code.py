"""
Distributed Surface Code — ClusterNodeProgram
Full quantum protocol using local CNOT gates and teleported CNOT via EPR
for cross-border stabilizer measurements.

Syndrome measurement strategy

Border stabilizers (ancilla in node A, one data qubit in node B):
  Teleported CNOT protocol. Since the ancilla is already measured, the
  protocol computes only the parity contribution of the neighbor data qubit
  and XORs it into the already-stored ancilla_measurements value.

Teleported CNOT — parity extraction
-------------------------------------
determine the parity contribution of data qubit d_B (in node B)
to stabilizer ancilla q_A.

For xQ (ancilla controls, data is target):
  1. Node A creates EPR (eA, eB); sends eB to B.
  2. A sends role="xQ" to B.
  3. B: CNOT(eB → d_B); measures eB → m_B.
  4. A: measures eA in X basis (H then measure) → m_A.
     Parity XOR contribution = m_B.

For zQ (data controls, ancilla is target):
  1–2. Same EPR setup and role signal.
  3. B: CNOT(d_B → eB); measures eB → m_B.
  4. A: measures eA in Z basis → m_A.
     Parity XOR contribution = m_A.
"""

import json
import random
import numpy as np

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit


class ClusterNodeProgram(Program):
    # Constants
    NOISE_PROBABILITY = 0.01  # Bit-flip probability on data qubits
    ENERGY_THRESHOLD = 0.95   # SVD energy threshold for compression

    def __init__(self, node_coords: tuple, layout_manager, coordinator_name: str = "coordinator"):
        self.node_coords = node_coords
        self.layout_manager = layout_manager
        self.coordinator_name = coordinator_name

        r, c = node_coords
        N = layout_manager.nodes_per_side
        self.neighbors = []
        if r > 0:
            self.neighbors.append(f"node_{r-1}_{c}")
        if r < N-1:
            self.neighbors.append(f"node_{r+1}_{c}")
        if c > 0:
            self.neighbors.append(f"node_{r}_{c-1}")
        if c < N-1:
            self.neighbors.append(f"node_{r}_{c+1}")

    @property
    def meta(self) -> ProgramMeta:
        B = self.layout_manager.block_size
        return ProgramMeta(
            name=f"node_{self.node_coords[0]}_{self.node_coords[1]}",
            csockets=self.neighbors + [self.coordinator_name],
            epr_sockets=self.neighbors,
            max_qubits=B * B + B,
        )
    
    # Run
    def run(self, context: ProgramContext):
        conn = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        B = self.layout_manager.block_size

        self.injected_errors      = set()
        self.applied_corrections  = set()
        self.ancilla_measurements = {}

        # 1. Allocate qubits
        self.local_qubits, self.qubit_roles = [], []
        for row in subgrid_data:
            row_q, row_r = [], []
            for cell in row:
                row_q.append(Qubit(conn))
                row_r.append(cell["role"])
            self.local_qubits.append(row_q)
            self.qubit_roles.append(row_r)

        # 2. Apply noise (bit-flip on data qubits)
        for r in range(B):
            for c in range(B):
                if self.qubit_roles[r][c] == "pQ" and random.random() < self.NOISE_PROBABILITY:
                    self.local_qubits[r][c].X()
                    self.injected_errors.add((r, c))

        if self.injected_errors:
            print(f"[{self.node_coords}] Noise: {len(self.injected_errors)} qubit(s) → {sorted(self.injected_errors)}")
        else:
            print(f"[{self.node_coords}] Noise: none")

        # 3. Measure interior stabilizers.
        for r in range(B):
            for c in range(B):
                role = self.qubit_roles[r][c]
                if role not in ("xQ", "zQ"):
                    continue

                ancilla = self.local_qubits[r][c]

                if role == "xQ":
                    ancilla.H()

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < B and 0 <= nc < B and self.qubit_roles[nr][nc] == "pQ":
                        data = self.local_qubits[nr][nc]
                        if role == "xQ":
                            ancilla.cnot(data)
                        else:
                            data.cnot(ancilla)

                if role == "xQ":
                    ancilla.H()

                m = ancilla.measure()
                yield from conn.flush()
                self.ancilla_measurements[(r, c)] = int(m)

        # 4. Border stabilizer contributions via teleported CNOT.
        border_xors = yield from self._teleported_cnot_borders(context)
        for (r, c), bit in border_xors.items():
            self.ancilla_measurements[(r, c)] = (
                self.ancilla_measurements.get((r, c), 0) ^ bit
            )

        # 5. Build SVD payload, communicate with coordinator, apply corrections
        payload = self._build_svd_payload()
        corrections = yield from self._communicate_with_coordinator(context, payload)
        self._apply_corrections(corrections)

        if self.applied_corrections:
            print(f"[{self.node_coords}] Corrections: {sorted(self.applied_corrections)}")
        else:
            print(f"[{self.node_coords}] Corrections: none")

        # 6. Send logical-Z parity to coordinator
        yield from self._send_logical_parity(context)
        yield from conn.flush()

    # Step 4 — Teleported CNOT border protocol
    def _teleported_cnot_borders(self, context) -> dict:
        """
        Process all four border directions and return a dict of
        {(local_r, local_c): parity_bit} for ancilla-side nodes.
        Directions are processed in two waves to keep partner nodes in sync.
        """
        r_node, c_node = self.node_coords
        N = self.layout_manager.nodes_per_side
        B = self.layout_manager.block_size
        xors = {}

        # Wave 1: horizontal (RIGHT then LEFT)
        if c_node < N - 1:
            result = yield from self._run_border_direction(context,
                neighbor = f"node_{r_node}_{c_node+1}",
                is_ancilla_side = True,
                axis = "col",
                local_fixed = B - 1,
            )
            xors.update(result)

        if c_node > 0:
            yield from self._run_border_direction(context,
                neighbor = f"node_{r_node}_{c_node-1}",
                is_ancilla_side = False,
                axis = "col",
                local_fixed = 0,
            )

        # Wave 2: vertical (DOWN then UP)
        if r_node < N - 1:
            result = yield from self._run_border_direction(context,
                neighbor = f"node_{r_node+1}_{c_node}",
                is_ancilla_side = True,
                axis = "row",
                local_fixed = B - 1,
            )
            xors.update(result)

        if r_node > 0:
            yield from self._run_border_direction(context,
                neighbor = f"node_{r_node-1}_{c_node}",
                is_ancilla_side = False,
                axis = "row",
                local_fixed = 0,
            )

        return xors

    def _run_border_direction(self, context, *, neighbor: str, is_ancilla_side: bool, axis: str, local_fixed: int) -> dict:
        """
        Process one border direction index by index.

        Each index goes through 4 synchronized rounds:
          A. EPR setup: create_keep / recv_keep
          B. Role signal: send/recv "xQ"/"zQ"/"skip"
          C. Gate + measure on both sides
          D. Classical bit exchange
        """
        conn = context.connection
        B = self.layout_manager.block_size
        csock = context.csockets[neighbor]
        epr_sock = context.epr_sockets[neighbor]
        xor_bits = {}

        for idx in range(B):
            r_loc = local_fixed if axis == "row" else idx
            c_loc = idx if axis == "row" else local_fixed

            if is_ancilla_side:
                anc_role = self.qubit_roles[r_loc][c_loc]
                role_signal = anc_role if anc_role in ("xQ", "zQ") else "skip"

                # Round A: create EPR
                eA = epr_sock.create_keep()[0]
                yield from conn.flush()

                # Round B: send role signal
                csock.send(role_signal)
                yield from conn.flush()

                if role_signal == "skip":
                    # Round C: free the EPR qubit
                    eA.measure()
                    yield from conn.flush()
                    # Round D: consume dummy ack
                    yield from csock.recv()
                    continue

                # Round C: measure eA in Z basis
                m_A = eA.measure()
                yield from conn.flush()
                m_A_val = int(m_A)

                # Round D: receive m_B from data side
                m_B_val = int((yield from csock.recv()))

                xor_bits[(r_loc, c_loc)] = m_A_val ^ m_B_val

            else:
                # Round A: receive EPR half
                eB = epr_sock.recv_keep()[0]
                yield from conn.flush()

                # Round B: receive role signal
                signal = yield from csock.recv()
                yield from conn.flush()

                if signal == "skip" or self.qubit_roles[r_loc][c_loc] != "pQ":
                    eB.measure()
                    yield from conn.flush()
                    csock.send("0")
                    continue

                data = self.local_qubits[r_loc][c_loc]

                # Round C: Apply teleported parity gate and measure eB.
                data.cnot(eB)

                m_B = eB.measure()
                # Flush to ensure measurement result is available before conversion
                yield from conn.flush()

                # Round D: send m_B to ancilla side
                csock.send(str(int(m_B)))

        return xor_bits

    # Step 5 — SVD payload
    def _build_local_system(self) -> tuple:
        #Build local H matrix and syndrome vector for this node.
        
        B = self.layout_manager.block_size
        r_node, c_node = self.node_coords
        d_pos, a_pos = [], []
        
        # Identify qubits and their global positions
        for r in range(B):
            for c in range(B):
                # Convert local to global coordinates
                gr, gc = r_node * B + r, c_node * B + c
                role = self.qubit_roles[r][c]
                if role == "pQ":
                    d_pos.append((gr, gc))
                elif role in ("xQ", "zQ"):
                    a_pos.append((gr, gc))
        
        # Build H matrix: entries are 1 when ancilla and data are neighbors
        H = np.zeros((len(a_pos), len(d_pos)), dtype=float)
        for i, (ar, ac) in enumerate(a_pos):
            for j, (dr, dc) in enumerate(d_pos):
                if abs(ar - dr) + abs(ac - dc) == 1:
                    H[i, j] = 1.0
        
        # Build syndrome vector from ancilla measurements in local coordinates
        s = np.array([self.ancilla_measurements.get((p[0] % B, p[1] % B), 0) 
                      for p in a_pos], dtype=int)
        return H, s, d_pos

    def _build_svd_payload(self, energy_threshold: float = None) -> dict:
        if energy_threshold is None:
            energy_threshold = self.ENERGY_THRESHOLD
        
        H, s, data_pos = self._build_local_system()

        # 1. Early exit: if no syndrome, node doesn't participate in global correction
        if H.size == 0 or not np.any(s):
            return {
                "active": False, 
                "node_id": list(self.node_coords),
                "data_positions": [list(p) for p in data_pos]
            }

        # 2. Compute SVD and determine compression rank
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
        n = H.shape[1]
        
        U, sigma, Vt = np.linalg.svd(H, full_matrices=False)

        # Determine rank k based on energy threshold
        total_energy = np.sum(sigma ** 2)
        k = n
        if total_energy > 1e-10:
            cumulative = np.cumsum(sigma ** 2)
            k = int(np.searchsorted(cumulative, energy_threshold * total_energy) + 1)
            k = min(k, n)

        # Apply SVD compression if beneficial
        if k < n:
            H_reduced = U[:, :k] * sigma[:k]
            V_k = Vt[:k, :].T
        else:
            # No compression: send H directly with identity projection
            H_reduced = H
            V_k = np.eye(n)

        return {
            "active":         True,
            "node_id":        list(self.node_coords),
            "H_reduced":      H_reduced.tolist(),
            "V_k":            V_k.tolist(),
            "s":              s.tolist(),
            "k":              k,
            "data_positions": [list(p) for p in data_pos],
            "global_offset":  [r_node * B, c_node * B],
        }

    def _communicate_with_coordinator(self, context, payload: dict) -> dict:
        csock = context.csockets[self.coordinator_name]
        csock.send(json.dumps(payload))
        yield from context.connection.flush()
        msg = yield from csock.recv()
        return json.loads(msg)

    def _apply_corrections(self, corrections: list):
        """Apply corrections from coordinator in global coordinates.
        
        Filters corrections to those within this node's block and converts
        from global to local coordinates before applying X corrections.
        """
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
        
        # Calculate this node's global coordinate boundaries
        r_start = r_node * B
        r_end = (r_node + 1) * B
        c_start = c_node * B
        c_end = (c_node + 1) * B

        for r_global, c_global in corrections:
            # Check if correction is within this node's block
            if r_start <= r_global < r_end and c_start <= c_global < c_end:
                # Convert to local coordinates
                r_local = r_global - r_start
                c_local = c_global - c_start
                
                # Apply correction only to data qubits (pQ)
                if self.qubit_roles[r_local][c_local] == "pQ":
                    key = (r_local, c_local)
                    self.applied_corrections.add(key)

    # Step 6 — Logical-Z parity

    def _send_logical_parity(self, context) -> None:
        """Send logical-Z parity to coordinator.
        
        Only the left-column node (c_node == 0) computes and sends the actual
        parity. Other nodes send -1 placeholder to maintain synchronization.
        """
        csock = context.csockets[self.coordinator_name]
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size

        if c_node != 0:
            # Non-boundary node: send placeholder
            csock.send(json.dumps(-1))
        else:
            # Boundary node: compute parity from left column
            parity = 0
            for r in range(B):
                if self.qubit_roles[r][0] == "pQ":
                    # XOR with injected error and applied corrections
                    parity ^= int((r, 0) in self.injected_errors)
                    parity ^= int((r, 0) in self.applied_corrections)
            csock.send(json.dumps(parity))

        yield from context.connection.flush()