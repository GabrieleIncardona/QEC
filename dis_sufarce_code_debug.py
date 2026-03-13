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

        # 2. Noise
        for r in range(B):
            for c in range(B):
                if self.qubit_roles[r][c] == "pQ" and random.random() < self.NOISE_PROBABILITY:
                    self.local_qubits[r][c].X()
                    self.injected_errors.add((r, c))

        print(f"[{self.node_coords}] Noise: {len(self.injected_errors) if self.injected_errors else 'none'}")

        """
        directions = ["North", "West", "East", "South"] # L'ordine standard (N, W, E, S)

        for direction in directions:
            # 1. Applica i CNOT per i vicini LOCALI in questa direzione
            self._apply_local_cnots(direction)
            
            # 2. Applica i CNOT per i vicini REMOTI (TeleGate) in questa direzione
            yield from self._apply_border_cnots(direction)
            
            # 3. Flush per mantenere tutti i nodi perfettamente in sincrono a ogni step
            yield from conn.flush()
        """

        # --- Phase 3: Local gates (interior neighbors only, no measurement yet) ---
        for r in range(B):
            for c in range(B):
                role = self.qubit_roles[r][c]
                if role not in ("xQ", "zQ"): continue
                ancilla = self.local_qubits[r][c]
                if role == "xQ": ancilla.H()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < B and 0 <= nc < B and self.qubit_roles[nr][nc] == "pQ":
                        data = self.local_qubits[nr][nc]
                        if role == "xQ": ancilla.cnot(data)
                        else: data.cnot(ancilla)
        yield from conn.flush()

        # --- Phase 4: TeleGate border protocol ---
        # Completes stabilizer CNOTs with remote data qubits via Cat-Ent / Cat-DisEnt.
        # Pauli corrections applied directly onto live ancilla/data qubits.
        yield from self._teleported_cnot_borders(context)

        # --- Phase 5: Measure all ancillas ---
        for r in range(B):
            for c in range(B):
                role = self.qubit_roles[r][c]
                if role not in ("xQ", "zQ"): continue
                ancilla = self.local_qubits[r][c]
                if role == "xQ": ancilla.H()
                m = ancilla.measure()
                yield from conn.flush()
                self.ancilla_measurements[(r, c)] = int(m)
                #print(f"[{self.node_coords}] Measured ancilla at local ({r},{c}) with role {role}: {int(m)}")
    

        # --- FASE 4: Correzione e Comunicazione ---
        payload = self._build_svd_payload()
        corrections = yield from self._communicate_with_coordinator(context, payload)
        self._apply_corrections(corrections)

        if self.applied_corrections:
            print(f"[{self.node_coords}] Corrections: {sorted(self.applied_corrections)}")
        else:
            print(f"[{self.node_coords}] Corrections: none")

        # 5. Fine
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
        conn     = context.connection
        B        = self.layout_manager.block_size
        csock    = context.csockets[neighbor]
        epr_sock = context.epr_sockets[neighbor]

        for idx in range(B):
            r_loc, c_loc = (local_fixed, idx) if axis == "col" else (idx, local_fixed)

            if is_ancilla_side:
                anc_role    = self.qubit_roles[r_loc][c_loc]
                role_signal = anc_role if anc_role in ("xQ", "zQ") else "skip"
                ancilla     = self.local_qubits[r_loc][c_loc]

                eA = epr_sock.create_keep()[0]
                csock.send(role_signal)
                yield from conn.flush()

                if role_signal == "skip":
                    eA.measure()
                    yield from conn.flush()
                    yield from csock.recv()  # dummy ack
                    continue

                if role_signal == "xQ":
                    # Cat-Ent A: CNOT(anc->eA), measure eA in Z, send m_A
                    ancilla.cnot(eA)
                    m_A = eA.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_A)))
                    yield from conn.flush()

                    # Cat-DisEnt A: recv m_B, apply Z^m_B on ancilla
                    m_B = int((yield from csock.recv()))
                    if m_B == 1:
                        ancilla.Z()
                    

                else:  # zQ
                    # Cat-Ent A: recv m_B from B, X^m_B on eA, CNOT(eA->anc)
                    m_B = int((yield from csock.recv()))
                    if m_B == 1:
                        eA.X()
                    eA.cnot(ancilla)

                    # Cat-DisEnt A: measure eA in X, send m_A
                    eA.H()
                    m_A = eA.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_A)))
                    yield from conn.flush()
                    
            else:
                eB = epr_sock.recv_keep()[0]
                yield from conn.flush()
                signal = yield from csock.recv()

                if signal == "skip" or self.qubit_roles[r_loc][c_loc] != "pQ":
                    eB.measure()
                    yield from conn.flush()
                    csock.send("0")
                    yield from conn.flush()
                    continue

                data = self.local_qubits[r_loc][c_loc]

                if signal == "xQ":
                    # Cat-Ent B: recv m_A, X^m_A on eB, CNOT(eB->data)
                    m_A = int((yield from csock.recv()))
                    if m_A == 1:
                        eB.X()
                    eB.cnot(data)

                    # Cat-DisEnt B: measure eB in X, send m_B
                    eB.H()
                    m_B = eB.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_B)))
                    yield from conn.flush()

                else:  # zQ
                    # Cat-Ent B: CNOT(data->eB), measure eB in Z, send m_B
                    data.cnot(eB)
                    m_B = eB.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_B)))
                    yield from conn.flush()

                    # Cat-DisEnt B: recv m_A, apply Z^m_A on data
                    m_A = int((yield from csock.recv()))
                    if m_A == 1:
                        data.Z()
                    # print (f"[{self.node_coords}] Border {signal} parity contribution from data at local ({r_loc},{c_loc}): {data.measure()}")

        return {}
    
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
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
        r_start, c_start = r_node * B, c_node * B

        for r_global, c_global in corrections:
            if r_start <= r_global < r_start + B and c_start <= c_global < c_start + B:
                r_local = r_global - r_start
                c_local = c_global - c_start
                
                if self.qubit_roles[r_local][c_local] == "pQ":
                    # APPLICAZIONE FISICA
                    self.local_qubits[r_local][c_local].X() 
                    self.applied_corrections.add((r_local, c_local))

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