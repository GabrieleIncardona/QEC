import json
import random
import numpy as np

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit


class ClusterNodeProgram(Program):
    NOISE_PROBABILITY = 1.0        # probability of X error on each data qubit before stabilizer measurements
    ENERGY_THRESHOLD  = 0.95        # fraction of total energy to retain in SVD dimensionality reduction (0 < threshold <= 1)
    NUM_ROUNDS        = 2           # number of rounds of stabilizer measurements (for spacetime decoding)

    def __init__(self, node_coords: tuple, layout_manager,
                 coordinator_name: str = "coordinator"):
        self.node_coords = node_coords
        self.layout_manager = layout_manager
        self.coordinator_name = coordinator_name

        r, c = node_coords
        N = layout_manager.nodes_per_side
        self.neighbors = []
        # calculate the neighboring nodes in the 2D grid (up to 4 neighbors: up, down, left, right)
        if r > 0: self.neighbors.append(f"node_{r-1}_{c}")
        if r < N - 1: self.neighbors.append(f"node_{r+1}_{c}")
        if c > 0: self.neighbors.append(f"node_{r}_{c-1}")
        if c < N - 1: self.neighbors.append(f"node_{r}_{c+1}")

    @property
    def meta(self) -> ProgramMeta:
        B = self.layout_manager.block_size
        return ProgramMeta(
            name = f"node_{self.node_coords[0]}_{self.node_coords[1]}",
            csockets = self.neighbors + [self.coordinator_name],
            epr_sockets = self.neighbors,
            max_qubits  = B * B + B,    # Max qubits: all physical qubits (B²) + EPR qubits on one border (B) for TeleGate protocol
        )

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #
    def run(self, context: ProgramContext):
        conn = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        B  = self.layout_manager.block_size

        self.injected_errors = set()        # Track which data qubits received noise-induced X errors
        self.applied_corrections = set()    # Track which data qubits received X corrections from the coordinator

        # 1. Allocate qubits
        self.local_qubits, self.qubit_roles = [], []
        for row in subgrid_data:
            row_q, row_r = [], []
            for cell in row:
                row_q.append(Qubit(conn))       # Allocate a qubit for every cell in the subgrid (data and ancilla)
                row_r.append(cell["role"])      # Store the role ("pQ", "xQ", "zQ") for each qubit for later reference
            self.local_qubits.append(row_q)
            self.qubit_roles.append(row_r)

        # ── 3. Two rounds of stabilizer measurements ──────────────────────
        # The gate data.Z() is applied to data qubits on the border during
        # Cat-DisEnt zQ — this is correct and necessary for the protocol.
        #
        # Spacetime decoding with classical feedback compensation:
        #   Between round 1 and round 2, Z gates accumulated on data qubits
        #   change the measurement of adjacent xQ ancillas in round 2.
        #   We maintain a classical register `z_parity[r][c]` that counts (mod 2)
        #   how many Z gates each data qubit has received so far.
        #   Before each xQ measurement in round 2, if the neighboring data qubit
        #   has odd z_parity, we flip the corresponding xQ syndrome.
        #
        # z_parity[r][c]: Parity of Z gates accumulated on data qubit (r,c)
        #   0 = even number of Z gates (no net effect)
        #   1 = odd number of Z gates (equivalent to active Z gate)

        z_parity = [[0] * B for _ in range(B)]  # Classical register tracking Z gates
        all_round_syndromes = []

        for round_idx in range(self.NUM_ROUNDS):

            if round_idx == 1:
                # 2. Apply noise
                for r in range(B):
                    for c in range(B):
                        if (self.qubit_roles[r][c] == "pQ"
                                and random.random() < self.NOISE_PROBABILITY):
                            self.local_qubits[r][c].X()
                            self.injected_errors.add((r, c))
                print(f"[{self.node_coords}] Noise: "
                f"{len(self.injected_errors) if self.injected_errors else 'none'}")

            # a) Re-allocate ancillas (data qubits unchanged)
            for r in range(B):
                for c in range(B):
                    if self.qubit_roles[r][c] in ("xQ", "zQ"):
                        self.local_qubits[r][c] = Qubit(conn)

            # b) Local gates
            for r in range(B):
                for c in range(B):
                    role = self.qubit_roles[r][c]
                    if role not in ("xQ", "zQ"):
                        # Only ancillas participate in local gates; data qubits are idle
                        # (except Z gates from TeleGate border protocol, handled separately and tracked in z_parity)
                        continue
                    ancilla = self.local_qubits[r][c]
                    if role == "xQ":
                        ancilla.H()  # Prepare xQ ancillas in |+⟩ state for X parity measurement
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Local CNOTs with neighboring data qubits
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < B and 0 <= nc < B
                                and self.qubit_roles[nr][nc] == "pQ"):
                            data = self.local_qubits[nr][nc]
                            if role == "xQ":
                                ancilla.cnot(data)
                            else:
                                data.cnot(ancilla)
            yield from conn.flush()

            # c) Original TeleGate protocol — returns Z gates applied this round
            round_z = yield from self._teleported_cnot_borders(context)

            # d) Update z_parity with Z gates from this round
            for (r, c) in round_z:
                z_parity[r][c] ^= 1

            # e) Measure ancillas with Z compensation for xQ
            # For each xQ ancilla, if a neighboring data qubit has z_parity==1,
            # its measurement in the X basis will be flipped → we compensate by XORing
            # the result with the contribution from each Z-active neighbor.
            round_syndrome = {}
            for r in range(B):
                for c in range(B):
                    role = self.qubit_roles[r][c]
                    if role not in ("xQ", "zQ"):
                        continue
                    ancilla = self.local_qubits[r][c]
                    if role == "xQ":
                        ancilla.H()
                    m = ancilla.measure()
                    yield from conn.flush()
                    raw = int(m)

                    # Z compensation for xQ: Z anticommutes with X
                    # Each neighboring data qubit with odd z_parity contributes
                    # a flip to the xQ measurement (because Z|ψ⟩ in X basis → flip).
                    if role == "xQ":
                        z_flip = 0
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < B and 0 <= nc < B
                                    and self.qubit_roles[nr][nc] == "pQ"
                                    and z_parity[nr][nc] == 1):
                                z_flip ^= 1
                        round_syndrome[(r, c)] = raw ^ z_flip
                    else:
                        # zQ measures in Z basis: Z commutes with Z → no compensation
                        round_syndrome[(r, c)] = raw

            all_round_syndromes.append(round_syndrome)
            active = {k: v for k, v in round_syndrome.items() if v == 1}
            print(f"[{self.node_coords}] Round {round_idx + 1} syndrome: "
                  f"{active if active else 'clean'}")

        # ── 4. Spacetime decoding: Calculation of all detection events ──────────
        self.detection_events = []  # List storing detection event dictionaries for each round
        
        # Create a virtual "Round 0" where all syndromes are 0
        previous_syndrome = {pos: 0 for pos in all_round_syndromes[0]}
        
        for round_idx, current_syndrome in enumerate(all_round_syndromes):
            # E_t = s_t XOR s_{t-1}
            diff = {pos: current_syndrome[pos] ^ previous_syndrome[pos] for pos in current_syndrome}
            self.detection_events.append(diff)
            
            # Prepare previous_syndrome for the next iteration
            previous_syndrome = current_syndrome
            
            # Debug printout
            active = {k: v for k, v in diff.items() if v == 1}
            print(f"[{self.node_coords}] Detection events Round {round_idx + 1}: "
                  f"{active if active else 'clean'}")

        # ── 5. SVD payload and corrections ────────────────────────────────
        payload = self._build_svd_payload()
        corrections = yield from self._communicate_with_coordinator(context, payload)
        self._apply_corrections(corrections)

        if self.applied_corrections:
            print(f"[{self.node_coords}] Corrections: {sorted(self.applied_corrections)}")
        else:
            print(f"[{self.node_coords}] Corrections: none")

        # ── 6. Logical-Z parity ───────────────────────────────────────────
        yield from self._send_logical_parity(context)
        yield from conn.flush()

    # ------------------------------------------------------------------ #
    #  TeleGate border protocol                                            #
    # ------------------------------------------------------------------ #
    def _teleported_cnot_borders(self, context):
        """
        Returns round_z: set of (r_loc, c_loc) coordinates of local data qubits
        that received data.Z() an odd number of times this round.
        """
        r_node, c_node = self.node_coords
        N = self.layout_manager.nodes_per_side
        B = self.layout_manager.block_size
        round_z = set()

        if c_node < N - 1:
            z_set = yield from self._run_border_direction(
                context,
                neighbor        = f"node_{r_node}_{c_node + 1}",
                is_ancilla_side = True,
                axis            = "col",
                local_fixed     = B - 1,
            )
            round_z ^= z_set

        if c_node > 0:
            z_set = yield from self._run_border_direction(
                context,
                neighbor        = f"node_{r_node}_{c_node - 1}",
                is_ancilla_side = False,
                axis            = "col",
                local_fixed     = 0,
            )
            round_z ^= z_set

        if r_node < N - 1:
            z_set = yield from self._run_border_direction(
                context,
                neighbor        = f"node_{r_node + 1}_{c_node}",
                is_ancilla_side = True,
                axis            = "row",
                local_fixed     = B - 1,
            )
            round_z ^= z_set

        if r_node > 0:
            z_set = yield from self._run_border_direction(
                context,
                neighbor        = f"node_{r_node - 1}_{c_node}",
                is_ancilla_side = False,
                axis            = "row",
                local_fixed     = 0,
            )
            round_z ^= z_set

        return round_z

    def _run_border_direction(self, context, *, neighbor: str,
                               is_ancilla_side: bool, axis: str,
                               local_fixed: int):
        """
        Original TeleGate Cat-Ent/Cat-DisEnt protocol.
        Returns z_applied: set of (r_loc, c_loc) coordinates of local data qubits
        that received data.Z() an odd number of times.

        ANCILLA side (xQ):
            Cat-Ent A:    CNOT(anc→eA), meas eA in Z, send m_A
            Cat-DisEnt A: recv m_B, Z^m_B on anc

        ANCILLA side (zQ):
            Cat-Ent A:    recv m_B, X^m_B on eA, CNOT(eA→anc)
            Cat-DisEnt A: meas eA in X, send m_A

        DATA side (xQ):
            Cat-Ent B:    recv m_A, X^m_A on eB, CNOT(eB→data)
            Cat-DisEnt B: meas eB in X, send m_B

        DATA side (zQ):
            Cat-Ent B:    CNOT(data→eB), meas eB in Z, send m_B
            Cat-DisEnt B: recv m_A, Z^m_A on data  ← tracked in z_applied
        """
        conn = context.connection
        B = self.layout_manager.block_size
        csock = context.csockets[neighbor]
        epr_sock = context.epr_sockets[neighbor]
        z_applied = set()

        for idx in range(B):
            r_loc, c_loc = (local_fixed, idx) if axis == "col" else (idx, local_fixed)

            if is_ancilla_side:
                anc_role = self.qubit_roles[r_loc][c_loc]

                # Check if there is a remote adjacent data qubit in this direction.
                # The remote data qubit is on the opposite border of the neighboring node,
                # at the same position along the border (same idx).
                # For axis=col: the remote neighbor is in the horizontal direction,
                #   so the remote data qubit is adjacent only if anc_role=="xQ"
                #   (xQ measures X parity with left/right neighbors).
                # For axis=row: the remote neighbor is in the vertical direction,
                #   so the remote data qubit is adjacent only if anc_role=="zQ"
                #   (zQ measures Z parity with up/down neighbors).
                # If the ancilla has no remote neighbors in the correct direction → skip.
                if anc_role == "xQ" and axis == "row":
                    role_signal = "skip"  # xQ has no vertical remote neighbors
                elif anc_role == "zQ" and axis == "col":
                    role_signal = "skip"  # zQ has no horizontal remote neighbors
                elif anc_role in ("xQ", "zQ"):
                    role_signal = anc_role
                else:
                    role_signal = "skip"

                ancilla = self.local_qubits[r_loc][c_loc]

                eA = epr_sock.create_keep()[0]
                csock.send(role_signal)
                yield from conn.flush()

                if role_signal == "skip":
                    eA.measure()
                    yield from conn.flush()
                    yield from csock.recv()
                    continue

                if role_signal == "xQ":
                    ancilla.cnot(eA)
                    m_A = eA.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_A)))
                    yield from conn.flush()
                    m_B = int((yield from csock.recv()))
                    if m_B == 1:
                        ancilla.Z()

                else:  # zQ
                    m_B = int((yield from csock.recv()))
                    if m_B == 1:
                        eA.X()
                    eA.cnot(ancilla)
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
                    m_A = int((yield from csock.recv()))
                    if m_A == 1:
                        eB.X()
                    eB.cnot(data)
                    eB.H()
                    m_B = eB.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_B)))
                    yield from conn.flush()

                else:  # zQ
                    data.cnot(eB)
                    m_B = eB.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_B)))
                    yield from conn.flush()
                    m_A = int((yield from csock.recv()))
                    if m_A == 1:
                        data.Z()
                        # Track Z gate (XOR: two Z gates cancel out)
                        if (r_loc, c_loc) in z_applied:
                            z_applied.discard((r_loc, c_loc))
                        else:
                            z_applied.add((r_loc, c_loc))

        return z_applied

    # ------------------------------------------------------------------ #
    #  SVD payload                                                         #
    # ------------------------------------------------------------------ #
    def _build_local_system(self) -> tuple:
        B = self.layout_manager.block_size
        r_node, c_node = self.node_coords
        d_pos, a_pos = [], []

        for r in range(B):
            for c in range(B):
                gr, gc = r_node * B + r, c_node * B + c
                role   = self.qubit_roles[r][c]
                if role == "pQ":
                    d_pos.append((gr, gc))
                elif role in ("xQ", "zQ"):
                    a_pos.append((gr, gc))

        H = np.zeros((len(a_pos), len(d_pos)), dtype=float)
        for i, (ar, ac) in enumerate(a_pos):
            for j, (dr, dc) in enumerate(d_pos):
                if abs(ar - dr) + abs(ac - dc) == 1:
                    H[i, j] = 1.0

        # ---- SPACETIME SYNDROME HISTORY MODIFICATION ----
        # Build the history of syndromes in time
        s_history = []
        for events_dict in self.detection_events:
            # Extract the measurements for this specific round
            s_t = [events_dict.get((p[0] % B, p[1] % B), 0) for p in a_pos]
            s_history.append(s_t)

        # s is now a 2D matrix of shape (NUM_ROUNDS, len(a_pos))
        s = np.array(s_history, dtype=int) 
        # ------------------------------------------------

        return H, s, d_pos

    def _build_svd_payload(self, energy_threshold: float = None) -> dict:
        if energy_threshold is None:
            energy_threshold = self.ENERGY_THRESHOLD

        H, s, data_pos = self._build_local_system()

        # np.any(s) works perfectly even if s is a 2D matrix.
        # Returns False only if ALL bits in ALL rounds are 0.
        if H.size == 0 or not np.any(s):
            return {
                "active":         False,
                "node_id":        list(self.node_coords),
                "data_positions": [list(p) for p in data_pos],
            }

        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
        n = H.shape[1]

        # SVD is purely spatial and remains unchanged!
        U, sigma, Vt = np.linalg.svd(H, full_matrices=False)

        total_energy = np.sum(sigma ** 2)
        k = n
        if total_energy > 1e-10:
            cumulative = np.cumsum(sigma ** 2)
            k = int(np.searchsorted(cumulative,
                                    energy_threshold * total_energy) + 1)
            k = min(k, n)

        if k < n:
            H_reduced = U[:, :k] * sigma[:k]
            V_k       = Vt[:k, :].T
        else:
            H_reduced = H
            V_k       = np.eye(n)

        return {
            "active":         True,
            "node_id":        list(self.node_coords),
            "H_reduced":      H_reduced.tolist(),
            "V_k":            V_k.tolist(),
            "s":              s.tolist(),  # Produces a 2D list [[round1_bits], [round2_bits], ...] suitable for JSON 
            "k":              k,
            "data_positions": [list(p) for p in data_pos],
            "global_offset":  [r_node * B, c_node * B],
        }

    def _communicate_with_coordinator(self, context, payload: dict):
        csock = context.csockets[self.coordinator_name]
        csock.send(json.dumps(payload))
        yield from context.connection.flush()
        msg = yield from csock.recv()
        return json.loads(msg)

    def _apply_corrections(self, corrections: list):
        r_node, c_node   = self.node_coords
        B                = self.layout_manager.block_size
        r_start, c_start = r_node * B, c_node * B

        for r_global, c_global in corrections:
            if (r_start <= r_global < r_start + B
                    and c_start <= c_global < c_start + B):
                r_local = r_global - r_start
                c_local = c_global - c_start
                if self.qubit_roles[r_local][c_local] == "pQ":
                    self.local_qubits[r_local][c_local].X()  # Apply X correction (bit-flip)
                    self.applied_corrections.add((r_local, c_local))

    # ------------------------------------------------------------------ #
    #  Logical-Z parity                                                    #
    # ------------------------------------------------------------------ #
    def _send_logical_parity(self, context):
        csock = context.csockets[self.coordinator_name]
        r_node, c_node = self.node_coords
        B  = self.layout_manager.block_size

        if c_node != 0:
            csock.send(json.dumps(-1))
        else:
            parity = 0
            for r in range(B):
                if self.qubit_roles[r][0] == "pQ":
                    parity ^= int((r, 0) in self.injected_errors)
                    parity ^= int((r, 0) in self.applied_corrections)
            csock.send(json.dumps(parity))

        yield from context.connection.flush()