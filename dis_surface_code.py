import json
import random
import numpy as np

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit


class ClusterNodeProgram(Program):
    NOISE_PROBABILITY = 0.01         # probability of X error on each data qubit before stabilizer measurements
    ENERGY_THRESHOLD  = 0.95        # fraction of total energy to retain in SVD dimensionality reduction (0 < threshold <= 1)
    NUM_ROUNDS        = 2           # number of rounds of stabilizer measurements (for spacetime decoding)

    def __init__(self, node_coords: tuple, layout_manager, error,
                 coordinator_name: str = "coordinator"):
        self.node_coords = node_coords
        self.layout_manager = layout_manager
        self.coordinator_name = coordinator_name
        self.error = error

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
            max_qubits  = B * B + B,    # Max qubits needed: all phisical qubits (B^2) + all epr qubit on one border (B) for TeleGate protocol
        )

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #
    def run(self, context: ProgramContext):
        conn = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        B  = self.layout_manager.block_size

        self.injected_X_errors = set()
        self.injected_Z_errors = set()
        self.applied_X_corrections = set()
        self.applied_Z_corrections = set()

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
        # Cat-DisEnt zQ.
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
        x_parity = [[0] * B for _ in range(B)]  # Classical register tracking X gates
        tele_flip = {}   # (r,c) → parity of Z() byproducts on xQ ancilla: flips X-basis syndrome
        all_round_syndromes = []

        for round_idx in range(self.NUM_ROUNDS):
            # Apply noise only before the first round, so that the second round's syndrome reflects the effect of noise + any Z gates from TeleGate.
            if round_idx == 1:
                # 2. Apply noise
                match self.error:
                    case "identity":
                        for r in range(B):
                            for c in range(B):
                                if (self.qubit_roles[r][c] == "pQ"
                                        and random.random() < self.NOISE_PROBABILITY):
                                    self.local_qubits[r][c].X()
                                    self.injected_X_errors.add((r, c))
                        print(f"[{self.node_coords}] Noise: "
                        f"{len(self.injected_X_errors) if self.injected_X_errors else 'none'}")
                    case "hadamard":
                        for r in range(B):
                            for c in range(B):
                                if self.qubit_roles[r][c]== "pq":
                                    # simulation a hadamard error
                                    self._noisy_H(self.local_qubits[r][c], r, c)
                                    # applay 2 time hadamard, in this way, HIH = I, but we applay error
                                    self._noisy_H(self.local_qubits[r][c], r, c)


                    case "none":
                        print(f"[{self.node_coords}] Ideal simulation: no noise applied.")

                    case _:
                        print(f"[{self.node_coords}] WARNING: Unknown error type '{self.error}'. Skipping noise.")

            # a) Re-allocate ancillas
            for r in range(B):
                for c in range(B):
                    if self.qubit_roles[r][c] in ("xQ", "zQ"):
                        self.local_qubits[r][c] = Qubit(conn)

            # b) Local gates
            for r in range(B):
                for c in range(B):
                    role = self.qubit_roles[r][c]
                    if role not in ("xQ", "zQ"):        # Only ancillas participate in local gates; data qubits are idle (except for Z gates from TeleGate border protocol, which are handled separately and tracked in z_parity)
                        continue
                    ancilla = self.local_qubits[r][c]
                    if role == "xQ":
                        ancilla.H()                     # Prepare xQ ancillas in |+⟩ for X parity measurement    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:       # local CNOT with neighboring data qubits
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
            round_z, round_x, round_tf = yield from self._teleported_cnot_borders(context)

            # d) Update parity registers
            for (r, c) in round_z:
                z_parity[r][c] ^= 1
            for (r, c) in round_x:
                x_parity[r][c] ^= 1
            for (r, c) in round_tf:
                tele_flip[(r, c)] = tele_flip.get((r, c), 0) ^ 1

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
                        tf = tele_flip.get((r, c), 0)
                        round_syndrome[(r, c)] = raw ^ z_flip ^ tf
                    else:
                        x_flip = 0
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < B and 0 <= nc < B 
                                and self.qubit_roles[nr][nc] == "pQ" 
                                and x_parity[nr][nc] == 1):
                                x_flip ^= 1
                        round_syndrome[(r, c)] = raw ^ x_flip

            all_round_syndromes.append(round_syndrome)
            active = {k: v for k, v in round_syndrome.items() if v == 1}
            print(f"[{self.node_coords}] Round {round_idx + 1} syndrome: "
                  f"{active if active else 'clean'}")

        # ── 4. Spacetime decoding: XOR between two rounds ──────────────────
        s1 = all_round_syndromes[0]
        s2 = all_round_syndromes[1]
        all_pos = set(s1.keys()) | set(s2.keys())
        self.ancilla_measurements = {pos: s1.get(pos, 0) ^ s2.get(pos, 0) for pos in all_pos}
        #self.ancilla_measurements = s1  # For NUM_ROUNDS=1, just use the single round syndrome without XOR

        # Print the final spacetime syndrome after XOR. Active syndromes indicate potential error locations that the coordinator will use for decoding.
        active_final = {k: v for k, v in self.ancilla_measurements.items() if v == 1}
        print(f"[{self.node_coords}] Spacetime syndrome (XOR): "
              f"{active_final if active_final else 'clean'}")

        # ── 5. SVD payload and corrections ────────────────────────────────
        payload_X, payload_Z = self._build_svd_payloads()
        corr_X, corr_Z = yield from self._communicate_with_coordinator(context, payload_X, payload_Z)
        self._apply_corrections(corr_X, gate="X")
        self._apply_corrections(corr_Z, gate="Z")

        if self.applied_X_corrections and self.applied_Z_corrections:
            print(f"[{self.node_coords}] Corrections X: {sorted(self.applied_X_corrections)} Corrections Z: {sorted(self.applied_Z_corrections)}")
        elif self.applied_Z_corrections:
            print(f"[{self.node_coords}] Corrections Z: {sorted(self.applied_Z_corrections)}")
        elif self.applied_X_corrections:
            print(f"[{self.node_coords}] Corrections X: {sorted(self.applied_X_corrections)}")
        else:
            print(f"[{self.node_coords}] Corrections: none")

        # ── 6. Logical-Z parity ───────────────────────────────────────────
        yield from self._send_logical_parity(context, x_parity)
        yield from conn.flush()

    # ------------------------------------------------------------------ #
    #  TeleGate border protocol                                            #
    # ------------------------------------------------------------------ #
    def _teleported_cnot_borders(self, context):
        r_node, c_node = self.node_coords
        N = self.layout_manager.nodes_per_side
        B = self.layout_manager.block_size
        round_z = set()
        round_x = set()
        round_tf = set()

        for c in range(N - 1):
            if c_node == c:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context, neighbor=f"node_{r_node}_{c_node + 1}",
                    is_ancilla_side=True, axis="col", local_fixed=B - 1,
                )
                round_z ^= z_set; round_x ^= x_set; round_tf ^= tf_set
            elif c_node == c + 1:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context, neighbor=f"node_{r_node}_{c_node - 1}",
                    is_ancilla_side=False, axis="col", local_fixed=0,
                )
                round_z ^= z_set; round_x ^= x_set; round_tf ^= tf_set

        for r in range(N - 1):
            if r_node == r:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context, neighbor=f"node_{r_node + 1}_{c_node}",
                    is_ancilla_side=True, axis="row", local_fixed=B - 1,
                )
                round_z ^= z_set; round_x ^= x_set; round_tf ^= tf_set
            elif r_node == r + 1:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context, neighbor=f"node_{r_node - 1}_{c_node}",
                    is_ancilla_side=False, axis="row", local_fixed=0,
                )
                round_z ^= z_set; round_x ^= x_set; round_tf ^= tf_set

        return round_z, round_x, round_tf

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
        x_applied = set()
        tele_flip = set()   # xQ ancillas that received Z() byproduct → X-basis measurement flipped

        for idx in range(B):
            r_loc, c_loc = (local_fixed, idx) if axis == "col" else (idx, local_fixed)
            

            if is_ancilla_side:
                anc_role = self.qubit_roles[r_loc][c_loc]
                """
                Check if there is a remote adjacent data qubit in this direction.
                The remote data qubit is on the opposite border of the neighboring node,
                at the same position along the border (same idx).
                For axis=col: the remote neighbor is in the horizontal direction,
                  so the remote data qubit is adjacent only if anc_role=="xQ"
                  (xQ measures X parity with left/right neighbors).
                For axis=row: the remote neighbor is in the vertical direction,
                  so the remote data qubit is adjacent only if anc_role=="zQ"
                  (zQ measures Z parity with up/down neighbors).
                If the ancilla has no remote neighbors in the correct direction → skip.
                """
                if anc_role == "xQ" and axis == "row":
                    role_signal = "skip"  # xQ has no vertical remote neighbors
                elif anc_role == "zQ" and axis == "col":
                    role_signal = "skip"  # zQ has no horizontal remote neighbors
                elif anc_role in ("xQ", "zQ"):
                    role_signal = anc_role
                else:
                    role_signal = "skip"

                ancilla = self.local_qubits[r_loc][c_loc]

                # eA = epr_sock.create_keep()[0]
                csock.send(role_signal)
                yield from conn.flush()

                if role_signal == "skip":
                    #eA.measure()
                    #yield from conn.flush()
                    yield from csock.recv()
                    continue

                eA = epr_sock.create_keep()[0]

                if role_signal == "xQ":
                    ancilla.cnot(eA)
                    m_A = eA.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_A)))
                    yield from conn.flush()
                    m_B = int((yield from csock.recv()))
                    if m_B == 1:
                        ancilla.Z()
                        tele_flip.add((r_loc, c_loc))

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
                # eB = epr_sock.recv_keep()[0]
                # yield from conn.flush()
                signal = yield from csock.recv()

                if signal == "skip" or self.qubit_roles[r_loc][c_loc] != "pQ":
                    #eB.measure()
                    #yield from conn.flush()
                    csock.send("0")
                    yield from conn.flush()
                    continue
                
                eB = epr_sock.recv_keep()[0]
                data = self.local_qubits[r_loc][c_loc]

                if signal == "xQ":
                    m_A = int((yield from csock.recv()))
                    if m_A == 1:
                        x_applied.add((r_loc, c_loc))
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

        return z_applied, x_applied, tele_flip

    # ------------------------------------------------------------------ #
    #  SVD payload                                                         #
    # ------------------------------------------------------------------ #
    def _build_local_system(self) -> tuple:
        """Build the parity-check system for this node.

        Returns H, s, d_pos where:
          - H: binary parity-check matrix, shape (num_ancilla, num_data)
          - s: syndrome vector, shape (num_ancilla,)
          - d_pos: list of global (row, col) positions of data qubits

        KEY DESIGN DECISION — separate xQ and zQ rows:
        -----------------------------------------------
        xQ ancillas implement X stabilizers and detect Z errors.
        zQ ancillas implement Z stabilizers and detect X errors.
        We therefore build H_Z (zQ rows) and H_X (xQ rows) separately and
        stack only the rows that have at least one non-zero syndrome entry,
        preferring the set that is self-consistent.  If both are active we
        stack them; the coordinator's GF(2) OSD handles the joint system
        correctly because xQ and zQ never share a column after elimination.
        """
        B = self.layout_manager.block_size
        r_node, c_node = self.node_coords
        d_pos = []      # positions of data qubits (pQ)
        zq_pos = []     # positions of zQ ancillas (detect X errors)
        xq_pos = []     # positions of xQ ancillas (detect Z errors)

        for r in range(B):
            for c in range(B):
                gr, gc = r_node * B + r, c_node * B + c # global position of this qubit
                role   = self.qubit_roles[r][c]
                if role == "pQ":
                    d_pos.append((gr, gc))
                elif role == "zQ":
                    zq_pos.append((gr, gc))
                elif role == "xQ":
                    xq_pos.append((gr, gc))

        n = len(d_pos)

        def _make_H_and_s(anc_pos):
            # Build (H_block, s_block) for a list of ancilla positions.
            H_block = np.zeros((len(anc_pos), n), dtype=int)
            for i, (ar, ac) in enumerate(anc_pos):
                for j, (dr, dc) in enumerate(d_pos):
                    if abs(ar - dr) + abs(ac - dc) == 1:
                        H_block[i, j] = 1                   # This ancilla is connected to this data qubit → 1 in H
            s_block = np.array(
                [self.ancilla_measurements.get((p[0] % B, p[1] % B), 0)     # Syndrome value for this ancilla, default to 0 if not measured (e.g. if no ancilla in this position) or if position not found due to some error. The coordinator will handle any inconsistencies.
                 for p in anc_pos],
                dtype=int,
            )
            return H_block, s_block

        H_Z, s_Z = _make_H_and_s(zq_pos)   # Z stabilizers → X error detection
        H_X, s_X = _make_H_and_s(xq_pos)   # X stabilizers → Z error detection
        return H_Z, s_Z, H_X, s_X, d_pos

    def _build_svd_payloads(self, energy_threshold=None):
        if energy_threshold is None:
            energy_threshold = self.ENERGY_THRESHOLD

        H_Z, s_Z, H_X, s_X, data_pos = self._build_local_system()

        def _make_payload(H, s, error_type):
            if H.size == 0 or not np.any(s):
                return {"active": False, "node_id": list(self.node_coords),
                        "error_type": error_type,
                        "data_positions": [list(p) for p in data_pos]}
            m, n = H.shape
            U, sigma, Vt = np.linalg.svd(H.astype(float), full_matrices=False)
            total_energy = np.sum(sigma ** 2)
            k = n
            if total_energy > 1e-10:
                cumulative = np.cumsum(sigma ** 2)
                k = min(int(np.searchsorted(cumulative, energy_threshold * total_energy) + 1), n)
            selected_cols = []
            for i in range(k):
                for j in np.argsort(-np.abs(Vt[i])):
                    if j not in selected_cols:
                        selected_cols.append(j); break
            for j in range(n):
                if len(selected_cols) >= k: break
                if j not in selected_cols: selected_cols.append(j)
            H_reduced = H[:, selected_cols].astype(int)
            V_k = np.zeros((n, k), dtype=float)
            for i, col in enumerate(selected_cols):
                V_k[col, i] = 1.0
            r_node, c_node = self.node_coords
            B = self.layout_manager.block_size
            return {
                "active": True, "node_id": list(self.node_coords),
                "error_type": error_type,
                "H_reduced": H_reduced.tolist(), "V_k": V_k.tolist(),
                "s": s.tolist(), "k": k,
                "data_positions": [list(p) for p in data_pos],
                "global_offset": [r_node * B, c_node * B],
            }

        return _make_payload(H_Z, s_Z, "X"), _make_payload(H_X, s_X, "Z")

    def _communicate_with_coordinator(self, context, payload_X, payload_Z):
        csock = context.csockets[self.coordinator_name]
        csock.send(json.dumps(payload_X))
        csock.send(json.dumps(payload_Z))
        yield from context.connection.flush()
        msg_X = yield from csock.recv()
        msg_Z = yield from csock.recv()
        return json.loads(msg_X), json.loads(msg_Z)

    def _apply_corrections(self, corrections: list, gate: str = "X"):
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
        r_start, c_start = r_node * B, c_node * B
        for r_global, c_global in corrections:
            if (r_start <= r_global < r_start + B
                    and c_start <= c_global < c_start + B):
                r_local = r_global - r_start
                c_local = c_global - c_start
                if self.qubit_roles[r_local][c_local] == "pQ":
                    if gate == "X":
                        self.local_qubits[r_local][c_local].X()
                        self.applied_X_corrections.add((r_local, c_local))
                    else:
                        self.local_qubits[r_local][c_local].Z()
                        self.applied_Z_corrections.add((r_local, c_local))

    # ------------------------------------------------------------------ #
    #  Logical-Z parity                                                    #
    # ------------------------------------------------------------------ #
    def _send_logical_parity(self, context, x_parity=None):
        """
        Compute the logical-Z parity classically.
 
        Z̄ = XOR of all X gates ever applied to data qubits on column 0.
        Three sources contribute:
          1. Noise-induced X errors      → self.injected_errors
          2. TeleGate X byproducts       → x_parity[r][0]
          3. Coordinator X corrections   → self.applied_corrections
 
        If the decoder is perfect: corrections cancel noise exactly,
        so (1) XOR (3) = 0, and the only residual is (2).
        If a logical error survives: (1) XOR (3) = 1 on the logical string,
        so parity = 1 (possibly XOR'd with the TeleGate contribution).
        """
        csock = context.csockets[self.coordinator_name]
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
 
        if c_node != 0:
            csock.send(json.dumps(-1))
        else:
            parity = 0
            for r in range(B):
                if self.qubit_roles[r][0] != "pQ":
                    continue
                if (r, 0) in self.injected_X_errors:
                    parity ^= 1
                if x_parity is not None and x_parity[r][0] == 1:
                    parity ^= 1
                if (r, 0) in self.applied_X_corrections:
                    parity ^= 1
            csock.send(json.dumps(parity))
 
        yield from context.connection.flush()

    # ------------------------------------------------------------------ #
    #  Logical-Z parity                                                    #
    # ------------------------------------------------------------------ #
    def _send_logical_X_parity(self, context, z_parity=None):
        csock = context.csockets[self.coordinator_name]
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size

        if r_node != 0:
            csock.send(json.dumps(-1))
        else:
            parity = 0
            for c in range(B):
                if self.qubit_roles[0][c] != "pQ":
                    continue
                if (0, c) in self.injected_Z_errors:
                    parity ^= 1
                if z_parity is not None and z_parity[0][c] == 1:
                    parity ^= 1
                if (0, c) in self.applied_Z_corrections:
                    parity ^= 1
            csock.send(json.dumps(parity))

        yield from context.connection.flush()

    def _noisy_H(self, qubit, r, c):
        """Hadamard noise """
        qubit.H()
        
        # Se l'utente ha scelto l'errore sull'Hadamard e l'errore scatta:
        if self.error == "hadamard" and random.random() < self.NOISE_PROBABILITY:
            # Modello depolarizzante post-gate
            scelta = random.choice(["X", "Y", "Z"])
            if scelta == "X":
                qubit.X()
                self.injected_X_errors.add((r, c))
            elif scelta == "Y":
                qubit.Y()
                self.injected_X_errors.add((r, c))
                self.injected_Z_errors.add((r, c))
            elif scelta == "Z":
                qubit.Z()
                self.injected_Z_errors.add((r, c))