import json
import random
import numpy as np

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit


class ClusterNodeProgram(Program):
    NOISE_PROBABILITY = 0.1        # probability of X error on each data qubit before stabilizer measurements
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
            max_qubits  = B * B + B,    # Max qubits needed: all phisical qubits (B^2) + all epr qubit on one border (B) for TeleGate protocol
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
        all_round_syndromes = []

        for round_idx in range(self.NUM_ROUNDS):
            # Apply noise only before the first round, so that the second round's syndrome reflects the effect of noise + any Z gates from TeleGate.
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
                        # zQ measures Z parity, which is unaffected by Z gates on data qubits (commutes), so no compensation needed.
                        round_syndrome[(r, c)] = raw

            all_round_syndromes.append(round_syndrome)
            active = {k: v for k, v in round_syndrome.items() if v == 1}
            print(f"[{self.node_coords}] Round {round_idx + 1} syndrome: "
                  f"{active if active else 'clean'}")

        # ── 4. Spacetime decoding: XOR between two rounds ──────────────────
        s1 = all_round_syndromes[0]
        s2 = all_round_syndromes[1]
        self.ancilla_measurements = {pos: s1[pos] ^ s2[pos] for pos in s1}
        #self.ancilla_measurements = s1  # For NUM_ROUNDS=1, just use the single round syndrome without XOR

        # Print the final spacetime syndrome after XOR. Active syndromes indicate potential error locations that the coordinator will use for decoding.
        active_final = {k: v for k, v in self.ancilla_measurements.items() if v == 1}
        print(f"[{self.node_coords}] Spacetime syndrome (XOR): "
              f"{active_final if active_final else 'clean'}")

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
        Execute TeleGate protocol for all borders of this node, in a globally
        canonical order to prevent deadlock in NxN grids.

        DEADLOCK ROOT CAUSE
        -------------------
        The original order (right, left, down, up) makes different nodes process
        their borders in incompatible sequences. In a 3x3 grid this creates circular
        waits: A waits for B which waits for C which waits for A.

        FIX: canonical global border ordering
        --------------------------------------
        All nodes process borders in the same global order:
          Phase 1 — horizontal links (col direction), row by row, left to right:
            (r,0)-(r,1), (r,1)-(r,2), ..., for r = 0..N-1
          Phase 2 — vertical links (row direction), column by column, top to bottom:
            (0,c)-(1,c), (1,c)-(2,c), ..., for c = 0..N-1

        For each link, the node with the SMALLER coordinate is always ancilla_side=True
        (calls create_keep). The node with the larger coordinate is always False (recv_keep).
        Since every node processes links in the same global order, both sides of each
        EPR link are always ready at the same step — no circular waits possible.

        Returns round_z: set of (r_loc, c_loc) of data qubits that received Z() this round.
        """
        r_node, c_node = self.node_coords
        N = self.layout_manager.nodes_per_side
        B = self.layout_manager.block_size
        round_z = set()

        # ── Phase 1: horizontal links (between left/right neighbors) ──────────
        # Process in order: column c=0..N-2, for each column all rows r=0..N-1
        for c in range(N - 1):
            # Link between (r_node, c) and (r_node, c+1)
            if c_node == c:          # this node is the LEFT side → ancilla_side=True
                z_set = yield from self._run_border_direction(
                    context,
                    neighbor = f"node_{r_node}_{c_node + 1}",
                    is_ancilla_side = True,
                    axis  = "col",
                    local_fixed = B - 1,   # rightmost column of this node
                )
                round_z ^= z_set
            elif c_node == c + 1:    # this node is the RIGHT side → ancilla_side=False
                z_set = yield from self._run_border_direction(
                    context,
                    neighbor = f"node_{r_node}_{c_node - 1}",
                    is_ancilla_side = False,
                    axis = "col",
                    local_fixed = 0,        # leftmost column of this node
                )
                round_z ^= z_set
            # else: this node is not part of this link, skip

        # ── Phase 2: vertical links (between top/bottom neighbors) ────────────
        # Process in order: row r=0..N-2, for each row all columns c=0..N-1
        for r in range(N - 1):
            # Link between (r, c_node) and (r+1, c_node)
            if r_node == r:          # this node is the TOP side → ancilla_side=True
                z_set = yield from self._run_border_direction(
                    context,
                    neighbor        = f"node_{r_node + 1}_{c_node}",
                    is_ancilla_side = True,
                    axis            = "row",
                    local_fixed     = B - 1,   # bottom row of this node
                )
                round_z ^= z_set
            elif r_node == r + 1:    # this node is the BOTTOM side → ancilla_side=False
                z_set = yield from self._run_border_direction(
                    context,
                    neighbor        = f"node_{r_node - 1}_{c_node}",
                    is_ancilla_side = False,
                    axis            = "row",
                    local_fixed     = 0,        # top row of this node
                )
                round_z ^= z_set
            # else: not part of this link, skip

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

                eA = epr_sock.create_keep()[0]
                csock.send(role_signal)
                yield from conn.flush()

                if role_signal == "skip":
                    eA.measure()
                    yield from conn.flush()
                    yield from csock.recv()
                    continue

                #eA = epr_sock.create_keep()[0]

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
                
                #eB = epr_sock.recv_keep()[0]
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

        # Only include rows whose syndrome is actually non-trivial, to avoid
        # inflating the system with inconsistent zero-syndrome / dependent rows.
        active_Z = np.any(s_Z)
        active_X = np.any(s_X)

        if active_Z and active_X:
            H = np.vstack([H_Z, H_X])           # Stack both Z and X stabilizer rows if both are active. The coordinator's OSD will handle the joint system correctly because xQ and zQ never share a column after elimination.
            s = np.concatenate([s_Z, s_X])      # Stack corresponding syndromes
        elif active_Z:
            H = H_Z
            s = s_Z
        elif active_X:
            H = H_X
            s = s_X
        else:
            # No active syndrome: return full H (both blocks) with zero syndrome
            # so the coordinator gets the correct structure even if inactive.
            H = np.vstack([H_Z, H_X]) if (len(zq_pos) + len(xq_pos)) > 0 else np.zeros((0, n), dtype=int)       # If there are no ancillas at all, return an empty H with shape (0, n) to avoid confusion with a single row of zeros.
            s = np.zeros(H.shape[0], dtype=int)     # All-zero syndrome

        return H, s, d_pos

    def _build_svd_payload(self, energy_threshold: float = None) -> dict:
        """Build the SVD-compressed payload to send to the coordinator.

        SVD-BASED COLUMN SELECTION FOR GF(2)-COMPATIBLE COMPRESSION
        -------------------------------------------------------------
        We use SVD to identify the k most 'energy-carrying' directions in the
        data-qubit space, then select the k representative qubits (columns of H)
        that best span those directions.


        Algorithm:
          1. SVD of H (real) -> right singular vectors Vt  (shape: min(m,n) x n)
          2. Keep top-k singular vectors (energy threshold).
          3. For each of the k dimensions, pick the data qubit j = argmax |Vt[i, j]|.
             These are the qubits most aligned with the principal directions of H.
          4. H_reduced = H[:, selected_cols]  (binary, m x k)
          5. V_k = binary selection matrix (n x k), with V_k[selected_cols[i], i] = 1.

        For large blocks (e.g. 4x4 nodes) this achieves genuine compression:
        k ~ 3-4 instead of n=8, halving the payload sent to the coordinator.
        For small blocks (2x2) k=n=2 so no compression.
        """
        if energy_threshold is None:
            energy_threshold = self.ENERGY_THRESHOLD

        H, s, data_pos = self._build_local_system()

        if H.size == 0 or not np.any(s):
            return {
                "active":         False,
                "node_id":        list(self.node_coords),
                "data_positions": [list(p) for p in data_pos],
            }

        r_node, c_node = self.node_coords
        B  = self.layout_manager.block_size
        m, n = H.shape

        # --- SVD to find principal directions in qubit space ---
        U, sigma, Vt = np.linalg.svd(H.astype(float), full_matrices=False)

        # Determine k via energy threshold
        total_energy = np.sum(sigma ** 2)
        k = n
        if total_energy > 1e-10:                # Avoid division by zero if H is all zeros (shouldn't happen if active, but just in case)
            cumulative = np.cumsum(sigma ** 2)  # Cumulative energy captured by top singular values
            k = int(np.searchsorted(cumulative, # Find the smallest k such that cumulative energy >= threshold * total_energy
                                    energy_threshold * total_energy) + 1)
            k = min(k, n)

        # --- Select k representative columns (data qubits) ---
        # For each of the top-k singular vectors, pick the qubit with highest loading.
        # This gives a set of k qubits that best span the principal error directions.
        selected_cols = []
        for i in range(k):
            # Candidates ordered by |loading| on singular vector i
            order = np.argsort(-np.abs(Vt[i]))
            for j in order:
                if j not in selected_cols:
                    selected_cols.append(j)
                    break

        # Fill up to k if any duplicates were skipped (shouldn't happen, but defensive)
        for j in range(n):
            if len(selected_cols) >= k:
                break
            if j not in selected_cols:
                selected_cols.append(j)

        # --- Build binary reduced system ---
        H_reduced = H[:, selected_cols].astype(int)    # (m, k) — exact binary submatrix

        # V_k: exact {0,1} selection matrix (n x k).
        # Back-projection: e_full[selected_cols[i]] = e_reduced[i], others = 0.
        # Using float dtype for compatibility with coordinator's np.array(V_k, dtype=float).
        V_k = np.zeros((n, k), dtype=float)
        for i, col in enumerate(selected_cols):
            V_k[col, i] = 1.0

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

    def _communicate_with_coordinator(self, context, payload: dict):
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
            if (r_start <= r_global < r_start + B
                    and c_start <= c_global < c_start + B):
                r_local = r_global - r_start
                c_local = c_global - c_start
                if self.qubit_roles[r_local][c_local] == "pQ":
                    self.local_qubits[r_local][c_local].X()
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