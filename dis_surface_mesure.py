import json
import random
import numpy as np

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit


class ClusterNodeProgram(Program):
    ENERGY_THRESHOLD  = 1.0       # fraction of total energy to retain in SVD dimensionality reduction (0 < threshold <= 1)
    NUM_ROUNDS        = 2           # number of rounds of stabilizer measurements (for spacetime decoding)

    def __init__(self, node_coords: tuple, layout_manager, error, prob,
                 coordinator_name: str = "coordinator"):
        self.node_coords = node_coords
        self.layout_manager = layout_manager
        self.coordinator_name = coordinator_name
        self.error = error
        self.NOISE_PROBABILITY = prob       # probability of X error on each data qubit before stabilizer measurements

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
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        actual_rows = len(subgrid_data)
        actual_cols = len(subgrid_data[0]) if subgrid_data else B
        # print(f"{self.node_coords} subgrid size: {actual_rows} rows x {actual_cols} cols")
        actual_qubits = actual_rows * actual_cols
        border_eprs = max(actual_rows, actual_cols)  # max EPR qubits on one border
        return ProgramMeta(
            name = f"node_{self.node_coords[0]}_{self.node_coords[1]}",
            csockets = self.neighbors + [self.coordinator_name],
            epr_sockets = self.neighbors,
            max_qubits  = actual_qubits + border_eprs,
        )

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #
    def run(self, context: ProgramContext):
        conn = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        self.B_rows = len(subgrid_data)
        self.B_cols = len(subgrid_data[0]) if subgrid_data else self.layout_manager.block_size

        self.injected_X_errors = set()
        self.injected_Z_errors = set()
        self.applied_X_corrections = set()
        self.applied_Z_corrections = set()
        self.errors = set()
        if self.error == "all":
            self.errors = {"identity", "hadamard","initialization","readout","cnot"}
        else:
            self.errors = {self.error}

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

        z_parity = [[0] * self.B_cols for _ in range(self.B_rows)]  # Classical register tracking Z gates
        x_parity = [[0] * self.B_cols for _ in range(self.B_rows)]  # Classical register tracking X gates
        tele_flip = {}   # (r,c) → parity of Z() byproducts on xQ ancilla: flips X-basis syndrome
        all_round_syndromes = []

        for round_idx in range(self.NUM_ROUNDS):
            # Apply noise only before the first round, so that the second round's syndrome reflects the effect of noise + any Z gates from TeleGate.
            if round_idx == 1:
                # 2. Apply noise
                for error in self.errors:
                    match error:
                        case "identity":
                            for r in range(self.B_rows):
                                for c in range(self.B_cols):
                                    if (self.qubit_roles[r][c] == "pQ"
                                            and random.random() < self.NOISE_PROBABILITY):
                                        self.local_qubits[r][c].X()
                                        self.injected_X_errors.add((r, c))
                            print(f"[{self.node_coords}] Noise: "
                            f"{len(self.injected_X_errors) if self.injected_X_errors else 'none'}")
                        case "hadamard":
                            for r in range(self.B_rows):
                                for c in range(self.B_cols):
                                    if self.qubit_roles[r][c]== "pQ":
                                        # simulation a hadamard error
                                        self._noisy_H(self.local_qubits[r][c], r, c)
                                        # applay 2 time hadamard, in this way, HIH = I, but we applay error
                                        self._noisy_H(self.local_qubits[r][c], r, c)

                        case "initialization":
                            print(f"[{self.node_coords}] initialization error: simulating by flipping all ancilla measurements in round 2.")
                        
                        case "readout":
                            print(f"[{self.node_coords}] readout error: simulating by flipping all ancilla measurements in round 2 with probability {self.NOISE_PROBABILITY}.")

                        case "cnot":
                            print(f"[{self.node_coords}] CNOT error: simulating by applying a random X error to the target of each CNOT with probability {self.NOISE_PROBABILITY}.")
                        case "none":
                            print(f"[{self.node_coords}] Ideal simulation: no noise applied.")
                        case _:
                            print(f"[{self.node_coords}] WARNING: Unknown error type '{self.error}'. Skipping noise.")

            # a) Re-allocate ancillas
            for r in range(self.B_rows):
                for c in range(self.B_cols):
                    role = self.qubit_roles[r][c]
                    if role in ("xQ", "zQ"):
                        ancilla = Qubit(conn)
                        
                        # Add error on ancilla initialization if the user selected it and the error occurs:
                        if (self.error == "initialization" or self.error == "all") and random.random() < self.NOISE_PROBABILITY and round_idx == 1:
                            # For simplicity, we model ancilla initialization error as an X error for zQ ancillas and a Z error for xQ ancillas. This means the ancilla starts in the wrong state (|1⟩ instead of |0⟩ for zQ, or |−⟩ instead of |+⟩ for xQ), which will flip the measurement outcome and simulate a inizialization error.
                            if role == "zQ":
                                ancilla.X()
                                print(f"[{self.node_coords}] Noise: X error on zQ ancilla at ({r}, {c})")
                            else:
                                ancilla.Z()
                                print(f"[{self.node_coords}] Noise: Z error on xQ ancilla at ({r}, {c})")
                        
                        self.local_qubits[r][c] = ancilla

            # b) Local gates
            for r in range(self.B_rows):
                for c in range(self.B_cols):
                    role = self.qubit_roles[r][c]
                    if role not in ("xQ", "zQ"):        # Only ancillas participate in local gates; data qubits are idle (except for Z gates from TeleGate border protocol, which are handled separately and tracked in z_parity)
                        continue
                    ancilla = self.local_qubits[r][c]
                    if role == "xQ":
                        ancilla.H()                     # Prepare xQ ancillas in |+⟩ for X parity measurement    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:       # local CNOT with neighboring data qubits
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.B_rows and 0 <= nc < self.B_cols
                                and self.qubit_roles[nr][nc] == "pQ"):
                            data = self.local_qubits[nr][nc]
                            if role == "xQ":
                                # X stabilizer: data is control, ancilla is target.
                                # CNOT(data→ancilla): ancilla in |+⟩ accumulates X parity of neighbors.
                                # Data qubit is the control → not disturbed.
                                self._noise_cnot(data, ancilla, (nr, nc), (r, c), round_idx)
                            else:
                                # Z stabilizer: ancilla is control, data is target.
                                # CNOT(ancilla->data): measures Z parity correctly.
                                self._noise_cnot(ancilla, data, (r, c), (nr, nc), round_idx)
            yield from conn.flush()

            # c) Original TeleGate protocol — returns Z gates applied this round
            round_z, round_x, round_tf = yield from self._teleported_cnot_borders(context, round_idx=round_idx)

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
            for r in range(self.B_rows):
                for c in range(self.B_cols):
                    role = self.qubit_roles[r][c]
                    if role not in ("xQ", "zQ"):
                        continue
                    ancilla = self.local_qubits[r][c]
                    if role == "xQ":
                        ancilla.H()

                    # 1. Measure the ancilla. If the user selected readout error and it occurs, we flip the measurement result to simulate a readout error.
                    m = ancilla.measure()

                    yield from conn.flush()

                    # 2. Applay readout error if the user selected it and the error occurs:
                    if (self.error == "readout" or self.error == "all") and random.random() < self.NOISE_PROBABILITY and round_idx == 1:
                        m = 1 - m  # flip the measurement result
                        print(f"[{self.node_coords}] Error flip at: ({r}, {c})")
                    raw = int(m)

                    # Z compensation for xQ: Z anticommutes with X
                    # Each neighboring data qubit with odd z_parity contributes
                    # a flip to the xQ measurement (because Z|ψ⟩ in X basis → flip).
                    if role == "xQ":
                        z_flip = 0
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < self.B_rows and 0 <= nc < self.B_cols
                                    and self.qubit_roles[nr][nc] == "pQ"
                                    and z_parity[nr][nc] == 1):
                                z_flip ^= 1
                        tf = tele_flip.get((r, c), 0)
                        round_syndrome[(r, c)] = raw ^ z_flip ^ tf
                    else:
                        x_flip = 0
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < self.B_rows and 0 <= nc < self.B_cols 
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
        yield from conn.flush()   # materialise correction gates before measurement

        if self.applied_X_corrections and self.applied_Z_corrections:
            print(f"[{self.node_coords}] Corrections X: {sorted(self.applied_X_corrections)} Corrections Z: {sorted(self.applied_Z_corrections)}")
        elif self.applied_Z_corrections:
            print(f"[{self.node_coords}] Corrections Z: {sorted(self.applied_Z_corrections)}")
        elif self.applied_X_corrections:
            print(f"[{self.node_coords}] Corrections X: {sorted(self.applied_X_corrections)}")
        else:
            print(f"[{self.node_coords}] Corrections: none")

        # ── 6. Logical-Z parity ───────────────────────────────────────────
        yield from self._send_logical_parity(context, z_parity)
        yield from conn.flush()

    # ------------------------------------------------------------------ #
    #  TeleGate border protocol                                            #
    # ------------------------------------------------------------------ #
    def _teleported_cnot_borders(self, context, round_idx=None):
        if self.neighbors == []:
            return set(), set(), set()  # No neighbors → no TeleGate interactions → no Z or X gates applied, no teleported flips
        r_node, c_node = self.node_coords
        N = self.layout_manager.nodes_per_side
        B = self.layout_manager.block_size
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        round_z = set()
        round_x = set()
        round_tf = set()

        for c in range(N - 1):
            if c_node == c:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context, neighbor=f"node_{r_node}_{c_node + 1}",
                    is_ancilla_side=True, axis="col", local_fixed=self.B_cols - 1,
                    border_len=self.B_rows, round_idx=round_idx,
                )
                round_z ^= z_set; round_x ^= x_set; round_tf ^= tf_set
            elif c_node == c + 1:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context, neighbor=f"node_{r_node}_{c_node - 1}",
                    is_ancilla_side=False, axis="col", local_fixed=0,
                    border_len=self.B_rows, round_idx=round_idx,
                )
                round_z ^= z_set; round_x ^= x_set; round_tf ^= tf_set

        for r in range(N - 1):
            if r_node == r:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context, neighbor=f"node_{r_node + 1}_{c_node}",
                    is_ancilla_side=True, axis="row", local_fixed=self.B_rows - 1,
                    border_len=self.B_cols, round_idx=round_idx,
                )
                round_z ^= z_set; round_x ^= x_set; round_tf ^= tf_set
            elif r_node == r + 1:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context, neighbor=f"node_{r_node - 1}_{c_node}",
                    is_ancilla_side=False, axis="row", local_fixed=0,
                    border_len=self.B_cols, round_idx=round_idx,
                )
                round_z ^= z_set; round_x ^= x_set; round_tf ^= tf_set

        return round_z, round_x, round_tf

    def _run_border_direction(self, context, *, neighbor: str,
                               is_ancilla_side: bool, axis: str,
                               local_fixed: int, border_len: int = None,
                               round_idx=None):
        """
        TeleGate Cat-Ent/Cat-DisEnt protocol — per-qubit role detection.

        For each border position we determine the interaction type:
          - LOCAL qubit is ancilla (xQ/zQ) AND remote qubit is pQ  → local is ANCILLA side
          - LOCAL qubit is pQ AND remote qubit is ancilla (xQ/zQ)  → local is DATA side
          - Any other combination                                   → skip

        Protocol for CNOT(data → ancilla) — X stabilizer (xQ):
          ANCILLA side:
            Cat-Ent:    CNOT(anc→eA), meas eA in Z, send m_A
            Cat-DisEnt: recv m_B, Z^m_B on anc  [tracked in tele_flip]
          DATA side:
            Cat-Ent:    recv m_A, X^m_A on eB, CNOT(data→eB), meas eB in X, send m_B

        Protocol for CNOT(ancilla → data) — Z stabilizer (zQ):
          ANCILLA side:
            Cat-Ent:    recv m_B, X^m_B on eA, CNOT(eA→anc), meas eA in X, send m_A
          DATA side:
            Cat-Ent:    CNOT(data→eB), meas eB in Z, send m_B
            Cat-DisEnt: recv m_A, Z^m_A on data  [tracked in z_applied]
        """
        conn = context.connection
        if axis =="col":
            B = self.B_rows
        else:
            B = self.B_cols
        if border_len is None:
            border_len = B
        csock = context.csockets[neighbor]
        epr_sock = context.epr_sockets[neighbor]
        z_applied = set()
        x_applied = set()
        tele_flip = set()

        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)

        for idx in range(border_len):
            r_loc, c_loc = (idx, local_fixed) if axis == "col" else (local_fixed, idx)
            print(f"[{self.node_coords}] Border idx {idx} at local ({r_loc}, {c_loc})")
            r_glob, c_glob = subgrid_data[r_loc][c_loc]["global_pos"]
            local_role = self.qubit_roles[r_loc][c_loc]

            # Compute the global position of the remote neighbour across the boundary
            if axis == "col":
                dc = 1 if local_fixed != 0 else -1
                nr_glob, nc_glob = r_glob, c_glob + dc
            else:
                dr = 1 if local_fixed != 0 else -1
                nr_glob, nc_glob = r_glob + dr, c_glob

            remote_role = self.layout_manager.get_qubit_role(nr_glob, nc_glob)

            # --- Determine per-qubit interaction type ---
            # LOCAL is ancilla, REMOTE is data → this node handles ancilla side
            if local_role in ("xQ", "zQ") and remote_role == "pQ":
                local_is_ancilla = True
                stab_type = local_role   # "xQ" or "zQ"
            # LOCAL is data, REMOTE is ancilla → this node handles data side
            elif local_role == "pQ" and remote_role in ("xQ", "zQ"):
                local_is_ancilla = False
                stab_type = remote_role
            else:
                # Neither useful pair (pQ-pQ or anc-anc) — both sides skip.
                # Both nodes independently detect this, so no message exchange needed.
                continue

            # --- Ancilla side ---
            if local_is_ancilla:
                ancilla = self.local_qubits[r_loc][c_loc]
                csock.send(stab_type)      # tell data side what type this is
                yield from conn.flush()

                eA = epr_sock.create_keep()[0]
                yield from conn.flush()

                if stab_type == "xQ":
                    # CNOT(data→ancilla): ancilla is TARGET
                    # Cat-Ent A: CNOT(anc→eA), meas eA in Z, send m_A
                    # Cat-DisEnt A: recv m_B, Z^m_B on anc
                    self._noise_cnot(ancilla, eA, (r_loc, c_loc), None, round_idx)
                    m_A = eA.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_A)))
                    yield from conn.flush()
                    m_B = int((yield from csock.recv()))
                    if m_B == 1:
                        ancilla.Z()
                        tele_flip.add((r_loc, c_loc))

                else:  # stab_type == "zQ": CNOT(ancilla→data), ancilla is CONTROL
                    # Cat-Ent A: recv m_B, X^m_B on eA, CNOT(eA→anc), meas eA in X, send m_A
                    m_B = int((yield from csock.recv()))
                    if m_B == 1:
                        eA.X()
                    self._noise_cnot(eA, ancilla, None, (r_loc, c_loc), round_idx)
                    eA.H()
                    m_A = eA.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_A)))
                    yield from conn.flush()

            # --- Data side ---
            else:
                signal = yield from csock.recv()   # "xQ" or "zQ" — never "skip" since skip is handled above without messaging

                eB = epr_sock.recv_keep()[0]
                yield from conn.flush()
                data = self.local_qubits[r_loc][c_loc]

                if signal == "xQ":
                    # CNOT(data→ancilla): data is CONTROL
                    # Cat-Ent B: recv m_A, X^m_A on eB, CNOT(data→eB), meas eB in X, send m_B
                    m_A = int((yield from csock.recv()))
                    if m_A == 1:
                        eB.X()
                    self._noise_cnot(data, eB, (r_loc, c_loc), None, round_idx)
                    eB.H()
                    m_B = eB.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_B)))
                    yield from conn.flush()

                else:  # signal == "zQ": CNOT(ancilla→data), data is TARGET
                    # Cat-Ent B: CNOT(data→eB), meas eB in Z, send m_B
                    # Cat-DisEnt B: recv m_A, Z^m_A on data → tracked in z_applied
                    self._noise_cnot(data, eB, (r_loc, c_loc), None, round_idx)
                    m_B = eB.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_B)))
                    yield from conn.flush()
                    m_A = int((yield from csock.recv()))
                    if m_A == 1:
                        data.Z()
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

        # The global position of qubit (r,c) in this node is derived from subgrid_data
        # to avoid the r_node*B offset being wrong for non-square partitions.
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        actual_rows = len(subgrid_data)
        actual_cols = len(subgrid_data[0]) if subgrid_data else B

        for r in range(actual_rows):
            for c in range(actual_cols):
                gr, gc = subgrid_data[r][c]["global_pos"]  # use stored global position
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
            # Convert each ancilla global position back to local using the actual subgrid.
            def _global_to_local(gr, gc):
                for ri in range(actual_rows):
                    for ci in range(actual_cols):
                        if subgrid_data[ri][ci]["global_pos"] == (gr, gc):
                            return (ri, ci)
                return None

            s_block = np.array(
                [self.ancilla_measurements.get(_global_to_local(p[0], p[1]), 0)
                 for p in anc_pos],
                dtype=int,
            )
            # Remove rows with zero syndrome (inactive stabilizers that don't detect any errors in this round and
            # are entirely on another node's subgrid). These rows have no
            # data-qubit column to assign an error to, so they only add
            # inconsistent constraints to the OSD solver.
            nonzero_rows = H_block.sum(axis=1) > 0
            H_block = H_block[nonzero_rows]
            s_block = s_block[nonzero_rows]

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
            print(f"k:{k} / {n} captures {cumulative[k-1] / total_energy:.2%} energy for {error_type}-type errors at node {self.node_coords}")

            # Select k most informative columns of H via SVD-guided greedy ranking.
            # Vt rows are right-singular vectors; the column of H with highest weight
            # in each leading mode captures the most variance. We pick one unique column
            # per mode, preserving the GF(2) structure of H for the OSD solver.
            selected_cols = []
            for i in range(k):
                for j in np.argsort(-np.abs(Vt[i])):
                    if j not in selected_cols:
                        selected_cols.append(int(j))
                        break
            for j in range(n):
                if len(selected_cols) >= k:
                    break
                if j not in selected_cols:
                    selected_cols.append(j)

            H_reduced = H[:, selected_cols].astype(int)   # (m, k) GF(2) submatrix

            # V_k: sparse (n, k) back-projection matrix.
            # V_k[selected_cols[i], i] = 1 so that the coordinator's
            # round(|V_k @ e_block|) % 2 correctly maps each reduced-space bit
            # back to its original data-qubit position.
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

        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        r_start = subgrid_data[0][0]["global_pos"][0]
        c_start = subgrid_data[0][0]["global_pos"][1]
        self.B_rows = len(subgrid_data)
        self.B_cols = len(subgrid_data[0]) if subgrid_data else B
        for r_global, c_global in corrections:
            if (r_start <= r_global < r_start + self.B_rows
                    and c_start <= c_global < c_start + self.B_cols):
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
        Compute the logical-Z parity via physical Z-basis measurements.

        Z̄ = tensor product of Z on all data qubits along the leftmost column
        of the global grid (local column 0 of nodes with c_node == 0).

        The zQ TeleGate applies Z byproducts to data qubits on the border.
        These are tracked in x_parity (passed as z_parity from the caller).
        We XOR the measured parity with the accumulated Z byproducts on each
        data qubit in the logical column to get the true logical parity.
        """
        csock = context.csockets[self.coordinator_name]
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        actual_rows = len(subgrid_data)

        if c_node != 0:
            csock.send(json.dumps(-1))
        else:
            parity = 0
            for r in range(actual_rows):
                if self.qubit_roles[r][0] != "pQ":
                    continue
                outcome = self.local_qubits[r][0].measure()
                yield from context.connection.flush()
                measured = int(outcome)
                # Compensate Z byproducts from TeleGate that were applied to this
                # data qubit and tracked in z_parity (passed as x_parity parameter).
                # A Z gate on a qubit in |0⟩/|1⟩ basis flips the measurement result.
                #if x_parity is not None and x_parity[r][0] == 1:
                #    measured ^= 1
                parity ^= measured

            print(f"[{self.node_coords}] Logical-Z physical parity = {parity}")
            csock.send(json.dumps(parity))

        yield from context.connection.flush()

    # ------------------------------------------------------------------ #
    #  Logical-X parity                                                    #
    # ------------------------------------------------------------------ #
    def _send_logical_X_parity(self, context, z_parity=None):
        """
        Compute the logical-X parity via physical X-basis measurements.

        X̄ = tensor product of X on all data qubits along the top row of the
        global grid (local row 0 of nodes with r_node == 0).

        Measuring in the X basis = apply H then measure in Z.
        With correct CNOT directions, data qubits are undisturbed by
        stabilizer measurements. After Z corrections they should be back
        in |+⟩ (X-parity 0) if decoding succeeded, or in |−⟩ (parity 1)
        if a logical Z error remains.

        z_parity is accepted for API compatibility but ignored.
        """
        csock = context.csockets[self.coordinator_name]
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size

        if r_node != 0:
            csock.send(json.dumps(-1))
        else:
            parity = 0
            subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
            actual_cols = len(subgrid_data[0]) if subgrid_data else B
            for c in range(actual_cols):
                if self.qubit_roles[0][c] != "pQ":
                    continue
                # Measure in X basis: H then measure in Z
                self.local_qubits[0][c].H()
                outcome = self.local_qubits[0][c].measure()
                yield from context.connection.flush()
                parity ^= int(outcome)

            print(f"[{self.node_coords}] Logical-X physical parity = {parity}")
            csock.send(json.dumps(parity))

        yield from context.connection.flush()

    def _noisy_H(self, qubit, r, c):
        """Hadamard noise """
        qubit.H()

        if (self.error == "hadamard" or self.error == "all") and random.random() < self.NOISE_PROBABILITY:
            # Deporalization error: randomly choose between X, Y, Z error with equal probability
            choice = random.choice(["X", "Y", "Z"])
            if choice == "X":
                qubit.X()
                self.injected_X_errors.add((r, c))
                print(f"[{self.node_coords}] Noise: "
                        f"{len(self.injected_X_errors) if self.injected_X_errors else 'none'}")
            elif choice == "Y":
                qubit.Y()
                self.injected_X_errors.add((r, c))
                print(f"[{self.node_coords}] Noise: "
                        f"{len(self.injected_X_errors) if self.injected_X_errors else 'none'}")
                self.injected_Z_errors.add((r, c))
                print(f"[{self.node_coords}] Noise: "
                        f"{len(self.injected_Z_errors) if self.injected_Z_errors else 'none'}")
            elif choice == "Z":
                qubit.Z()
                self.injected_Z_errors.add((r, c))
                print(f"[{self.node_coords}] Noise: "
                        f"{len(self.injected_Z_errors) if self.injected_Z_errors else 'none'}")


    def _noise_cnot(self, qubit1, qubit2, coords1, coords2, round_idx=None):
        qubit1.cnot(qubit2)

        if (self.error == "cnot" or self.error == "all") and random.random() < self.NOISE_PROBABILITY and round_idx == 1:
            qubits_and_coords = [(qubit1, coords1), (qubit2, coords2)]

            for qb, coords in qubits_and_coords:
                choice = random.choice(["X", "Y", "Z", "I"])

                def toggle_error(error_set, pos):
                    if pos in error_set:
                        error_set.remove(pos)
                    else:
                        error_set.add(pos)

                is_local_data = (
                    coords is not None
                    and 0 <= coords[0] < self.layout_manager.block_size
                    and 0 <= coords[1] < self.layout_manager.block_size
                    and self.qubit_roles[coords[0]][coords[1]] == "pQ"
                )

                if choice in ["X", "Y"]:
                    qb.X()
                    if is_local_data:
                        toggle_error(self.injected_X_errors, coords)

                if choice in ["Z", "Y"]:
                    qb.Z()
                    if is_local_data:
                        toggle_error(self.injected_Z_errors, coords)

            print(f"[{self.node_coords}] CNOT Noise applied. "
                f"X errors: {len(self.injected_X_errors)}, "
                f"Z errors: {len(self.injected_Z_errors)}")