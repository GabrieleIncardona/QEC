"""
Cluster Node Program — No-SVD variant
Each node sends the full (un-compressed) parity-check matrix H and
syndrome vector s to the coordinator.  The coordinator is responsible
for the SVD dimensionality reduction before running OSD.

Differences from dis_surface_mesure.py
---------------------------------------
* _build_svd_payloads  →  _build_full_payloads
  Skips every SVD step; sends H_Z / H_X as-is together with s and
  the data-qubit positions.  The "active" flag logic is kept unchanged.
* _communicate_with_coordinator is unchanged (same message protocol).
* Everything else (noise, TeleGate, logical-parity) is identical.
"""

import json
import random
import numpy as np

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit


class ClusterNodeProgram(Program):
    ENERGY_THRESHOLD  = 0.98       # kept for API compatibility; used only by coordinator
    NUM_ROUNDS        = 2

    def __init__(self, node_coords: tuple, layout_manager, error, prob,
                 coordinator_name: str = "coordinator"):
        self.node_coords = node_coords
        self.layout_manager = layout_manager
        self.coordinator_name = coordinator_name
        self.error = error
        self.NOISE_PROBABILITY = prob

        r, c = node_coords
        N = layout_manager.nodes_per_side
        self.neighbors = []
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
        actual_qubits = actual_rows * actual_cols
        border_eprs = max(actual_rows, actual_cols) if self.neighbors else 0
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
            self.errors = {"identity", "hadamard", "initialization", "readout", "cnot"}
        else:
            self.errors = {self.error}

        # 1. Allocate qubits
        self.local_qubits, self.qubit_roles = [], []
        for row in subgrid_data:
            row_q, row_r = [], []
            for cell in row:
                row_q.append(Qubit(conn))
                row_r.append(cell["role"])
            self.local_qubits.append(row_q)
            self.qubit_roles.append(row_r)

        z_parity = [[0] * self.B_cols for _ in range(self.B_rows)]
        x_parity = [[0] * self.B_cols for _ in range(self.B_rows)]
        tele_flip = {}
        all_round_syndromes = []

        print(f"[{self.node_coords}] DEBUG: starting rounds loop")
        for round_idx in range(self.NUM_ROUNDS):
            print(f"[{self.node_coords}] DEBUG: round_idx={round_idx} start")
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
                                    if self.qubit_roles[r][c] == "pQ":
                                        self._noisy_H(self.local_qubits[r][c], r, c)
                                        self._noisy_H(self.local_qubits[r][c], r, c)
                        case "initialization":
                            print(f"[{self.node_coords}] initialization error: simulating by flipping ancilla measurements.")
                        case "readout":
                            print(f"[{self.node_coords}] readout error: simulating by flipping measurements with p={self.NOISE_PROBABILITY}.")
                        case "cnot":
                            print(f"[{self.node_coords}] CNOT error: applying random X error with p={self.NOISE_PROBABILITY}.")
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
                        if (self.error in ("initialization", "all")
                                and random.random() < self.NOISE_PROBABILITY
                                and round_idx == 1):
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
                    if role not in ("xQ", "zQ"):
                        continue
                    ancilla = self.local_qubits[r][c]
                    if role == "xQ":
                        ancilla.H()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.B_rows and 0 <= nc < self.B_cols
                                and self.qubit_roles[nr][nc] == "pQ"):
                            data = self.local_qubits[nr][nc]
                            if role == "xQ":
                                self._noise_cnot(data, ancilla, (nr, nc), (r, c), round_idx)
                            else:
                                self._noise_cnot(ancilla, data, (r, c), (nr, nc), round_idx)
            yield from conn.flush()

            # c) TeleGate border protocol
            round_z, round_x, round_tf = yield from self._teleported_cnot_borders(context, round_idx=round_idx)

            # d) Update parity registers
            for (r, c) in round_z:
                z_parity[r][c] ^= 1
            for (r, c) in round_x:
                x_parity[r][c] ^= 1
            for (r, c) in round_tf:
                tele_flip[(r, c)] = tele_flip.get((r, c), 0) ^ 1

            # e) Measure ancillas
            round_syndrome = {}
            for r in range(self.B_rows):
                for c in range(self.B_cols):
                    role = self.qubit_roles[r][c]
                    if role not in ("xQ", "zQ"):
                        continue
                    ancilla = self.local_qubits[r][c]
                    if role == "xQ":
                        ancilla.H()
                    m = ancilla.measure()
                    yield from conn.flush()
                    if (self.error in ("readout", "all")
                            and random.random() < self.NOISE_PROBABILITY
                            and round_idx == 1):
                        m = 1 - m
                        print(f"[{self.node_coords}] Error flip at: ({r}, {c})")
                    raw = int(m)
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

        # 4. Spacetime decoding: XOR between the two rounds
        s1 = all_round_syndromes[0]
        s2 = all_round_syndromes[1]
        all_pos = set(s1.keys()) | set(s2.keys())
        self.ancilla_measurements = {pos: s1.get(pos, 0) ^ s2.get(pos, 0) for pos in all_pos}

        active_final = {k: v for k, v in self.ancilla_measurements.items() if v == 1}
        print(f"[{self.node_coords}] Spacetime syndrome (XOR): "
              f"{active_final if active_final else 'clean'}")

        # 5. Build FULL payloads (no SVD here) and exchange with coordinator
        payload_X, payload_Z = self._build_full_payloads()
        corr_X, corr_Z = yield from self._communicate_with_coordinator(context, payload_X, payload_Z)
        self._apply_corrections(corr_X, gate="X")
        self._apply_corrections(corr_Z, gate="Z")
        yield from conn.flush()

        if self.applied_X_corrections and self.applied_Z_corrections:
            print(f"[{self.node_coords}] Corrections X: {sorted(self.applied_X_corrections)} Corrections Z: {sorted(self.applied_Z_corrections)}")
        elif self.applied_Z_corrections:
            print(f"[{self.node_coords}] Corrections Z: {sorted(self.applied_Z_corrections)}")
        elif self.applied_X_corrections:
            print(f"[{self.node_coords}] Corrections X: {sorted(self.applied_X_corrections)}")
        else:
            print(f"[{self.node_coords}] Corrections: none")

        # 6. Logical-Z parity
        yield from self._send_logical_parity(context, z_parity)
        yield from conn.flush()

    # ------------------------------------------------------------------ #
    #  TeleGate border protocol  (unchanged from original)                 #
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
            # print(f"[{self.node_coords}] Border idx {idx} at local ({r_loc}, {c_loc})")
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
    def _build_local_system(self):
        B = self.layout_manager.block_size
        r_node, c_node = self.node_coords
        d_pos, zq_pos, xq_pos = [], [], []

        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        actual_rows = len(subgrid_data)
        actual_cols = len(subgrid_data[0]) if subgrid_data else B

        for r in range(actual_rows):
            for c in range(actual_cols):
                gr, gc = subgrid_data[r][c]["global_pos"]
                role   = self.qubit_roles[r][c]
                if role == "pQ":
                    d_pos.append((gr, gc))
                elif role == "zQ":
                    zq_pos.append((gr, gc))
                elif role == "xQ":
                    xq_pos.append((gr, gc))

        n = len(d_pos)

        def _make_H_and_s(anc_pos):
            H_block = np.zeros((len(anc_pos), n), dtype=int)
            for i, (ar, ac) in enumerate(anc_pos):
                for j, (dr, dc) in enumerate(d_pos):
                    if abs(ar - dr) + abs(ac - dc) == 1:
                        H_block[i, j] = 1
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
            nonzero_rows = H_block.sum(axis=1) > 0
            H_block = H_block[nonzero_rows]
            s_block = s_block[nonzero_rows]
            return H_block, s_block

        H_Z, s_Z = _make_H_and_s(zq_pos)
        H_X, s_X = _make_H_and_s(xq_pos)
        return H_Z, s_Z, H_X, s_X, d_pos

    # ------------------------------------------------------------------ #
    #  Build FULL payloads — no SVD, send raw H                            #
    # ------------------------------------------------------------------ #
    def _build_full_payloads(self):
        """
        Build payload dictionaries containing the full (un-compressed)
        parity-check matrix H and syndrome s.

        The coordinator will perform SVD and dimensionality reduction on
        its side before running OSD.

        Payload fields
        --------------
        active          : bool  — False when H is empty or syndrome is all-zero
        node_id         : [r, c]
        error_type      : "X" or "Z"
        H_full          : list[list[int]]  — full (m, n) GF(2) matrix
        s               : list[int]        — syndrome vector length m
        n               : int              — number of data qubits
        data_positions  : list[[r, c]]
        global_offset   : [r_offset, c_offset]
        """
        H_Z, s_Z, H_X, s_X, data_pos = self._build_local_system()
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size

        def _make_payload(H, s, error_type):
            if H.size == 0 or not np.any(s):
                return {
                    "active": False,
                    "node_id": list(self.node_coords),
                    "error_type": error_type,
                    "data_positions": [list(p) for p in data_pos],
                }
            m, n = H.shape
            print(f"[{self.node_coords}] Sending full H ({m}×{n}) "
                  f"for {error_type}-type errors (no local SVD)")
            return {
                "active": True,
                "node_id": list(self.node_coords),
                "error_type": error_type,
                "H_full": H.tolist(),
                "s": s.tolist(),
                "n": n,
                "data_positions": [list(p) for p in data_pos],
                "global_offset": [r_node * B, c_node * B],
            }

        return _make_payload(H_Z, s_Z, "X"), _make_payload(H_X, s_X, "Z")

    # ------------------------------------------------------------------ #
    #  Communicate with coordinator  (unchanged)                           #
    # ------------------------------------------------------------------ #
    def _communicate_with_coordinator(self, context, payload_X, payload_Z):
        csock = context.csockets[self.coordinator_name]
        csock.send(json.dumps(payload_X))
        csock.send(json.dumps(payload_Z))
        yield from context.connection.flush()
        msg_X = yield from csock.recv()
        msg_Z = yield from csock.recv()
        return json.loads(msg_X), json.loads(msg_Z)

    # ------------------------------------------------------------------ #
    #  Apply corrections  (unchanged)                                      #
    # ------------------------------------------------------------------ #
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
    #  Logical-Z parity  (unchanged)                                       #
    # ------------------------------------------------------------------ #
    def _send_logical_parity(self, context, x_parity=None):
        csock = context.csockets[self.coordinator_name]
        r_node, c_node = self.node_coords
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
                parity ^= int(outcome)
            print(f"[{self.node_coords}] Logical-Z physical parity = {parity}")
            csock.send(json.dumps(parity))
        yield from context.connection.flush()

    # ------------------------------------------------------------------ #
    #  Noise helpers  (unchanged)                                          #
    # ------------------------------------------------------------------ #
    def _noisy_H(self, qubit, r, c):
        qubit.H()
        if (self.error in ("hadamard", "all")) and random.random() < self.NOISE_PROBABILITY:
            choice = random.choice(["X", "Y", "Z"])
            if choice == "X":
                qubit.X()
                self.injected_X_errors.add((r, c))
            elif choice == "Y":
                qubit.Y()
                self.injected_X_errors.add((r, c))
                self.injected_Z_errors.add((r, c))
            elif choice == "Z":
                qubit.Z()
                self.injected_Z_errors.add((r, c))

    def _noise_cnot(self, qubit1, qubit2, coords1, coords2, round_idx=None):
        qubit1.cnot(qubit2)
        if (self.error in ("cnot", "all")
                and random.random() < self.NOISE_PROBABILITY
                and round_idx == 1):
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