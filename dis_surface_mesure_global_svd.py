"""
Cluster Node Program — No-SVD variant with local BP (Min-Sum) pre-filter
Each node applies Min-Sum Belief Propagation locally before sending data
to the coordinator.

If BP converges (syndrome zeroed), the node sends active=False with
bp_corrections so the coordinator can route them back.
If BP does not converge, the node sends the full H and residual syndrome
(s_residual = s XOR H @ e_bp) so the coordinator's SVD+OSD works on a
partially-cleaned syndrome.

Differences from dis_surface_mesure_global_svd.py (original)
--------------------------------------------------------------
* Added _bp_local()  — Min-Sum BP decoder (more robust on short cycles).
* _build_full_payloads now runs BP before building the payload:
    - BP converges  → active=False, bp_corrections filled
    - BP partial    → active=True,  s = s_residual, H_full unchanged
* Removed debug print statements (DEBUG: starting rounds loop / round_idx).
* Everything else (noise, TeleGate, logical-parity) is identical.
"""

import json
import random
import numpy as np

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit


class ClusterNodeProgram(Program):
    ENERGY_THRESHOLD = 0.98
    NUM_ROUNDS       = 2
    # Min-Sum scaling factor (0.75 corrects the magnitude under-estimation
    # of plain Min-Sum while keeping cycle-robustness)
    BP_ALPHA         = 0.75
    BP_MAX_ITER      = 20

    def __init__(self, node_coords: tuple, layout_manager, error, prob,
                 coordinator_name: str = "coordinator"):
        self.node_coords      = node_coords
        self.layout_manager   = layout_manager
        self.coordinator_name = coordinator_name
        self.error            = error
        self.NOISE_PROBABILITY = prob

        r, c = node_coords
        N = layout_manager.nodes_per_side
        self.neighbors = []
        if r > 0:     self.neighbors.append(f"node_{r-1}_{c}")
        if r < N - 1: self.neighbors.append(f"node_{r+1}_{c}")
        if c > 0:     self.neighbors.append(f"node_{r}_{c-1}")
        if c < N - 1: self.neighbors.append(f"node_{r}_{c+1}")

    @property
    def meta(self) -> ProgramMeta:
        B = self.layout_manager.block_size
        subgrid_data  = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        actual_rows   = len(subgrid_data)
        actual_cols   = len(subgrid_data[0]) if subgrid_data else B
        actual_qubits = actual_rows * actual_cols
        border_eprs   = max(actual_rows, actual_cols) if self.neighbors else 0
        return ProgramMeta(
            name        = f"node_{self.node_coords[0]}_{self.node_coords[1]}",
            csockets    = self.neighbors + [self.coordinator_name],
            epr_sockets = self.neighbors,
            max_qubits  = actual_qubits + border_eprs,
        )

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #
    def run(self, context: ProgramContext):
        conn         = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        self.B_rows  = len(subgrid_data)
        self.B_cols  = len(subgrid_data[0]) if subgrid_data else self.layout_manager.block_size

        self.injected_X_errors    = set()
        self.injected_Z_errors    = set()
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

        z_parity          = [[0] * self.B_cols for _ in range(self.B_rows)]
        x_parity          = [[0] * self.B_cols for _ in range(self.B_rows)]
        tele_flip         = {}
        all_round_syndromes = []

        for round_idx in range(self.NUM_ROUNDS):
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
            round_z, round_x, round_tf = yield from self._teleported_cnot_borders(
                context, round_idx=round_idx
            )

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
            active_synd = {k: v for k, v in round_syndrome.items() if v == 1}
            print(f"[{self.node_coords}] Round {round_idx + 1} syndrome: "
                  f"{active_synd if active_synd else 'clean'}")

        # 4. Spacetime decoding: XOR between the two rounds
        s1      = all_round_syndromes[0]
        s2      = all_round_syndromes[1]
        all_pos = set(s1.keys()) | set(s2.keys())
        self.ancilla_measurements = {
            pos: s1.get(pos, 0) ^ s2.get(pos, 0) for pos in all_pos
        }

        active_final = {k: v for k, v in self.ancilla_measurements.items() if v == 1}
        print(f"[{self.node_coords}] Spacetime syndrome (XOR): "
              f"{active_final if active_final else 'clean'}")

        # 5. Build payloads with local BP pre-filter and exchange with coordinator
        payload_X, payload_Z = self._build_full_payloads()
        corr_X, corr_Z = yield from self._communicate_with_coordinator(
            context, payload_X, payload_Z
        )
        self._apply_corrections(corr_X, gate="X")
        self._apply_corrections(corr_Z, gate="Z")
        yield from conn.flush()

        if self.applied_X_corrections and self.applied_Z_corrections:
            print(f"[{self.node_coords}] Corrections X: {sorted(self.applied_X_corrections)} "
                  f"Corrections Z: {sorted(self.applied_Z_corrections)}")
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
    #  TeleGate border protocol                                           #
    # ------------------------------------------------------------------ #
    def _teleported_cnot_borders(self, context, round_idx=None):
        if self.neighbors == []:
            return set(), set(), set()
        r_node, c_node = self.node_coords
        N = self.layout_manager.nodes_per_side
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
        conn      = context.connection
        B         = self.B_rows if axis == "col" else self.B_cols
        if border_len is None:
            border_len = B
        csock    = context.csockets[neighbor]
        epr_sock = context.epr_sockets[neighbor]
        z_applied = set()
        x_applied = set()
        tele_flip = set()

        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)

        for idx in range(border_len):
            r_loc, c_loc = (idx, local_fixed) if axis == "col" else (local_fixed, idx)
            r_glob, c_glob = subgrid_data[r_loc][c_loc]["global_pos"]
            local_role     = self.qubit_roles[r_loc][c_loc]

            if axis == "col":
                dc = 1 if local_fixed != 0 else -1
                nr_glob, nc_glob = r_glob, c_glob + dc
            else:
                dr = 1 if local_fixed != 0 else -1
                nr_glob, nc_glob = r_glob + dr, c_glob

            remote_role = self.layout_manager.get_qubit_role(nr_glob, nc_glob)

            if local_role in ("xQ", "zQ") and remote_role == "pQ":
                local_is_ancilla = True
                stab_type = local_role
            elif local_role == "pQ" and remote_role in ("xQ", "zQ"):
                local_is_ancilla = False
                stab_type = remote_role
            else:
                continue

            if local_is_ancilla:
                ancilla = self.local_qubits[r_loc][c_loc]
                csock.send(stab_type)
                yield from conn.flush()
                eA = epr_sock.create_keep()[0]
                yield from conn.flush()

                if stab_type == "xQ":
                    self._noise_cnot(ancilla, eA, (r_loc, c_loc), None, round_idx)
                    m_A = eA.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_A)))
                    yield from conn.flush()
                    m_B = int((yield from csock.recv()))
                    if m_B == 1:
                        ancilla.Z()
                        tele_flip.add((r_loc, c_loc))
                else:
                    m_B = int((yield from csock.recv()))
                    if m_B == 1:
                        eA.X()
                    self._noise_cnot(eA, ancilla, None, (r_loc, c_loc), round_idx)
                    eA.H()
                    m_A = eA.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_A)))
                    yield from conn.flush()
            else:
                signal = yield from csock.recv()
                eB     = epr_sock.recv_keep()[0]
                yield from conn.flush()
                data   = self.local_qubits[r_loc][c_loc]

                if signal == "xQ":
                    m_A = int((yield from csock.recv()))
                    if m_A == 1:
                        eB.X()
                    self._noise_cnot(data, eB, (r_loc, c_loc), None, round_idx)
                    eB.H()
                    m_B = eB.measure()
                    yield from conn.flush()
                    csock.send(str(int(m_B)))
                    yield from conn.flush()
                else:
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
    #  Local system builder                                               #
    # ------------------------------------------------------------------ #
    def _build_local_system(self):
        B = self.layout_manager.block_size
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        actual_rows  = len(subgrid_data)
        actual_cols  = len(subgrid_data[0]) if subgrid_data else B

        d_pos, zq_pos, xq_pos = [], [], []
        for r in range(actual_rows):
            for c in range(actual_cols):
                gr, gc = subgrid_data[r][c]["global_pos"]
                role   = self.qubit_roles[r][c]
                if   role == "pQ": d_pos.append((gr, gc))
                elif role == "zQ": zq_pos.append((gr, gc))
                elif role == "xQ": xq_pos.append((gr, gc))

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
            return H_block[nonzero_rows], s_block[nonzero_rows]

        H_Z, s_Z = _make_H_and_s(zq_pos)
        H_X, s_X = _make_H_and_s(xq_pos)
        return H_Z, s_Z, H_X, s_X, d_pos

    # ------------------------------------------------------------------ #
    #  Min-Sum Belief Propagation (local pre-filter)                       #
    # ------------------------------------------------------------------ #
    def _bp_local(self, H: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Scaled Min-Sum BP on GF(2).

        Returns e_hat — best hard-decision error estimate found.
        Converges when (H @ e_hat) % 2 == s; otherwise returns the
        last iterate (may be partial / wrong, but still useful as a
        warm-start residual reduction).

        Parameters
        ----------
        H : (m, n) binary parity-check matrix
        s : (m,)  binary syndrome vector
        """
        m, n = H.shape
        p    = self.NOISE_PROBABILITY
        # Channel LLR: log P(bit=0) / P(bit=1)
        ch_llr = np.full(n, np.log((1.0 - p) / p))

        msg_v2c = np.zeros((m, n))   # variable → check messages
        msg_c2v = np.zeros((m, n))   # check   → variable messages

        e_hat = np.zeros(n, dtype=int)

        for _ in range(self.BP_MAX_ITER):
            # ── Check-node update (Scaled Min-Sum) ──────────────────────
            for i in range(m):
                neighbors = np.where(H[i] == 1)[0]
                for j in neighbors:
                    others = neighbors[neighbors != j]
                    if len(others) == 0:
                        msg_c2v[i, j] = 0.0
                        continue
                    # Sign: product of signs of incoming messages
                    sign = 1
                    for k in others:
                        v = msg_v2c[i, k]
                        sign *= (-1 if v < 0 else 1)
                    # Account for syndrome bit: flip sign if s[i] == 1
                    if s[i] == 1:
                        sign *= -1
                    # Magnitude: scaled minimum of absolute values
                    magnitude = self.BP_ALPHA * float(
                        np.min(np.abs(msg_v2c[i, others]))
                    )
                    msg_c2v[i, j] = sign * magnitude

            # ── Variable-node update ─────────────────────────────────────
            for j in range(n):
                neighbors = np.where(H[:, j] == 1)[0]
                total = ch_llr[j] + float(np.sum(msg_c2v[neighbors, j]))
                for i in neighbors:
                    msg_v2c[i, j] = total - msg_c2v[i, j]

            # ── Hard decision ────────────────────────────────────────────
            llr_total = ch_llr + msg_c2v.sum(axis=0)
            e_hat     = (llr_total < 0).astype(int)

            # ── Convergence check ────────────────────────────────────────
            if np.all((H @ e_hat) % 2 == s):
                return e_hat

        return e_hat

    # ------------------------------------------------------------------ #
    #  Build full payloads with BP pre-filter                              #
    # ------------------------------------------------------------------ #
    def _build_full_payloads(self):
        """
        Build payload dictionaries.

        For each error type (X / Z):
          1. Run Min-Sum BP on the local (H, s).
          2. Compute s_residual = (s + H @ e_bp) % 2.
          3a. If s_residual is all-zero:
                BP solved it completely → active=False, bp_corrections set.
                The coordinator will route these corrections back without SVD/OSD.
          3b. Otherwise:
                Send active=True with H_full and s_residual.
                The coordinator runs SVD+OSD on the cleaner residual syndrome.
        """
        H_Z, s_Z, H_X, s_X, data_pos = self._build_local_system()
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size

        def _make_payload(H, s, error_type):
            # Empty or all-zero syndrome — nothing to do
            if H.size == 0 or not np.any(s):
                return {
                    "active":         False,
                    "node_id":        list(self.node_coords),
                    "error_type":     error_type,
                    "data_positions": [list(p) for p in data_pos],
                    "bp_corrections": [],
                }

            # Run Min-Sum BP
            e_bp       = self._bp_local(H, s)
            s_residual = (s + H @ e_bp) % 2

            if not np.any(s_residual):
                # BP converged — send corrections, no H needed
                bp_corr = [list(data_pos[j]) for j in range(len(e_bp)) if e_bp[j] == 1]
                return {
                    "active":         False,
                    "node_id":        list(self.node_coords),
                    "error_type":     error_type,
                    "data_positions": [list(p) for p in data_pos],
                    "bp_corrections": bp_corr,
                }

            # BP did not converge — send full H with residual syndrome
            m, n = H.shape
            return {
                "active":         True,
                "node_id":        list(self.node_coords),
                "error_type":     error_type,
                "H_full":         H.tolist(),
                "s":              s_residual.tolist(),
                "n":              n,
                "data_positions": [list(p) for p in data_pos],
                "global_offset":  [r_node * B, c_node * B],
                "bp_corrections": [list(data_pos[j]) for j in range(len(e_bp)) if e_bp[j] == 1],
            }

        return _make_payload(H_Z, s_Z, "X"), _make_payload(H_X, s_X, "Z")

    # ------------------------------------------------------------------ #
    #  Communicate with coordinator                                        #
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
    #  Apply corrections                                                 #
    # ------------------------------------------------------------------ #
    def _apply_corrections(self, corrections: list, gate: str = "X"):
        B            = self.layout_manager.block_size
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        r_start      = subgrid_data[0][0]["global_pos"][0]
        c_start      = subgrid_data[0][0]["global_pos"][1]
        self.B_rows  = len(subgrid_data)
        self.B_cols  = len(subgrid_data[0]) if subgrid_data else B
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
    #  Logical-Z parity                                                   #
    # ------------------------------------------------------------------ #
    def _send_logical_parity(self, context, x_parity=None):
        csock        = context.csockets[self.coordinator_name]
        r_node, c_node = self.node_coords
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        actual_rows  = len(subgrid_data)

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
    #  Noise helpers                                                     #
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