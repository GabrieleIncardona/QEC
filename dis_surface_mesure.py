import json
import random
import time as t

import numpy as np
from netqasm.sdk.qubit import Qubit
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta


class ClusterNodeProgram(Program):
    ENERGY_THRESHOLD = 0.98  # fraction of total energy to retain in SVD dimensionality reduction (0 < threshold <= 1)
    NUM_ROUNDS = (
        2  # number of rounds of stabilizer measurements (for spacetime decoding)
    )
    BP_ALPHA = 0.75  # Min-Sum scaling factor
    BP_MAX_ITER = 20  # max BP iterations

    def __init__(
        self,
        node_coords: tuple,
        layout_manager,
        error,
        prob,
        coordinator_name: str = "coordinator",
    ):
        self.node_coords = node_coords
        self.layout_manager = layout_manager
        self.coordinator_name = coordinator_name
        self.error = error
        self.NOISE_PROBABILITY = prob  # probability of X error on each data qubit before stabilizer measurements

        r, c = node_coords
        N = layout_manager.nodes_per_side
        self.neighbors = []
        # calculate the neighboring nodes in the 2D grid (up to 4 neighbors: up, down, left, right)
        if r > 0:
            self.neighbors.append(f"node_{r - 1}_{c}")
        if r < N - 1:
            self.neighbors.append(f"node_{r + 1}_{c}")
        if c > 0:
            self.neighbors.append(f"node_{r}_{c - 1}")
        if c < N - 1:
            self.neighbors.append(f"node_{r}_{c + 1}")

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
            name=f"node_{self.node_coords[0]}_{self.node_coords[1]}",
            csockets=self.neighbors + [self.coordinator_name],
            epr_sockets=self.neighbors,
            max_qubits=actual_qubits + border_eprs,
        )

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #
    def run(self, context: ProgramContext):
        self.cnot_count = 0
        conn = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        self.B_rows = len(subgrid_data)
        self.B_cols = (
            len(subgrid_data[0]) if subgrid_data else self.layout_manager.block_size
        )

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
                row_q.append(
                    Qubit(conn)
                )  # Allocate a qubit for every cell in the subgrid (data and ancilla)
                row_r.append(
                    cell["role"]
                )  # Store the role ("pQ", "xQ", "zQ") for each qubit for later reference
            self.local_qubits.append(row_q)
            self.qubit_roles.append(row_r)

        # ── 2. Two rounds of stabilizer measurements ──────────────────────

        all_round_syndromes = []

        for round_idx in range(self.NUM_ROUNDS):
            # Apply noise only before the first round, so that the second round's syndrome reflects the effect of noise.
            if round_idx == 1:
                # Inject noise before the second measurement round
                for error in self.errors:
                    match error:
                        case "identity":
                            for r in range(self.B_rows):
                                for c in range(self.B_cols):
                                    if (
                                        self.qubit_roles[r][c] == "pQ"
                                        and random.random() < self.NOISE_PROBABILITY
                                    ):
                                        self.local_qubits[r][c].X()
                                        self.injected_X_errors.add((r, c))
                            print(
                                f"[{self.node_coords}] Noise: "
                                f"{len(self.injected_X_errors) if self.injected_X_errors else 'none'}"
                            )
                        case "hadamard":
                            for r in range(self.B_rows):
                                for c in range(self.B_cols):
                                    if self.qubit_roles[r][c] == "pQ":
                                        # First noisy H: may inject a depolarizing error
                                        self._noisy_H(self.local_qubits[r][c], r, c)
                                        # Second noisy H restores the net identity (H² = I),
                                        # but each call independently samples noise
                                        self._noisy_H(self.local_qubits[r][c], r, c)

                        case "initialization":
                            print(
                                f"[{self.node_coords}] initialization error: simulating by flipping all ancilla measurements in round 2."
                            )

                        case "readout":
                            print(
                                f"[{self.node_coords}] readout error: simulating by flipping all ancilla measurements in round 2 with probability {self.NOISE_PROBABILITY}."
                            )

                        case "cnot":
                            print(
                                f"[{self.node_coords}] CNOT error: simulating by applying a random X error to the target of each CNOT with probability {self.NOISE_PROBABILITY}."
                            )
                        case "none":
                            print(
                                f"[{self.node_coords}] Ideal simulation: no noise applied."
                            )
                        case _:
                            print(
                                f"[{self.node_coords}] WARNING: Unknown error type '{self.error}'. Skipping noise."
                            )

            round_syndrome = {}

            # ============================================================== #
            #  Z sub-round — Z stabilizers (data = CONTROL, ancilla |0⟩)      #
            #  Measured FIRST and read out BEFORE any X-check CNOT touches    #
            #  the data, so X and Z extractions never cross-talk on a shared  #
            #  data qubit and the syndrome is repeatable round-to-round.      #
            # ============================================================== #
            yield from self._run_stabilizer_subround(
                context, role="zQ", round_idx=round_idx, round_syndrome=round_syndrome
            )

            # ============================================================== #
            #  X sub-round — X stabilizers (ancilla |+⟩ = CONTROL, data)      #
            # ============================================================== #
            yield from self._run_stabilizer_subround(
                context, role="xQ", round_idx=round_idx, round_syndrome=round_syndrome
            )

            all_round_syndromes.append(round_syndrome)
            active = {k: v for k, v in round_syndrome.items() if v == 1}
            print(
                f"[{self.node_coords}] Round {round_idx + 1} syndrome: "
                f"{active if active else 'clean'}"
            )

        # ── 3. Spacetime decoding: XOR between the two rounds ──────────────
        s1 = all_round_syndromes[0]
        s2 = all_round_syndromes[1]
        all_pos = set(s1.keys()) | set(s2.keys())
        self.ancilla_measurements = {
            pos: s1.get(pos, 0) ^ s2.get(pos, 0) for pos in all_pos
        }
        # self.ancilla_measurements = s1  # For NUM_ROUNDS=1, just use the single round syndrome without XOR

        # Print the final spacetime syndrome after XOR. Active syndromes indicate potential error locations that the coordinator will use for decoding.
        active_final = {k: v for k, v in self.ancilla_measurements.items() if v == 1}
        print(
            f"[{self.node_coords}] Spacetime syndrome (XOR): "
            f"{active_final if active_final else 'clean'}"
        )

        # ── 4. Build SVD payloads, exchange with coordinator, apply corrections ──
        t_start = t.time()
        payload_X, payload_Z = self._build_svd_payloads()
        t_local_end = t.time()
        corr_X, corr_Z, t_corr = yield from self._communicate_with_coordinator(
            context, payload_X, payload_Z
        )
        self._apply_corrections(corr_X, gate="X")
        self._apply_corrections(corr_Z, gate="Z")
        yield from conn.flush()  # materialise correction gates before measurement

        if self.applied_X_corrections and self.applied_Z_corrections:
            print(
                f"[{self.node_coords}] Corrections X: {sorted(self.applied_X_corrections)} Corrections Z: {sorted(self.applied_Z_corrections)}"
            )
        elif self.applied_Z_corrections:
            print(
                f"[{self.node_coords}] Corrections Z: {sorted(self.applied_Z_corrections)}"
            )
        elif self.applied_X_corrections:
            print(
                f"[{self.node_coords}] Corrections X: {sorted(self.applied_X_corrections)}"
            )
        else:
            print(f"[{self.node_coords}] Corrections: none")

        # ── 5. Logical-Z parity ───────────────────────────────────────────
        yield from self._send_logical_parity(context)
        yield from conn.flush()

        # ── 6. Send CNOT count to coordinator ────────────────────────────
        context.csockets[self.coordinator_name].send(json.dumps(self.cnot_count))
        yield from conn.flush()

        # ── 7. Report decoding time to coordinator (local SVD build + OSD,
        #      communication latency excluded) ──────────────────────────────
        decoding_time = (t_local_end - t_start) + t_corr
        context.csockets[self.coordinator_name].send(json.dumps(decoding_time))
        yield from conn.flush()

    # ------------------------------------------------------------------ #
    #  One stabilizer sub-round (all 'zQ' OR all 'xQ' checks)              #
    # ------------------------------------------------------------------ #
    def _run_stabilizer_subround(self, context, *, role, round_idx, round_syndrome):
        """Extract one stabilizer type completely: allocate fresh ancillas, apply the
        local CNOTs, run the border TeleGate for this type only, then measure.
        Running the two types in separate ordered sub-rounds (zQ before xQ) makes the
        X- and Z-check circuits commute on every shared data qubit, so the syndrome is
        repeatable from round to round (no schedule-induced cross-talk)."""
        conn = context.connection

        # a) Allocate fresh ancillas for this type
        for r in range(self.B_rows):
            for c in range(self.B_cols):
                if self.qubit_roles[r][c] != role:
                    continue
                ancilla = Qubit(conn)
                # Initialisation error: prepare the ancilla in the orthogonal (wrong) state.
                if (
                    self.error in ("initialization", "all")
                    and random.random() < self.NOISE_PROBABILITY
                    and round_idx == 1
                ):
                    # zQ: |0⟩→|1⟩ ; xQ: |0⟩→|1⟩ which becomes |−⟩ after the H below.
                    ancilla.X()
                    print(
                        f"[{self.node_coords}] Noise: init error on {role} ancilla at ({r}, {c})"
                    )
                if role == "xQ":
                    ancilla.H()  # prepare |+⟩ for the X-parity measurement
                self.local_qubits[r][c] = ancilla

        # b) Local CNOTs with neighbouring data qubits
        for r in range(self.B_rows):
            for c in range(self.B_cols):
                if self.qubit_roles[r][c] != role:
                    continue
                ancilla = self.local_qubits[r][c]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < self.B_rows
                        and 0 <= nc < self.B_cols
                        and self.qubit_roles[nr][nc] == "pQ"
                    ):
                        data = self.local_qubits[nr][nc]
                        if role == "xQ":
                            # X-stabilizer: ancilla |+⟩ is CONTROL, data is TARGET.
                            self._noise_cnot(ancilla, data, (r, c), (nr, nc), round_idx)
                        else:
                            # Z-stabilizer: data is CONTROL, ancilla |0⟩ is TARGET.
                            self._noise_cnot(data, ancilla, (nr, nc), (r, c), round_idx)
        yield from conn.flush()

        # c) Border TeleGate, restricted to this stabilizer type
        yield from self._teleported_cnot_borders(
            context, round_idx=round_idx, only_stab=role
        )

        # d) Measure the ancillas of this type
        for r in range(self.B_rows):
            for c in range(self.B_cols):
                if self.qubit_roles[r][c] != role:
                    continue
                ancilla = self.local_qubits[r][c]
                if role == "xQ":
                    ancilla.H()  # rotate to the X basis before measuring
                m = ancilla.measure()
                yield from conn.flush()
                if (
                    self.error in ("readout", "all")
                    and random.random() < self.NOISE_PROBABILITY
                    and round_idx == 1
                ):
                    m = 1 - m
                    print(f"[{self.node_coords}] Error flip at: ({r}, {c})")
                round_syndrome[(r, c)] = int(m)

    # ------------------------------------------------------------------ #
    #  TeleGate border protocol                                            #
    # ------------------------------------------------------------------ #
    def _teleported_cnot_borders(self, context, round_idx=None, only_stab=None):
        if self.neighbors == []:
            return (
                set(),
                set(),
                set(),
            )  # No neighbors → no TeleGate interactions → no Z or X gates applied, no teleported flips
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
                    context,
                    neighbor=f"node_{r_node}_{c_node + 1}",
                    is_ancilla_side=True,
                    axis="col",
                    local_fixed=self.B_cols - 1,
                    border_len=self.B_rows,
                    round_idx=round_idx,
                    only_stab=only_stab,
                )
                round_z ^= z_set
                round_x ^= x_set
                round_tf ^= tf_set
            elif c_node == c + 1:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context,
                    neighbor=f"node_{r_node}_{c_node - 1}",
                    is_ancilla_side=False,
                    axis="col",
                    local_fixed=0,
                    border_len=self.B_rows,
                    round_idx=round_idx,
                    only_stab=only_stab,
                )
                round_z ^= z_set
                round_x ^= x_set
                round_tf ^= tf_set

        for r in range(N - 1):
            if r_node == r:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context,
                    neighbor=f"node_{r_node + 1}_{c_node}",
                    is_ancilla_side=True,
                    axis="row",
                    local_fixed=self.B_rows - 1,
                    border_len=self.B_cols,
                    round_idx=round_idx,
                    only_stab=only_stab,
                )
                round_z ^= z_set
                round_x ^= x_set
                round_tf ^= tf_set
            elif r_node == r + 1:
                z_set, x_set, tf_set = yield from self._run_border_direction(
                    context,
                    neighbor=f"node_{r_node - 1}_{c_node}",
                    is_ancilla_side=False,
                    axis="row",
                    local_fixed=0,
                    border_len=self.B_cols,
                    round_idx=round_idx,
                    only_stab=only_stab,
                )
                round_z ^= z_set
                round_x ^= x_set
                round_tf ^= tf_set

        return round_z, round_x, round_tf

    def _run_border_direction(
        self,
        context,
        *,
        neighbor: str,
        is_ancilla_side: bool,
        axis: str,
        local_fixed: int,
        border_len: int = None,
        round_idx=None,
        only_stab=None,
    ):
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
        if axis == "col":
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
                stab_type = local_role  # "xQ" or "zQ"
            # LOCAL is data, REMOTE is ancilla → this node handles data side
            elif local_role == "pQ" and remote_role in ("xQ", "zQ"):
                local_is_ancilla = False
                stab_type = remote_role
            else:
                # Neither useful pair (pQ-pQ or anc-anc) — both sides skip.
                # Both nodes independently detect this, so no message exchange needed.
                continue

            # Restrict this pass to one stabilizer type. Both nodes compute the same
            # stab_type for a given border position, so they skip symmetrically (no
            # message exchange for skipped positions) and stay in lock-step.
            if only_stab is not None and stab_type != only_stab:
                continue

            # --- Ancilla side ---
            if local_is_ancilla:
                ancilla = self.local_qubits[r_loc][c_loc]
                csock.send(stab_type)  # tell data side what type this is
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
                        # Byproduct correction on the control (ancilla). After applying it the
                        # teleported CNOT is exact, so the ancilla's X measurement is already
                        # correct — do NOT also flag tele_flip (that would double-correct and
                        # inject the random m_B outcome back into the syndrome).
                        ancilla.Z()

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
                signal = yield from csock.recv()  # "xQ" or "zQ" — never "skip" since skip is handled above without messaging

                eB = epr_sock.recv_keep()[0]
                yield from conn.flush()
                data = self.local_qubits[r_loc][c_loc]

                if signal == "xQ":
                    # CNOT(data→ancilla): data is CONTROL
                    # Cat-Ent B: recv m_A, X^m_A on eB, CNOT(data→eB), meas eB in X, send m_B
                    m_A = int((yield from csock.recv()))
                    if m_A == 1:
                        eB.X()
                    # eB (linked to the remote ancilla) is the CONTROL, local data is the TARGET:
                    # CNOT(eB → data) realises the non-local CNOT(ancilla → data) for the X stabilizer.
                    self._noise_cnot(eB, data, None, (r_loc, c_loc), round_idx)
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
                        # Byproduct correction on the control (data). After applying it the
                        # teleported CNOT(data → ancilla) is exact and the data qubit is left
                        # undisturbed — do NOT record it in z_applied/z_parity (that would
                        # inject the random m_A outcome into the X-stabilizer syndrome).
                        data.Z()

        return z_applied, x_applied, tele_flip

    def _bp_local(self, H: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Scaled Min-Sum BP on GF(2). Returns best hard-decision error estimate."""
        m, n = H.shape
        p_safe = np.clip(self.NOISE_PROBABILITY, 1e-10, 1 - 1e-10)
        ch_llr = np.full(n, np.log((1.0 - p_safe) / p_safe))
        msg_v2c = np.zeros((m, n))
        msg_c2v = np.zeros((m, n))
        e_hat = np.zeros(n, dtype=int)
        for _ in range(self.BP_MAX_ITER):
            for i in range(m):
                neighbors = np.where(H[i] == 1)[0]
                for j in neighbors:
                    others = neighbors[neighbors != j]
                    if len(others) == 0:
                        msg_c2v[i, j] = 0.0
                        continue
                    sign = 1
                    for k in others:
                        sign *= -1 if msg_v2c[i, k] < 0 else 1
                    if s[i] == 1:
                        sign *= -1
                    magnitude = self.BP_ALPHA * float(
                        np.min(np.abs(msg_v2c[i, others]))
                    )
                    msg_c2v[i, j] = sign * magnitude
            for j in range(n):
                neighbors = np.where(H[:, j] == 1)[0]
                total = ch_llr[j] + float(np.sum(msg_c2v[neighbors, j]))
                for i in neighbors:
                    msg_v2c[i, j] = total - msg_c2v[i, j]
            llr_total = ch_llr + msg_c2v.sum(axis=0)
            e_hat = (llr_total < 0).astype(int)
            if np.all((H @ e_hat) % 2 == s):
                return e_hat
        return e_hat

    # ------------------------------------------------------------------ #
    #  SVD payload                                                         #
    # ------------------------------------------------------------------ #
    def _build_local_system(self) -> tuple:
        """Build the parity-check system for this node.
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
        d_pos = []  # positions of data qubits (pQ)
        zq_pos = []  # positions of zQ ancillas (detect X errors)
        xq_pos = []  # positions of xQ ancillas (detect Z errors)

        # use actual subgrid dimensions so border nodes include all their qubits.
        # The global position of qubit (r,c) in this node is derived from subgrid_data
        # to avoid the r_node*B offset being wrong for non-square partitions.
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        actual_rows = len(subgrid_data)
        actual_cols = len(subgrid_data[0]) if subgrid_data else B

        for r in range(actual_rows):
            for c in range(actual_cols):
                gr, gc = subgrid_data[r][c]["global_pos"]  # use stored global position
                role = self.qubit_roles[r][c]
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
                        H_block[i, j] = (
                            1  # This ancilla is connected to this data qubit → 1 in H.
                        )

            # Convert each ancilla global position back to local using the actual subgrid.
            def _global_to_local(gr, gc):
                for ri in range(actual_rows):
                    for ci in range(actual_cols):
                        if subgrid_data[ri][ci]["global_pos"] == (gr, gc):
                            return (ri, ci)
                return None

            s_block = np.array(
                [
                    self.ancilla_measurements.get(_global_to_local(p[0], p[1]), 0)
                    for p in anc_pos
                ],
                dtype=int,
            )

            # remove rows with all zeros (border ancillas whose neighbors
            # are entirely on another node's subgrid). These rows have no
            # data-qubit column to assign an error to, so they only add
            # inconsistent constraints to the OSD solver.
            nonzero_rows = H_block.sum(axis=1) > 0
            H_block = H_block[nonzero_rows]
            s_block = s_block[nonzero_rows]

            return H_block, s_block

        H_Z, s_Z = _make_H_and_s(zq_pos)  # Z stabilizers → X error detection
        H_X, s_X = _make_H_and_s(xq_pos)  # X stabilizers → Z error detection
        return H_Z, s_Z, H_X, s_X, d_pos

    def _build_svd_payloads(self, energy_threshold=None):
        if energy_threshold is None:
            energy_threshold = self.ENERGY_THRESHOLD

        H_Z, s_Z, H_X, s_X, data_pos = self._build_local_system()

        def _make_payload(H, s, error_type):
            if H.size == 0 or not np.any(s):
                return {
                    "active": False,
                    "node_id": list(self.node_coords),
                    "error_type": error_type,
                    "data_positions": [list(p) for p in data_pos],
                    "bp_corrections": [],
                }

            # 1. Run local BP
            e_bp = self._bp_local(H, s)
            s_residual = (s + H @ e_bp) % 2
            bp_corrections = [
                list(data_pos[j]) for j in range(len(e_bp)) if e_bp[j] == 1
            ]

            if not np.any(s_residual):
                # BP converged — no SVD needed
                return {
                    "active": False,
                    "node_id": list(self.node_coords),
                    "error_type": error_type,
                    "data_positions": [list(p) for p in data_pos],
                    "bp_corrections": bp_corrections,
                }

            # 2. BP did not converge — SVD on residual
            m_h, n_h = H.shape
            U, sigma, Vt = np.linalg.svd(H.astype(float), full_matrices=False)
            total_energy = np.sum(sigma**2)
            max_k = min(m_h, n_h)
            if total_energy > 1e-10:
                cumulative = np.cumsum(sigma**2)
                k = min(
                    int(
                        np.searchsorted(cumulative, energy_threshold * total_energy) + 1
                    ),
                    max_k,
                )
            else:
                k = max_k

            energy_retained = (
                np.cumsum(sigma**2)[k - 1] / total_energy
                if total_energy > 1e-10
                else 1.0
            )
            print(
                f"[{self.node_coords}] Local SVD ({error_type}): k={k}/{n_h} "
                f"({energy_retained:.2%} energy retained)"
            )

            U_k = U[:, :k]
            Sig_k = np.diag(sigma[:k])
            H_reduced = U_k @ Sig_k  # (m, k) real-valued
            V_k = Vt[:k, :].T  # (n, k)

            r_node, c_node = self.node_coords
            B = self.layout_manager.block_size
            return {
                "active": True,
                "node_id": list(self.node_coords),
                "error_type": error_type,
                "H_reduced": H_reduced.tolist(),
                "V_k": V_k.tolist(),
                "s": s_residual.tolist(),
                "k": k,
                "data_positions": [list(p) for p in data_pos],
                "global_offset": [r_node * B, c_node * B],
                "bp_corrections": bp_corrections,
                "llr": (U_k @ Sig_k @ Vt[:k, :]).diagonal().tolist()
                if k <= n_h
                else [],
            }

        return _make_payload(H_Z, s_Z, "X"), _make_payload(H_X, s_X, "Z")

    def _communicate_with_coordinator(self, context, payload_X, payload_Z):
        csock = context.csockets[self.coordinator_name]
        csock.send(json.dumps(payload_X))
        csock.send(json.dumps(payload_Z))
        yield from context.connection.flush()
        msg_X = yield from csock.recv()
        tmp_time_x = yield from csock.recv()
        t_x = float(json.loads(tmp_time_x))
        msg_Z = yield from csock.recv()
        tmp_time_z = yield from csock.recv()
        t_z = float(json.loads(tmp_time_z))
        return json.loads(msg_X), json.loads(msg_Z), t_x + t_z

    def _apply_corrections(self, corrections: list, gate: str = "X"):
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        r_start = subgrid_data[0][0]["global_pos"][0]
        c_start = subgrid_data[0][0]["global_pos"][1]
        self.B_rows = len(subgrid_data)
        self.B_cols = len(subgrid_data[0]) if subgrid_data else B
        for r_global, c_global in corrections:
            if (
                r_start <= r_global < r_start + self.B_rows
                and c_start <= c_global < c_start + self.B_cols
            ):
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
        # Logical-Z is a Z-string connecting the two Z-type (rough) boundaries.
        # With this layout the rough boundaries are LEFT/RIGHT (columns 0 and 12,
        # where zQ ancillas sit), so the logical-Z string runs HORIZONTALLY along
        # a row (here global row 0), spanning all columns. The nodes with
        # r_node == 0 each measure their local row-0 data qubits in the Z basis
        # and the coordinator XORs the partial parities.
        #
        # NB: a vertical column-0 string would be WRONG — it shares a single qubit
        # with each boundary X-stabilizer (e.g. xQ(2,1) touches only (2,0)), so it
        # anticommutes with them and the X-stabilizer measurements randomise its
        # parity, giving spurious logical errors even at zero noise.
        csock = context.csockets[self.coordinator_name]
        r_node, c_node = self.node_coords
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        actual_cols = (
            len(subgrid_data[0]) if subgrid_data else self.layout_manager.block_size
        )

        if r_node != 0:
            csock.send(json.dumps(-1))
        else:
            parity = 0
            for c in range(actual_cols):
                if self.qubit_roles[0][c] != "pQ":
                    continue
                outcome = self.local_qubits[0][c].measure()
                yield from context.connection.flush()
                parity ^= int(outcome)

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
        """Hadamard noise"""
        qubit.H()

        if (
            self.error == "hadamard" or self.error == "all"
        ) and random.random() < self.NOISE_PROBABILITY:
            # Depolarizing channel: apply a random Pauli (X, Y, or Z) with equal probability
            choice = random.choice(["X", "Y", "Z"])
            if choice == "X":
                qubit.X()
                self.injected_X_errors.add((r, c))
                print(
                    f"[{self.node_coords}] Noise: "
                    f"{len(self.injected_X_errors) if self.injected_X_errors else 'none'}"
                )
            elif choice == "Y":
                qubit.Y()
                self.injected_X_errors.add((r, c))
                print(
                    f"[{self.node_coords}] Noise: "
                    f"{len(self.injected_X_errors) if self.injected_X_errors else 'none'}"
                )
                self.injected_Z_errors.add((r, c))
                print(
                    f"[{self.node_coords}] Noise: "
                    f"{len(self.injected_Z_errors) if self.injected_Z_errors else 'none'}"
                )
            elif choice == "Z":
                qubit.Z()
                self.injected_Z_errors.add((r, c))
                print(
                    f"[{self.node_coords}] Noise: "
                    f"{len(self.injected_Z_errors) if self.injected_Z_errors else 'none'}"
                )

    def _noise_cnot(self, qubit1, qubit2, coords1, coords2, round_idx=None):
        qubit1.cnot(qubit2)
        self.cnot_count += 1

        if (
            (self.error == "cnot" or self.error == "all")
            and random.random() < self.NOISE_PROBABILITY
            and round_idx == 1
        ):
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

            print(
                f"[{self.node_coords}] CNOT Noise applied. "
                f"X errors: {len(self.injected_X_errors)}, "
                f"Z errors: {len(self.injected_Z_errors)}"
            )
