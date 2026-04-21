"""
Coordinator Program — OSD Decoder
Receives SVD-compressed payloads from all cluster nodes, assembles a
block-diagonal global parity-check matrix, runs OSD over GF(2), then
back-projects corrections to each node via the stored V_k matrices.
"""

import json
import numpy as np
from scipy.linalg import block_diag

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta


class CoordinatorProgram(Program):
    def __init__(self, layout_manager):
        self.layout_manager = layout_manager
        N = layout_manager.nodes_per_side
        self.node_names = [f"node_{r}_{c}" for r in range(N) for c in range(N)]

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="coordinator",
            csockets=self.node_names,
            epr_sockets=[],
            max_qubits=1,       # SquidASM requires at least 1 qubit, but the coordinator doesn't actually use it. We just won't do anything with it.
        )
    
    # Run
    def run(self, context: ProgramContext):
        # Step 1: receive payloads
        payloads_X, payloads_Z = [], []
        for name in self.node_names:
            msg_X = yield from context.csockets[name].recv()
            msg_Z = yield from context.csockets[name].recv()
            payloads_X.append(json.loads(msg_X))
            payloads_Z.append(json.loads(msg_Z))

        # Step 2: assemble block-diagonal system and run OSD separately for X and Z errors
        for payloads, error_type in [(payloads_X, "X"), (payloads_Z, "Z")]:
            H_global, s_global, registry = self._assemble_global_system(payloads)
            if H_global is not None:
                e_global = self._osd_gf2(H_global, s_global)
                corrections = self._project_corrections(e_global, registry) if np.any(e_global) else {}
            else:
                corrections = {}
            yield from self._send_corrections(context, payloads, corrections)

        # Step 6: aggregate logical-Z parities
        global_parity = 0
        for name in self.node_names:
            msg = yield from context.csockets[name].recv()
            val = json.loads(msg)
            if val != -1:
                global_parity ^= val

        status = "OK — no logical error" if global_parity == 0 else "FAIL — logical error survived!"
        print(f"\n=== Logical Z (global) = {global_parity} → {status} ===\n")

        yield from context.connection.flush()
        return global_parity

    # Step 2 — Assemble block-diagonal global system
    def _assemble_global_system(self, payloads: list) -> tuple:
        active = [p for p in payloads if p.get("active", False)]
        if not active:
            return None, None, []

        H_blocks, s_list, registry = [], [], []
        col_offset = 0                                      # Column offset for each block in the global H, used for back-projection of corrections

        for p in active:
            H_red = np.array(p["H_reduced"])   # (m_i, k_i)
            V_k = np.array(p["V_k"])          # (n_i, k_i)
            s_i = np.array(p["s"], dtype=int)
            k_i = int(p["k"])

            H_blocks.append(H_red)
            s_list.extend(s_i.tolist())

            registry.append({                                # Store info needed for back-projection of corrections to each node
                "node_id":        tuple(p["node_id"]),
                "V_k":            V_k,
                "data_positions": [tuple(pos) for pos in p["data_positions"]],
                "col_start":      col_offset,
                "col_end":        col_offset + k_i,
            })
            col_offset += k_i

        H_global = block_diag(*H_blocks)        # Block-diagonal global parity-check matrix
        s_global = np.array(s_list, dtype=int)  # Global syndrome vector
        return H_global, s_global, registry     # Return registry for back-projection of corrections

    # Step 3 — OSD over GF(2)
    def _osd_gf2(self, H_global: np.ndarray, s_global: np.ndarray, osd_order: int = 2) -> np.ndarray:
        m, K_tot = H_global.shape

        H = H_global.astype(int) % 2
        s = s_global.astype(int) % 2

        # --- Gaussian elimination over GF(2) ---
        H_sys, s_sys = H.copy(), s.copy()
        pivot_cols = []
        pivot_rows = []
        row = 0

        for col in range(K_tot):
            if row >= m:
                break
            # Find a row with a 1 in this column
            found = next((r for r in range(row, m) if H_sys[r, col] == 1), -1)
            if found == -1:
                continue
            # Swap
            H_sys[[row, found]] = H_sys[[found, row]]
            s_sys[[row, found]] = s_sys[[found, row]]
            # Eliminate column in all other rows
            for r2 in range(m):
                if r2 != row and H_sys[r2, col] == 1:
                    H_sys[r2] = (H_sys[r2] + H_sys[row]) % 2
                    s_sys[r2] = (s_sys[r2] + s_sys[row]) % 2
            pivot_cols.append(col)
            pivot_rows.append(row)
            row += 1

        non_pivot = [c for c in range(K_tot) if c not in pivot_cols]            # Columns corresponding to free variables (non-pivots)
        test_cols = non_pivot[:osd_order]                                       # Columns to test for flipping in OSD (limited by osd_order)
        n_test = len(test_cols)

        best_weight = K_tot + 1
        best_e = np.zeros(K_tot, dtype=int)

        # --- Enumerate test patterns on free (non-pivot) columns ---
        for pattern in range(2 ** n_test):
            e = np.zeros(K_tot, dtype=int)
            for b, tc in enumerate(test_cols):
                e[tc] = (pattern >> b) & 1              # Set test pattern bits in e for the selected free columns

            # Solve pivot bits mod 2
            for pc, pr in zip(pivot_cols, pivot_rows):
                rhs = s_sys[pr]
                for fc in non_pivot:
                    rhs = (rhs + H_sys[pr, fc] * e[fc]) % 2         # Update rhs by adding contributions from free variables
                e[pc] = rhs

            weight = int(np.sum(e))
            if weight < best_weight:
                best_weight = weight
                best_e = e.copy()

        return best_e

    # Step 4 — Back-project global error to per-node corrections
    def _project_corrections(self, e_global: np.ndarray, registry: list) -> dict:
        corrections = {}
        for reg in registry:
            e_block = e_global[reg["col_start"]:reg["col_end"]]   # shape (k,)
            V_k = reg["V_k"]                                      # shape (n, k)

            # Back-project from reduced space to full data-qubit space and binarise.
            # V_k @ e_block maps the k-dimensional correction back to n data qubits.
            # We round the absolute value and take mod 2 to recover a GF(2) vector.
            e_local = (np.round(np.abs(np.array(V_k, dtype=float) @ e_block.astype(float))) % 2).astype(int)

            corrections[reg["node_id"]] = [
                list(reg["data_positions"][j])
                for j in range(len(e_local)) if e_local[j] == 1
            ]
        return corrections
    
    # Step 5 — Send corrections to nodes
    def _send_corrections(self, context: ProgramContext, payloads: list, corrections_per_node: dict):
        for name in self.node_names:
            parts = name.replace("node_", "").split("_")
            node_id = (int(parts[0]), int(parts[1]))
            corr = corrections_per_node.get(node_id, [])
            context.csockets[name].send(json.dumps(corr))
        yield from context.connection.flush()