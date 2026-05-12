"""
Coordinator Program — SVD-on-Coordinator variant with BP corrections support
Receives payloads from all cluster nodes. Each payload is either:
  - active=False, bp_corrections=[...]  → BP on the node solved it completely;
                                          route the corrections back directly.
  - active=True,  H_full, s (residual)  → BP did not converge; apply global
                                          SVD + OSD on the residual syndrome.

Differences from coordinator_global_svd.py (original)
------------------------------------------------------
* _send_corrections now merges OSD corrections with bp_corrections from
  inactive payloads, so BP-solved nodes receive their corrections too.
* _assemble_global_system and _osd_gf2 and _project_corrections are unchanged.
* Logical-parity aggregation is unchanged.
"""

import json
import numpy as np
from scipy.linalg import block_diag

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta


class CoordinatorProgram(Program):
    ENERGY_THRESHOLD = 0.98

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
            max_qubits=1,
        )

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #
    def run(self, context: ProgramContext):
        # Step 1: receive payloads (full-H or bp_corrections) from all nodes
        payloads_X, payloads_Z = [], []
        for name in self.node_names:
            msg_X = yield from context.csockets[name].recv()
            msg_Z = yield from context.csockets[name].recv()
            payloads_X.append(json.loads(msg_X))
            payloads_Z.append(json.loads(msg_Z))

        # Steps 2-5: SVD + OSD + corrections (X and Z separately)
        for payloads, error_type in [(payloads_X, "X"), (payloads_Z, "Z")]:
            H_global, s_global, registry = self._assemble_global_system(payloads)
            if H_global is not None:
                e_global    = self._osd_gf2(H_global, s_global)
                corrections = (
                    self._project_corrections(e_global, registry)
                    if np.any(e_global) else {}
                )
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

    # ------------------------------------------------------------------ #
    #  Step 2 — Assemble block-diagonal system then apply global SVD       #
    # ------------------------------------------------------------------ #
    def _assemble_global_system(self, payloads: list) -> tuple:
        """
        Only processes active payloads (BP did not converge on the node).
        Inactive payloads carry bp_corrections handled in _send_corrections.
        """
        active = [p for p in payloads if p.get("active", False)]
        if not active:
            return None, None, []

        H_blocks        = []
        s_list          = []
        node_col_ranges = []
        col_offset      = 0

        for p in active:
            H_full = np.array(p["H_full"], dtype=int)
            s_i    = np.array(p["s"],      dtype=int)
            n_i    = int(p["n"])

            H_blocks.append(H_full)
            s_list.extend(s_i.tolist())
            node_col_ranges.append({
                "node_id":        tuple(p["node_id"]),
                "data_positions": [tuple(pos) for pos in p["data_positions"]],
                "col_start":      col_offset,
                "col_end":        col_offset + n_i,
            })
            col_offset += n_i

        H_global_full = block_diag(*H_blocks)
        s_global      = np.array(s_list, dtype=int)
        N_total       = H_global_full.shape[1]

        # Single SVD on the full global matrix
        U, sigma, Vt  = np.linalg.svd(H_global_full.astype(float), full_matrices=False)
        total_energy  = np.sum(sigma ** 2)

        if total_energy > 1e-10:
            cumulative = np.cumsum(sigma ** 2)
            k_global   = min(
                int(np.searchsorted(cumulative,
                                    self.ENERGY_THRESHOLD * total_energy) + 1),
                N_total,
            )
        else:
            k_global = N_total

        print(f"[coordinator] Global SVD: k={k_global}/{N_total} "
              f"({cumulative[k_global-1] / total_energy:.2%} energy retained)")

        # SVD-guided greedy column selection
        selected_cols = []
        for i in range(k_global):
            for j in np.argsort(-np.abs(Vt[i])):
                if j not in selected_cols:
                    selected_cols.append(int(j))
                    break
        for j in range(N_total):
            if len(selected_cols) >= k_global:
                break
            if j not in selected_cols:
                selected_cols.append(j)

        H_global_reduced = H_global_full[:, selected_cols].astype(int)

        V_global = np.zeros((N_total, k_global), dtype=float)
        for i, col in enumerate(selected_cols):
            V_global[col, i] = 1.0

        registry = []
        for ncr in node_col_ranges:
            rows = slice(ncr["col_start"], ncr["col_end"])
            V_k  = V_global[rows, :]
            registry.append({
                "node_id":        ncr["node_id"],
                "V_k":            V_k,
                "data_positions": ncr["data_positions"],
                "col_start":      0,
                "col_end":        k_global,
            })

        return H_global_reduced, s_global, registry

    # ------------------------------------------------------------------ #
    #  Step 3 — OSD over GF(2)                                           #
    # ------------------------------------------------------------------ #
    def _osd_gf2(self, H_global: np.ndarray, s_global: np.ndarray,
                 osd_order: int = 2) -> np.ndarray:
        m, K_tot = H_global.shape
        H = H_global.astype(int) % 2
        s = s_global.astype(int) % 2

        H_sys, s_sys   = H.copy(), s.copy()
        pivot_cols, pivot_rows = [], []
        row = 0

        for col in range(K_tot):
            if row >= m:
                break
            found = next((r for r in range(row, m) if H_sys[r, col] == 1), -1)
            if found == -1:
                continue
            H_sys[[row, found]] = H_sys[[found, row]]
            s_sys[[row, found]] = s_sys[[found, row]]
            for r2 in range(m):
                if r2 != row and H_sys[r2, col] == 1:
                    H_sys[r2] = (H_sys[r2] + H_sys[row]) % 2
                    s_sys[r2] = (s_sys[r2] + s_sys[row]) % 2
            pivot_cols.append(col)
            pivot_rows.append(row)
            row += 1

        non_pivot  = [c for c in range(K_tot) if c not in pivot_cols]
        test_cols  = non_pivot[:osd_order]
        n_test     = len(test_cols)

        best_weight = K_tot + 1
        best_e      = np.zeros(K_tot, dtype=int)

        for pattern in range(2 ** n_test):
            e = np.zeros(K_tot, dtype=int)
            for b, tc in enumerate(test_cols):
                e[tc] = (pattern >> b) & 1
            for pc, pr in zip(pivot_cols, pivot_rows):
                rhs = s_sys[pr]
                for fc in non_pivot:
                    rhs = (rhs + H_sys[pr, fc] * e[fc]) % 2
                e[pc] = rhs
            weight = int(np.sum(e))
            if weight < best_weight:
                best_weight = weight
                best_e      = e.copy()

        return best_e

    # ------------------------------------------------------------------ #
    #  Step 4 — Back-project corrections                                 #
    # ------------------------------------------------------------------ #
    def _project_corrections(self, e_global: np.ndarray, registry: list) -> dict:
        corrections = {}
        for reg in registry:
            e_block = e_global[reg["col_start"]:reg["col_end"]]
            V_k     = reg["V_k"]
            e_local = (
                np.round(np.abs(np.array(V_k, dtype=float) @ e_block.astype(float))) % 2
            ).astype(int)
            corrections[reg["node_id"]] = [
                list(reg["data_positions"][j])
                for j in range(len(e_local)) if e_local[j] == 1
            ]
        return corrections

    # ------------------------------------------------------------------ #
    #  Step 5 — Send corrections (merges OSD + BP corrections)             #
    # ------------------------------------------------------------------ #
    def _send_corrections(self, context: ProgramContext, payloads: list,
                          corrections_per_node: dict):
        """
        For each node, send the union of:
          - OSD corrections (from active payloads processed by SVD+OSD)
          - BP corrections  (from inactive payloads where BP converged,
                             or partial BP corrections stored in active payloads)

        This ensures that nodes whose syndromes were fully resolved by BP
        still receive the right correction list.
        """
        # Build a map: node_id → bp_corrections from ALL payloads
        bp_map: dict[tuple, list] = {}
        for p in payloads:
            node_id   = tuple(p["node_id"])
            bp_corrs  = p.get("bp_corrections", [])
            if bp_corrs:
                existing = bp_map.get(node_id, [])
                bp_map[node_id] = existing + bp_corrs

        for name in self.node_names:
            parts   = name.replace("node_", "").split("_")
            node_id = (int(parts[0]), int(parts[1]))

            # Start with OSD corrections (may be empty)
            osd_corr = corrections_per_node.get(node_id, [])
            bp_corr  = bp_map.get(node_id, [])

            # Merge: XOR the two correction sets (corrections cancel if duplicated)
            # Convert to sets of tuples for XOR merge
            osd_set = set(tuple(c) for c in osd_corr)
            bp_set  = set(tuple(c) for c in bp_corr)
            merged  = list(osd_set.symmetric_difference(bp_set))

            context.csockets[name].send(json.dumps(merged))

        yield from context.connection.flush()