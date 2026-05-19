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
            active_nodes = [(p["node_id"], p.get("s", []), "H_reduced" in p)
                           for p in payloads if p.get("active", False)]
            inactive_nodes = [(p["node_id"], p.get("bp_corrections", []))
                             for p in payloads if not p.get("active", False)]
            print(f"\n[coordinator] {error_type}-errors: "
                  f"active={[n for n,s,_ in active_nodes]} "
                  f"(syndromes len={[len(s) for _,s,_ in active_nodes]}), "
                  f"inactive={[n for n,_ in inactive_nodes]} "
                  f"(bp_corr={[len(c) for _,c in inactive_nodes]})")
            H_global, s_global, llr_global, registry = self._assemble_global_system(payloads)
            if H_global is not None:
                e_global = self._osd_gf2(H_global, s_global, llr_global=llr_global)
                K_tot = H_global.shape[1]
                if np.sum(e_global) > K_tot // 2:
                    print(f"[coordinator] WARNING: OSD returned {np.sum(e_global)}/{K_tot} corrections "
                          f"for {error_type}-errors — discarding (likely degenerate system)")
                    e_global = np.zeros_like(e_global)
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

        # Step 7: aggregate CNOT counts
        global_cnot_count = 0
        for name in self.node_names:
            msg = yield from context.csockets[name].recv()
            global_cnot_count += json.loads(msg)
        print(f"=== Global CNOT count = {global_cnot_count} ===\n")

        # Step 8: aggregate decoding times
        max_time = 0
        for name in self.node_names:
            msg = yield from context.csockets[name].recv()
            node_time = json.loads(msg)
            if node_time > max_time:
                max_time = node_time
        print(f"=== Max decoding time across nodes = {max_time:.2f} seconds ===\n")

        yield from context.connection.flush()
        return global_parity, global_cnot_count

    # Step 2 — Assemble block-diagonal global system
    def _assemble_global_system(self, payloads: list) -> tuple:
        active = [p for p in payloads if p.get("active", False)]
        if not active:
            return None, None, None, []

        H_blocks, s_list, llr_list, registry = [], [], [], []
        col_offset = 0

        for p in active:
            H_red_raw = np.array(p["H_reduced"], dtype=float)
            V_k = np.array(p["V_k"], dtype=float)
            k_i = int(p["k"])
            H_red = (np.abs(np.round(H_red_raw).astype(int)) % 2).astype(int)
            s_i = np.array(p["s"], dtype=int)

            H_blocks.append(H_red)
            s_list.extend(s_i.tolist())

            if "llr" in p and len(p["llr"]) == k_i:
                llr_list.extend(p["llr"])
            else:
                llr_list.extend([0.0] * k_i)

            registry.append({
                "node_id":        tuple(p["node_id"]),
                "V_k":            V_k,
                "data_positions": [tuple(pos) for pos in p["data_positions"]],
                "col_start":      col_offset,
                "col_end":        col_offset + k_i,
            })
            col_offset += k_i

        H_global  = block_diag(*H_blocks)
        s_global  = np.array(s_list, dtype=int)
        llr_global = np.array(llr_list, dtype=float)
        return H_global, s_global, llr_global, registry

    # Step 3 — Soft-decision OSD over GF(2)
    def _osd_gf2(self, H_global: np.ndarray, s_global: np.ndarray,
                 llr_global: np.ndarray = None, osd_order: int = 2) -> np.ndarray:
        m, K_tot = H_global.shape
        H = H_global.astype(int) % 2
        s = s_global.astype(int) % 2

        if llr_global is not None and len(llr_global) == K_tot:
            reliability = np.abs(llr_global)
        else:
            reliability = np.ones(K_tot)

        col_order = np.argsort(-reliability)
        inv_order  = np.empty(K_tot, dtype=int)
        inv_order[col_order] = np.arange(K_tot)

        H_ord = H[:, col_order].copy()
        H_sys, s_sys = H_ord.copy(), s.copy()
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
                    s_sys[r2]  = (s_sys[r2]  + s_sys[row])  % 2
            pivot_cols.append(col)
            pivot_rows.append(row)
            row += 1

        non_pivot = [c for c in range(K_tot) if c not in pivot_cols]
        non_pivot_sorted = sorted(non_pivot, key=lambda c: reliability[col_order[c]])
        test_cols = non_pivot_sorted[:osd_order]
        n_test = len(test_cols)

        def soft_cost(e_ord):
            return float(np.sum(reliability[col_order] * e_ord))

        best_cost = float("inf")
        best_e_ord = np.zeros(K_tot, dtype=int)

        for pattern in range(2 ** n_test):
            e_ord = np.zeros(K_tot, dtype=int)
            for b, tc in enumerate(test_cols):
                e_ord[tc] = (pattern >> b) & 1
            for pc, pr in zip(pivot_cols, pivot_rows):
                rhs = s_sys[pr]
                for fc in non_pivot:
                    rhs = (rhs + H_sys[pr, fc] * e_ord[fc]) % 2
                e_ord[pc] = rhs
            cost = soft_cost(e_ord)
            if cost < best_cost:
                best_cost  = cost
                best_e_ord = e_ord.copy()

        return best_e_ord[inv_order]

    # Step 4 — Back-project global error to per-node corrections
    def _project_corrections(self, e_global: np.ndarray, registry: list) -> dict:
        corrections = {}
        for reg in registry:
            e_block = e_global[reg["col_start"]:reg["col_end"]]
            V_k = reg["V_k"]
            e_local = (np.abs(np.round(np.array(V_k, dtype=float) @ e_block.astype(float)).astype(int)) % 2).astype(int)
            corrections[reg["node_id"]] = [
                list(reg["data_positions"][j])
                for j in range(len(e_local)) if e_local[j] == 1
            ]
        return corrections

    # Step 5 — Send corrections merging OSD + BP
    def _send_corrections(self, context: ProgramContext, payloads: list, corrections_per_node: dict):
        """Merge OSD corrections (active nodes) with BP corrections (inactive/converged nodes)."""
        bp_map: dict = {}
        for p in payloads:
            node_id = tuple(p["node_id"])
            if p.get("active", False):
                continue
            bp_corrs = p.get("bp_corrections", [])
            if bp_corrs:
                existing = bp_map.get(node_id, [])
                bp_map[node_id] = existing + bp_corrs

        for name in self.node_names:
            parts = name.replace("node_", "").split("_")
            node_id = (int(parts[0]), int(parts[1]))
            osd_corr = corrections_per_node.get(node_id, [])
            bp_corr  = bp_map.get(node_id, [])
            osd_set = set(tuple(c) for c in osd_corr)
            bp_set  = set(tuple(c) for c in bp_corr)
            merged  = list(osd_set.symmetric_difference(bp_set))
            context.csockets[name].send(json.dumps(merged))
        yield from context.connection.flush()