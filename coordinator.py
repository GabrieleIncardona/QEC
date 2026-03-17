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
            max_qubits=1,
        )
    # Run

    def run(self, context: ProgramContext):
        # Step 1: receive payloads
        payloads = []
        for name in self.node_names:
            msg = yield from context.csockets[name].recv()
            payloads.append(json.loads(msg))

        # Step 2: assemble block-diagonal system
        H_global, s_global, registry = self._assemble_global_system(payloads)

        # Step 3: OSD over GF(2)
        if H_global is not None:
            e_global = self._osd_gf2(H_global, s_global)
            # Only apply corrections if a non-trivial error was found
            if np.any(e_global):
                corrections_per_node = self._project_corrections(e_global, registry)
            else:
                corrections_per_node = {}
        else:
            corrections_per_node = {}

        # Step 4+5: send corrections to each node
        yield from self._send_corrections(context, payloads, corrections_per_node)

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

    # Step 2 — Assemble block-diagonal global system
    def _assemble_global_system(self, payloads: list) -> tuple:
        active = [p for p in payloads if p.get("active", False)]
        if not active:
            return None, None, []

        H_blocks, s_list, registry = [], [], []
        col_offset = 0

        # Assumiamo che tutti i nodi abbiano lo stesso numero di round
        num_rounds = len(active[0]["s"]) 
        num_ancillas_total = sum(np.array(p["H_reduced"]).shape[0] for p in active)

        for p in active:
            H_red = np.array(p["H_reduced"]) 
            V_k = np.array(p["V_k"])
            # s_i è ora [[r1_bits], [r2_bits]]
            s_i_matrix = np.array(p["s"], dtype=int) 
            k_i = int(p["k"])

            H_blocks.append(H_red)
            # Appiattiamo lo storico delle sindromi: [round1_all_nodes, round2_all_nodes]
            # Ma dobbiamo farlo con attenzione dopo il block_diag
            
            registry.append({
                "node_id": tuple(p["node_id"]),
                "V_k": V_k,
                "data_positions": [tuple(pos) for pos in p["data_positions"]],
                "col_start": col_offset,
                "col_end": col_offset + k_i,
                "s_raw": s_i_matrix # Salviamo la matrice intera
            })
            col_offset += k_i

        # Matrice spaziale base
        H_space = block_diag(*H_blocks)
        
        # --- COSTRUZIONE MATRICE SPAZIOTEMPORALE ---
        # Se abbiamo T round e M ancille, la nuova H sarà (T*M) x (T*K + (T-1)*M)
        # Per semplicità, creiamo una versione che decodifica i data qubit 
        # considerando la persistenza degli errori.
        
        T = num_rounds
        M, K = H_space.shape
        
        # H_global sarà una matrice a blocchi dove H_space si ripete sulla diagonale
        # e le identità collegano i round per gli errori di misura
        H_global = block_diag(*[H_space for _ in range(T)])
        
        # Costruiamo s_global concatenando i round: [s1_node1, s1_node2, ..., s2_node1, ...]
        s_final = []
        for t in range(T):
            for reg in registry:
                s_final.extend(reg["s_raw"][t].tolist())
        s_global = np.array(s_final, dtype=int)

        return H_global, s_global, registry
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

        non_pivot = [c for c in range(K_tot) if c not in pivot_cols]
        test_cols = non_pivot[:osd_order]
        n_test = len(test_cols)

        best_weight = K_tot + 1
        best_e = np.zeros(K_tot, dtype=int)

        # --- Enumerate test patterns on free (non-pivot) columns ---
        for pattern in range(2 ** n_test):
            e = np.zeros(K_tot, dtype=int)
            for b, tc in enumerate(test_cols):
                e[tc] = (pattern >> b) & 1

            # Solve pivot bits mod 2
            for pc, pr in zip(pivot_cols, pivot_rows):
                rhs = s_sys[pr]
                for fc in non_pivot:
                    rhs = (rhs + H_sys[pr, fc] * e[fc]) % 2
                e[pc] = rhs

            weight = int(np.sum(e))
            if weight < best_weight:
                best_weight = weight
                best_e = e.copy()

        return best_e

    # Step 4 — Back-project global error to per-node corrections
    def _project_corrections(self, e_global: np.ndarray, registry: list) -> dict:
        corrections = {}
        K_tot_space = registry[-1]["col_end"] 
        
        for reg in registry:
            e_accumulated = np.zeros(reg["col_end"] - reg["col_start"], dtype=int)
            
            # Partiamo da t=1 per ignorare le parità iniziali del setup
            # e considerare solo il rumore iniettato tra Round 1 e Round 2
            for t in range(1, len(reg["s_raw"])): 
                start = t * K_tot_space + reg["col_start"]
                end = t * K_tot_space + reg["col_end"]
                e_accumulated = (e_accumulated + e_global[start:end]) % 2
            
            V_k = reg["V_k"]
            e_local = (np.round(np.abs(V_k @ e_accumulated)) % 2).astype(int)

            corrections[reg["node_id"]] = [
                list(reg["data_positions"][j])
                for j in range(len(e_local)) if e_local[j] == 1
            ]
        return corrections
    
    # Step 5 — Send corrections to nodes
    def _send_corrections(self, context: ProgramContext, payloads: list, corrections_per_node: dict):
        for p in payloads:
            node_id = tuple(p["node_id"])
            node_name = f"node_{node_id[0]}_{node_id[1]}"
            corr = corrections_per_node.get(node_id, [])
            context.csockets[node_name].send(json.dumps(corr))
        yield from context.connection.flush()