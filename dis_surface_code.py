from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit
from netqasm.sdk.epr_socket import EPRSocket


# ---------------------------------------------------------------------------
# Helper: per ogni direzione definisce chi fa create_keep (initiator)
# e chi fa recv_keep (responder) in modo coerente tra i nodi.
# La convenienza scelta è:
#   RIGHT/DOWN → il nodo corrente è l'initiator (create_keep)
#   LEFT/UP    → il nodo corrente è il responder  (recv_keep)
# Questo garantisce che per ogni coppia (A→B) A chiama create e B chiama recv,
# eliminando il deadlock simmetrico.
# ---------------------------------------------------------------------------

class ClusterNodeProgram(Program):
    def __init__(self, node_coords, layout_manager):
        self.node_coords = node_coords
        self.layout_manager = layout_manager

    # ------------------------------------------------------------------
    # FASE 1: Creazione dei qubit EPR sui bordi
    # ------------------------------------------------------------------
    def _setup_epr_borders(self, context):
        """
        Popola self.epr_borders con i qubit EPR verso i 4 vicini.
        Ordine coerente globalmente:
          - RIGHT / DOWN  → create_keep  (initiator)
          - LEFT  / UP    → recv_keep    (responder)
        In questo modo per ogni link fisico un solo nodo chiama create
        e l'altro chiama recv, senza deadlock.
        """
        r, c = self.node_coords
        N = self.layout_manager.nodes_per_side
        B = self.layout_manager.block_size
        conn = context.connection

        borders = {"RIGHT": [], "LEFT": [], "UP": [], "DOWN": []}

        # RIGHT — questo nodo è initiator
        if c < N - 1:
            sock = context.get_epr_socket(f"node_{r}_{c + 1}")
            for _ in range(B):
                borders["RIGHT"].append(sock.create_keep()[0])
                yield from conn.flush()

        # LEFT — questo nodo è responder
        if c > 0:
            sock = context.get_epr_socket(f"node_{r}_{c - 1}")
            for _ in range(B):
                borders["LEFT"].append(sock.recv_keep()[0])
                yield from conn.flush()

        # UP — questo nodo è responder
        if r > 0:
            sock = context.get_epr_socket(f"node_{r - 1}_{c}")
            for _ in range(B):
                borders["UP"].append(sock.recv_keep()[0])
                yield from conn.flush()

        # DOWN — questo nodo è initiator
        if r < N - 1:
            sock = context.get_epr_socket(f"node_{r + 1}_{c}")
            for _ in range(B):
                borders["DOWN"].append(sock.create_keep()[0])
                yield from conn.flush()

        self.epr_borders = borders

    # ------------------------------------------------------------------
    # FASE 2: Allocazione qubit locali
    # ------------------------------------------------------------------
    def _allocate_local_qubits(self, context, subgrid_data):
        conn = context.connection
        self.local_qubits = []
        self.qubit_roles = []

        for row in subgrid_data:
            row_q, row_roles = [], []
            for cell in row:
                row_q.append(Qubit(conn))
                row_roles.append(cell["role"])
            self.local_qubits.append(row_q)
            self.qubit_roles.append(row_roles)

    # ------------------------------------------------------------------
    # Helper: applica i CNOT di bordo usando il qubit EPR condiviso
    # Restituisce il valore di misura da inviare al vicino.
    # ------------------------------------------------------------------
    def _apply_border_cnot(self, ancilla, epr_qubit, connection):
        """
        Esegue il teleportation-assisted CNOT sul bordo:
          ancilla ──CNOT──> epr_qubit  → misura epr_qubit → restituisce m
        Il chiamante dovrà inviare m al vicino (fase 3) e ricevere la
        correzione dal vicino (fase 4).
        """
        ancilla.cnot(epr_qubit)
        m = epr_qubit.measure()
        return m  # il flush e il send vengono fatti in _measure_stabilizers

    # ------------------------------------------------------------------
    # FASE 3: Misura degli stabilizzatori + raccolta messaggi da inviare
    # ------------------------------------------------------------------
    def _measure_stabilizers(self, context):
        """
        Esegue le misure degli stabilizzatori X e Z.
        NON invia né riceve messaggi classici durante questa fase:
        raccoglie solo i valori m da spedire, poi li spedisce TUTTI
        alla fine (fase 4), così i due nodi non si aspettano mai a vicenda
        nel mezzo del loop.
        """
        conn = context.connection
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
        N = self.layout_manager.nodes_per_side

        # pending_sends[direction] = lista di (index, m_value)
        self.pending_sends = {"RIGHT": [], "LEFT": [], "UP": [], "DOWN": []}

        def apply_interior_cnots(ancilla, neighbors):
            for nq in neighbors:
                yield ancilla.cnot(nq)

        for r in range(B):
            for c in range(B):
                role = self.qubit_roles[r][c]
                if role not in ("xQ", "zQ"):
                    continue

                ancilla = self.local_qubits[r][c]

                if role == "zQ":
                    ancilla.H()

                # --- Vicini interni ---
                interior_neighbors = []
                if r > 0:
                    interior_neighbors.append(self.local_qubits[r - 1][c])
                if r < B - 1:
                    interior_neighbors.append(self.local_qubits[r + 1][c])
                if c > 0:
                    interior_neighbors.append(self.local_qubits[r][c - 1])
                if c < B - 1:
                    interior_neighbors.append(self.local_qubits[r][c + 1])

                for nq in interior_neighbors:
                    yield ancilla.cnot(nq)

                # --- Vicini di bordo (EPR) ---
                # Per ogni bordo: esegui CNOT + misura EPR, accumula il risultato
                if r == 0 and r_node > 0:           # bordo UP
                    epr = self.epr_borders["UP"][c]
                    ancilla.cnot(epr)
                    m = epr.measure()
                    yield from conn.flush()
                    self.pending_sends["UP"].append((c, int(m)))

                if r == B - 1 and r_node < N - 1:   # bordo DOWN
                    epr = self.epr_borders["DOWN"][c]
                    ancilla.cnot(epr)
                    m = epr.measure()
                    yield from conn.flush()
                    self.pending_sends["DOWN"].append((c, int(m)))

                if c == 0 and c_node > 0:           # bordo LEFT
                    epr = self.epr_borders["LEFT"][r]
                    ancilla.cnot(epr)
                    m = epr.measure()
                    yield from conn.flush()
                    self.pending_sends["LEFT"].append((r, int(m)))

                if c == B - 1 and c_node < N - 1:   # bordo RIGHT
                    epr = self.epr_borders["RIGHT"][r]
                    ancilla.cnot(epr)
                    m = epr.measure()
                    yield from conn.flush()
                    self.pending_sends["RIGHT"].append((r, int(m)))

                if role == "zQ":
                    ancilla.H()

                yield ancilla.measure()

    # ------------------------------------------------------------------
    # FASE 4: Scambio classico dei risultati EPR (send poi receive)
    # ------------------------------------------------------------------
    def _exchange_classical_corrections(self, context):
        """
        Tutti i nodi eseguono PRIMA tutti i send, POI tutti i receive.
        Questo schema simmetrico elimina il deadlock: nessun nodo
        si blocca in un receive mentre l'altro è ancora in un send.

        Ordine di send/receive scelto in modo speculare:
          - chi ha inviato verso RIGHT riceve da RIGHT (e viceversa)
          - chi ha inviato verso DOWN  riceve da DOWN  (e viceversa)
        I nodi adiacenti concordano sullo stesso ordine perché uno
        invia RIGHT mentre l'altro riceve LEFT.
        """
        conn = context.connection
        r_node, c_node = self.node_coords
        N = self.layout_manager.nodes_per_side

        # --- SEND (tutti i send prima di qualsiasi receive) ---
        if r_node > 0:       # invia verso UP
            neighbor = f"node_{r_node - 1}_{c_node}"
            for idx, m in self.pending_sends["UP"]:
                conn.send_classical(neighbor, [idx, m])

        if r_node < N - 1:   # invia verso DOWN
            neighbor = f"node_{r_node + 1}_{c_node}"
            for idx, m in self.pending_sends["DOWN"]:
                conn.send_classical(neighbor, [idx, m])

        if c_node > 0:       # invia verso LEFT
            neighbor = f"node_{r_node}_{c_node - 1}"
            for idx, m in self.pending_sends["LEFT"]:
                conn.send_classical(neighbor, [idx, m])

        if c_node < N - 1:   # invia verso RIGHT
            neighbor = f"node_{r_node}_{c_node + 1}"
            for idx, m in self.pending_sends["RIGHT"]:
                conn.send_classical(neighbor, [idx, m])

        yield from conn.flush()

        # --- RECEIVE (stesso ordine speculare, ora tutti i send sono avvenuti) ---
        received = {"UP": [], "DOWN": [], "LEFT": [], "RIGHT": []}
        B = self.layout_manager.block_size

        if r_node < N - 1:   # ricevi da DOWN (il vicino ci ha inviato verso UP)
            neighbor = f"node_{r_node + 1}_{c_node}"
            for _ in range(B):
                msg = yield from conn.receive_classical(neighbor)
                received["DOWN"].append((msg[0], msg[1]))

        if r_node > 0:       # ricevi da UP
            neighbor = f"node_{r_node - 1}_{c_node}"
            for _ in range(B):
                msg = yield from conn.receive_classical(neighbor)
                received["UP"].append((msg[0], msg[1]))

        if c_node < N - 1:   # ricevi da RIGHT
            neighbor = f"node_{r_node}_{c_node + 1}"
            for _ in range(B):
                msg = yield from conn.receive_classical(neighbor)
                received["RIGHT"].append((msg[0], msg[1]))

        if c_node > 0:       # ricevi da LEFT
            neighbor = f"node_{r_node}_{c_node - 1}"
            for _ in range(B):
                msg = yield from conn.receive_classical(neighbor)
                received["LEFT"].append((msg[0], msg[1]))

        self.received_corrections = received

    # ------------------------------------------------------------------
    # FASE 5: Applica le correzioni X sui qubit di bordo
    # ------------------------------------------------------------------
    def _apply_corrections(self):
        """
        Applica X sui qubit fisici di bordo se il vicino ha misurato m=1.
        Eseguita dopo la ricezione completa, senza flush necessari.
        """
        for idx, m in self.received_corrections.get("UP", []):
            if m == 1:
                self.local_qubits[0][idx].X()

        for idx, m in self.received_corrections.get("DOWN", []):
            if m == 1:
                self.local_qubits[-1][idx].X()

        for idx, m in self.received_corrections.get("LEFT", []):
            if m == 1:
                self.local_qubits[idx][0].X()

        for idx, m in self.received_corrections.get("RIGHT", []):
            if m == 1:
                self.local_qubits[idx][-1].X()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self, context: ProgramContext):
        conn = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)

        # Fase 1 — EPR sui bordi
        yield from self._setup_epr_borders(context)

        # Fase 2 — Qubit locali
        self._allocate_local_qubits(context, subgrid_data)

        # Fase 3 — Misura stabilizzatori (accumula i pending_sends)
        yield from self._measure_stabilizers(context)

        # Fase 4 — Scambio classico (tutti i send → tutti i receive)
        yield from self._exchange_classical_corrections(context)

        # Fase 5 — Correzioni sui qubit di bordo
        self._apply_corrections()

        yield from conn.flush()