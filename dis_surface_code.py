from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit
from netqasm.sdk.epr_socket import EPRSocket

import random

class ClusterNodeProgram(Program):
    def __init__(self, node_coords, layout_manager):
        self.node_coords = node_coords
        self.layout_manager = layout_manager

    # Phase 1: Set up EPR pairs on borders with neighbors
    def _setup_epr_borders(self, context, subgrid_data):
        r, c = self.node_coords
        N = self.layout_manager.nodes_per_side
        B = self.layout_manager.block_size
        conn = context.connection

        borders = {"RIGHT": [], "LEFT": [], "UP": [], "DOWN": []}

        # Map for saving the index of EPR pairs for each border, based on the subgrid roles
        self.epr_index_map = {"UP": {}, "DOWN": {}, "LEFT": {}, "RIGHT": {}}

        # UP/DOWN → index by column c
        for col in range(B):
            if subgrid_data[0][col]["role"] in ("xQ", "zQ"):
                idx = len(self.epr_index_map["UP"])
                self.epr_index_map["UP"][col] = idx
            if subgrid_data[B - 1][col]["role"] in ("xQ", "zQ"):
                idx = len(self.epr_index_map["DOWN"])
                self.epr_index_map["DOWN"][col] = idx

        # LEFT/RIGHT → index by row r
        for row in range(B):
            if subgrid_data[row][0]["role"] in ("xQ", "zQ"):
                idx = len(self.epr_index_map["LEFT"])
                self.epr_index_map["LEFT"][row] = idx
            if subgrid_data[row][B - 1]["role"] in ("xQ", "zQ"):
                idx = len(self.epr_index_map["RIGHT"])
                self.epr_index_map["RIGHT"][row] = idx

        n_up    = len(self.epr_index_map["UP"])
        n_down  = len(self.epr_index_map["DOWN"])
        n_left  = len(self.epr_index_map["LEFT"])
        n_right = len(self.epr_index_map["RIGHT"])

        # RIGHT — this node is initiator
        if c < N - 1:
            sock = context.get_epr_socket(f"node_{r}_{c + 1}")
            for _ in range(n_right):
                borders["RIGHT"].append(sock.create_keep()[0])
                yield from conn.flush()

        # LEFT — this node is responder
        if c > 0:
            sock = context.get_epr_socket(f"node_{r}_{c - 1}")
            for _ in range(n_left):
                borders["LEFT"].append(sock.recv_keep()[0])
                yield from conn.flush()

        # UP — this node is responder
        if r > 0:
            sock = context.get_epr_socket(f"node_{r - 1}_{c}")
            for _ in range(n_up):
                borders["UP"].append(sock.recv_keep()[0])
                yield from conn.flush()

        # DOWN — this node is initiator
        if r < N - 1:
            sock = context.get_epr_socket(f"node_{r + 1}_{c}")
            for _ in range(n_down):
                borders["DOWN"].append(sock.create_keep()[0])
                yield from conn.flush()

        self.epr_borders = borders

    # Phase 2: Allocate local qubits based on subgrid data
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

    
    def _apply_noise(self):
        B = self.layout_manager.block_size
        probability = 0.01
        for r in range(B):
            for c in range(B):
                if self.qubit_roles[r][c] == "pQ":
                    if random.random() < probability:
                        self.local_qubits[r][c].X()  # Simulation of a bit-flip error on data qubits

    # Phase 3: Measure stabilizers and accumulate pending sends
    def _measure_stabilizers(self, context):
        conn = context.connection
        r_node, c_node = self.node_coords
        B = self.layout_manager.block_size
        N = self.layout_manager.nodes_per_side

        self.pending_sends = {"RIGHT": [], "LEFT": [], "UP": [], "DOWN": []}

        for r in range(B):
            for c in range(B):
                role = self.qubit_roles[r][c]
                if role not in ("xQ", "zQ"):
                    continue

                ancilla = self.local_qubits[r][c]

                if role == "zQ":
                    ancilla.H()

                # Neighboring qubits in the same subgrid
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
                    ancilla.cnot(nq)

                # Edge UP — use the map for index lookup
                if r == 0 and r_node > 0 and c in self.epr_index_map["UP"]:
                    epr_idx = self.epr_index_map["UP"][c]
                    epr = self.epr_borders["UP"][epr_idx]
                    ancilla.cnot(epr)
                    m = epr.measure()
                    yield from conn.flush()
                    self.pending_sends["UP"].append((c, int(m)))

                # Edge DOWN
                if r == B - 1 and r_node < N - 1 and c in self.epr_index_map["DOWN"]:
                    epr_idx = self.epr_index_map["DOWN"][c]
                    epr = self.epr_borders["DOWN"][epr_idx]
                    ancilla.cnot(epr)
                    m = epr.measure()
                    yield from conn.flush()
                    self.pending_sends["DOWN"].append((c, int(m)))

                # Edge LEFT
                if c == 0 and c_node > 0 and r in self.epr_index_map["LEFT"]:
                    epr_idx = self.epr_index_map["LEFT"][r]
                    epr = self.epr_borders["LEFT"][epr_idx]
                    ancilla.cnot(epr)
                    m = epr.measure()
                    yield from conn.flush()
                    self.pending_sends["LEFT"].append((r, int(m)))

                # Edge RIGHT
                if c == B - 1 and c_node < N - 1 and r in self.epr_index_map["RIGHT"]:
                    epr_idx = self.epr_index_map["RIGHT"][r]
                    epr = self.epr_borders["RIGHT"][epr_idx]
                    ancilla.cnot(epr)
                    m = epr.measure()
                    yield from conn.flush()
                    self.pending_sends["RIGHT"].append((r, int(m)))

                if role == "zQ":
                    ancilla.H()

                ancilla.measure()
                yield from conn.flush()

    # Phase 4: Exchange classical corrections with neighbors (all sends → all receives)
    def _exchange_classical_corrections(self, context):
        conn = context.connection
        r_node, c_node = self.node_coords
        N = self.layout_manager.nodes_per_side

        # SEND
        if r_node > 0:
            neighbor = f"node_{r_node - 1}_{c_node}"
            for idx, m in self.pending_sends["UP"]:
                conn.send_classical(neighbor, [idx, m])

        if r_node < N - 1:
            neighbor = f"node_{r_node + 1}_{c_node}"
            for idx, m in self.pending_sends["DOWN"]:
                conn.send_classical(neighbor, [idx, m])

        if c_node > 0:
            neighbor = f"node_{r_node}_{c_node - 1}"
            for idx, m in self.pending_sends["LEFT"]:
                conn.send_classical(neighbor, [idx, m])

        if c_node < N - 1:
            neighbor = f"node_{r_node}_{c_node + 1}"
            for idx, m in self.pending_sends["RIGHT"]:
                conn.send_classical(neighbor, [idx, m])

        yield from conn.flush()

        # RECEIVE — the neighbor's sent messages correspond to this node's opposite border
        received = {"UP": [], "DOWN": [], "LEFT": [], "RIGHT": []}

        if r_node < N - 1:   # receive from DOWN
            neighbor = f"node_{r_node + 1}_{c_node}"
            n_msgs = len(self.epr_index_map["DOWN"])
            for _ in range(n_msgs):
                msg = yield from conn.receive_classical(neighbor)
                received["DOWN"].append((msg[0], msg[1]))

        if r_node > 0:       # receive from UP
            neighbor = f"node_{r_node - 1}_{c_node}"
            n_msgs = len(self.epr_index_map["UP"])
            for _ in range(n_msgs):
                msg = yield from conn.receive_classical(neighbor)
                received["UP"].append((msg[0], msg[1]))

        if c_node < N - 1:   # receive from RIGHT
            neighbor = f"node_{r_node}_{c_node + 1}"
            n_msgs = len(self.epr_index_map["RIGHT"])
            for _ in range(n_msgs):
                msg = yield from conn.receive_classical(neighbor)
                received["RIGHT"].append((msg[0], msg[1]))

        if c_node > 0:       # receive from LEFT
            neighbor = f"node_{r_node}_{c_node - 1}"
            n_msgs = len(self.epr_index_map["LEFT"])
            for _ in range(n_msgs):
                msg = yield from conn.receive_classical(neighbor)
                received["LEFT"].append((msg[0], msg[1]))

        self.received_corrections = received

    # Phase 5: Apply corrections on border qubits based on received measurements
    def _apply_corrections(self):
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

    def run(self, context: ProgramContext):
        conn = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)

        yield from self._setup_epr_borders(context, subgrid_data)

        self._allocate_local_qubits(context, subgrid_data)
        self._apply_noise()
        yield from self._measure_stabilizers(context)
        yield from self._exchange_classical_corrections(context)

        self._apply_corrections()
        yield from conn.flush()