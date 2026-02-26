from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit
from netqasm.sdk.epr_socket import EPRSocket

class ClusterNodeProgram(Program):
    def __init__(self, node_coords, layout_manager):
        self.node_coords = node_coords
        self.layout_manager = layout_manager

    def run(self, context: ProgramContext):
        self.epr_borders = {
            "RIGHT": [],
            "LEFT": [],
            "UP": [],
            "DOWN": []
        }
        connection = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        r_node, c_node = self.node_coords

        # Nighbor RIGHT
        if c_node < self.layout_manager.nodes_per_side - 1:
            neighbor_right = f"node_{r_node}_{c_node + 1}"
            epr_socket = context.get_epr_socket(neighbor_right)
            
            for r_idx in range(self.layout_manager.block_size):
                epr_qubit = epr_socket.create_keep()[0]
                
                self.epr_borders["RIGHT"].append(epr_qubit)
                yield from connection.flush()

        # Neighbor LEFT
        if c_node > 0:
            neighbor_left = f"node_{r_node}_{c_node - 1}"
            epr_socket_left = context.get_epr_socket(neighbor_left)
            for r_idx in range(self.layout_manager.block_size):
                # Create EPR pair and keep the local qubit
                epr_qubit_left = epr_socket_left.recv_keep()[0]
                self.epr_borders["LEFT"].append(epr_qubit_left)
                yield from connection.flush()


        # Neighbor UP
        if r_node > 0:
            neighbor_up = f"node_{r_node - 1}_{c_node}"
            epr_socket_up = context.get_epr_socket(neighbor_up)
            
            for c_idx in range(self.layout_manager.block_size):
                epr_qubit_up = epr_socket_up.recv_keep()[0]
                self.epr_borders["UP"].append(epr_qubit_up)
                yield from connection.flush()
        
        # Neighbor DOWN
        if r_node < self.layout_manager.nodes_per_side - 1:
            neighbor_down = f"node_{r_node + 1}_{c_node}"
            epr_socket_down = context.get_epr_socket(neighbor_down)
            
            for c_idx in range(self.layout_manager.block_size):
                epr_qubit_down = epr_socket_down.create_keep()[0]
                self.epr_borders["DOWN"].append(epr_qubit_down)
                yield from connection.flush()
        
        
        yield from connection.flush()

        # Z stabilizer measurement
        if 