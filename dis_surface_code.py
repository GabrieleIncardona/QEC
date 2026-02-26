from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from netqasm.sdk.qubit import Qubit
from netqasm.sdk.epr_socket import EPRSocket

class ClusterNodeProgram(Program):
    def __init__(self, node_coords, layout_manager):
        self.node_coords = node_coords
        self.layout_manager = layout_manager

    def run(self, context: ProgramContext):
        connection = context.connection
        subgrid_data = self.layout_manager.get_subgrid_for_node(*self.node_coords)
        
        local_qubits = []
        for r in range(len(subgrid_data)):
            row = []
            for c in range(len(subgrid_data[0])):
                q = Qubit(connection)
                row.append(q)
            local_qubits.append(row)

        # 2. Definizione dei socket EPR per i bordi
        # Esempio: Se non sono l'ultima colonna, avrò un socket verso il vicino a destra
        # epr_socket = EPRSocket("Nodo_Destra") 
        
        # 3. Logica di errore/simulazione locale
        # (Qui applicheresti i tuoi Pauli errors sulla sottomatrice)

        yield from connection.flush()
        return {"status": "Subgrid initialized", "coords": self.node_coords}