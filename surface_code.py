class SurfaceLayout:
    """Manages the surface code grid layout and node subgrid assignments."""
    
    def __init__(self, global_size: int, nodes_per_side: int):
        self.global_size = global_size
        self.nodes_per_side = nodes_per_side
        self.block_size = global_size // nodes_per_side

    def get_qubit_role(self, r_global: int, c_global: int) -> str:
        """Determines the qubit role based on the global grid."""
        # Standard logic: 
        # (r+c) even -> Data Qubit (pQ)
        # (r+c) odd -> Ancilla (xQ or zQ)
        if (r_global + c_global) % 2 == 0:
            return "pQ"
        else:
            # Division between X and Z stabilizers
            # In a checkerboard, we use the parity of r or c to distinguish xQ from zQ
            return "zQ" if r_global % 2 != 0 else "xQ"

    def get_subgrid_for_node(self, node_row: int, node_col: int) -> list:
        subgrid = []
        r_start = node_row * self.block_size
        c_start = node_col * self.block_size

        r_end = self.global_size if node_row == self.nodes_per_side - 1 else r_start + self.block_size
        c_end = self.global_size if node_col == self.nodes_per_side - 1 else c_start + self.block_size

        for r in range(r_start, r_end):
            row = []
            for c in range(c_start, c_end):
                role = self.get_qubit_role(r, c)
                
                is_border = (r == r_start or r == r_end - 1 or
                            c == c_start or c == c_end - 1)
                
                row.append({
                    "role": role,
                    "is_border": is_border,
                    "global_pos": (r, c)
                })
            subgrid.append(row)
        return subgrid