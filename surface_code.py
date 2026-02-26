class SurfaceLayout:
    def __init__(self, global_size, nodes_per_side):
        self.global_size = global_size
        self.nodes_per_side = nodes_per_side
        self.block_size = global_size // nodes_per_side

    def get_subgrid_for_node(self, node_row, node_col):
        subgrid = []
        r_start = node_row * self.block_size
        c_start = node_col * self.block_size
        
        for r in range(r_start, r_start + self.block_size):
            row = []
            for c in range(c_start, c_start + self.block_size):
                role = "pQ" if (r + c) % 2 == 0 else ("zQ" if c % 2 == 0 else "xQ")
                
                # Check if this qubit is on the border of the subgrid (for potential EPR connections)
                is_border = (r == r_start or r == r_start + self.block_size - 1 or
                             c == c_start or c == c_start + self.block_size - 1)
                
                row.append({"role": role, "is_border": is_border, "global_pos": (r, c)})
            subgrid.append(row)
        return subgrid