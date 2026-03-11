class SurfaceLayout:
    """Manages the surface code grid layout and node subgrid assignments."""
    
    def __init__(self, global_size: int, nodes_per_side: int):
        """Initialize surface code layout.
        
        Args:
            global_size: Total qubit grid size (global_size x global_size)
            nodes_per_side: Number of cluster nodes per dimension (N x N)
        """
        self.global_size = global_size
        self.nodes_per_side = nodes_per_side
        self.block_size = global_size // nodes_per_side

    def get_subgrid_for_node(self, node_row: int, node_col: int) -> list:
        """Get the subgrid details for a specific cluster node.
        
        Args:
            node_row: Row index of the cluster node
            node_col: Column index of the cluster node
            
        Returns:
            List of lists containing qubit details with keys:
            - 'role': Qubit type ("pQ" for data, "xQ"/"zQ" for ancilla)
            - 'is_border': Whether qubit is on subgrid boundary
            - 'global_pos': Global (row, col) coordinate
        """
        subgrid = []
        r_start = node_row * self.block_size
        c_start = node_col * self.block_size
        
        for r in range(r_start, r_start + self.block_size):
            row = []
            for c in range(c_start, c_start + self.block_size):
                # Determine qubit role based on checkerboard pattern
                role = "pQ" if (r + c) % 2 == 0 else ("zQ" if c % 2 == 0 else "xQ")
                
                # Check if on boundary for EPR connection tracking
                is_border = (r == r_start or r == r_start + self.block_size - 1 or
                             c == c_start or c == c_start + self.block_size - 1)
                
                row.append({
                    "role": role,
                    "is_border": is_border,
                    "global_pos": (r, c)
                })
            subgrid.append(row)
        return subgrid