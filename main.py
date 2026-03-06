import traceback


from squidasm.run.stack.run import run as run_simulation
from squidasm.util.util import create_complete_graph_network
#from squidasm.run.stack.config import DepolariseLinkConfig, DefaultCLinkConfig, GenericQDeviceConfig
from netsquid_netbuilder.modules.qlinks.perfect import PerfectQLinkConfig
from netsquid_netbuilder.modules.clinks.default import DefaultCLinkConfig

from surface_code import SurfaceLayout
from dis_surface_code import ClusterNodeProgram

def main():
    global_size = 4        # planar surface code with 4x4 qubits (16 qubits)
    nodes_per_side = 2    # 2x2 nodes, each node menage 2x2 qubits (4 qubits)     
    
    # 1. Creation of the Surface Layout Manager
    layout_manager = SurfaceLayout(global_size, nodes_per_side)

    # 2. Defination of Node Names
    node_names = []
    for r in range(nodes_per_side):
        for c in range(nodes_per_side):
            node_names.append(f"node_{r}_{c}")

    # 3. Configuration of the Network
    cfg = create_complete_graph_network(
        node_names=node_names,
        link_typ="perfect",
        link_cfg=PerfectQLinkConfig(state_delay=0),
        clink_typ="default", # classical Link
        clink_cfg=DefaultCLinkConfig(delay=0),
    )

    # 4. creation of Programs for each node
    programs = {}
    for r in range(nodes_per_side):
        for c in range(nodes_per_side):
            node_id = f"node_{r}_{c}"
            programs[node_id] = ClusterNodeProgram(
                node_coords=(r, c), 
                layout_manager=layout_manager
            )

    # 5. Run the simulation
    print(f"Run simulation of Distribuit Surface Code...")
    print(f"global grid: {global_size}x{global_size}")
    print(f"number node: {len(node_names)} (every node menage {layout_manager.block_size}x{layout_manager.block_size} qubit)")
    
    try:
        results = run_simulation(config=cfg, programs=programs, num_times=1)
        print("\n--- RISULTS ---")
        """for node, data in results.items():
            print(f"{node}: {data}")"""
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()