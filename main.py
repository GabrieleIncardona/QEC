import traceback
from contextlib import redirect_stdout
import netsquid as n
import argparse

from squidasm.run.stack.run import run as run_simulation
from squidasm.util.util import create_complete_graph_network
from netsquid_netbuilder.modules.qlinks.perfect import PerfectQLinkConfig
from netsquid_netbuilder.modules.clinks.default import DefaultCLinkConfig

from surface_code import SurfaceLayout
#from dis_surface_code import ClusterNodeProgram
from coordinator import CoordinatorProgram
from dis_surface_code_debug import ClusterNodeProgram

def main(error):
    global_size    = 5   # planar surface code with 10x10 qubits (100 qubits total)
    nodes_per_side = 2   # 3x3 grid of nodes, each node manages a 3x3 subgrid

    # Step 1: Create the Surface Layout Manager
    layout_manager = SurfaceLayout(global_size, nodes_per_side)

    # Step 2: Define node names - cluster nodes + coordinator
    cluster_node_names = []
    for r in range(nodes_per_side):
        for c in range(nodes_per_side):
            cluster_node_names.append(f"node_{r}_{c}")

    coordinator_name = "coordinator"
    all_node_names   = cluster_node_names + [coordinator_name]

    # Step 3: Configure the network (complete graph: every node can reach every other)
    cfg = create_complete_graph_network(
        node_names=all_node_names,
        link_typ="perfect",
        link_cfg=PerfectQLinkConfig(state_delay=100),
        clink_typ="default",
        clink_cfg=DefaultCLinkConfig(delay=500),
    )

    # Step 4: Create programs for each cluster node and coordinator
    programs = {}

    for r in range(nodes_per_side):
        for c in range(nodes_per_side):
            node_id = f"node_{r}_{c}"
            programs[node_id] = ClusterNodeProgram(
                node_coords=(r, c),
                layout_manager=layout_manager,
                coordinator_name=coordinator_name,
                error = error
            )

    programs[coordinator_name] = CoordinatorProgram(
        layout_manager=layout_manager,
    )

    # Step 5: Run the simulation
    num_runs = 100
    print("Running Distributed Surface Code simulation...")
    print(f"Global grid   : {global_size}x{global_size} qubits")
    print(f"Cluster nodes : {len(cluster_node_names)} nodes ")
    print(f"Decoder       : local SVD compression + global OSD at coordinator")

    try:
        results = run_simulation(config=cfg, programs=programs, num_times=num_runs)

        parities = []
        for node_results in results:
            for res in node_results:
                if isinstance(res, int):  # Logical parity results are integers (0 or 1) only sent by the coordinator
                    parities.append(res)
                    
        # Calculate statistics
        failures = sum(parities)                # Every 1 is a failure
        successes = num_runs - failures         # The rest are successes
        accuracy = (successes / num_runs) * 100 # Calculate percentage
        
        print(f"SIMULATION COMPLETE ({num_runs} runs)")
        print(f"Failures (Logical Error) : {failures}")
        print(f" Successes: {successes}")
        print(f"Accuracy: {accuracy:.2f}%")

    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--error", 
        type=str, 
        default="identity", 
        choices=["identity", "hadamard","initialization","readout","cnot", "all"],
        help="Type of error (default: %(default)s)"
    )
    
    args = parser.parse_args()
    with open('output.txt', 'w') as f:
        with redirect_stdout(f):
            main(args.error)
    sim_time_ns = n.sim_time()
    sim_time_ms = sim_time_ns/1_000_000

    print(f"Execution time: {sim_time_ms} ms")