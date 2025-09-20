# aig_qor/parse.py

import re
import networkx as nx
import torch
from torch_geometric.data import Data


def parse_bench(file_path: str) -> Data:
    """
    Parse a .bench file into a PyTorch Geometric Data graph.
    Nodes represent signals/gates with attributes:
      - type: input/output/gate/unknown
      - gate: gate type (INPUT, OUTPUT, AND, OR, etc.)

    Returns:
        torch_geometric.data.Data with:
            - x: node features [num_nodes, 3]
            - edge_index: [2, num_edges]
    """
    G = nx.DiGraph()
    assign_pat = re.compile(r'^\s*(\S+)\s*=\s*(\w+)\((.*?)\)', re.IGNORECASE)
    defined_signals = set()
    used_signals = set()

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = assign_pat.match(line)
            if match:
                output_signal, gate_type, inputs_str = match.groups()
                output_signal = output_signal.strip()
                gate_type = gate_type.upper().strip()
                inputs = [inp.strip() for inp in inputs_str.split(",") if inp.strip()]

                defined_signals.add(output_signal)
                G.add_node(output_signal, type="gate", gate=gate_type)

                for inp in inputs:
                    used_signals.add(inp)
                    if inp not in G:
                        G.add_node(inp, type="unknown", gate=None)
                    G.add_edge(inp, output_signal)

    # Mark inputs
    for signal in used_signals:
        if signal not in defined_signals:
            if signal in G:
                G.nodes[signal]["type"] = "input"
                G.nodes[signal]["gate"] = "INPUT"
            else:
                G.add_node(signal, type="input", gate="INPUT")

    # Mark outputs
    for signal in defined_signals:
        if G.out_degree(signal) == 0:
            G.nodes[signal]["type"] = "output"
            G.nodes[signal]["gate"] = "OUTPUT"

    # Map nodes to indices
    node_map = {n: i for i, n in enumerate(G.nodes())}
    edge_index = [[node_map[u], node_map[v]] for u, v in G.edges()]

    # Feature encoding: [is_input, is_output, gate_type_id]
    gate_types = {
        "INPUT": 0, "OUTPUT": 1, "AND": 2, "NAND": 3,
        "OR": 4, "NOR": 5, "XOR": 6, "XNOR": 7, "NOT": 8
    }
    x = []
    for node in G.nodes():
        attr = G.nodes[node]
        features = [
            1 if attr.get("type") == "input" else 0,
            1 if attr.get("type") == "output" else 0,
            gate_types.get(attr.get("gate", "AND"), 2),
        ]
        x.append(features)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    )
