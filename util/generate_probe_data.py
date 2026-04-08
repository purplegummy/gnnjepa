import random
import torch
from generate_data import build_graph, generate_region_features, status_to_onehot, make_pyg_snapshot, sir_step
from torch_geometric.utils import from_networkx


def generate_probe_data(num_nodes=100, num_steps=1000,
                        base_beta=0.05, base_gamma=0.15, waning_rate=0.05):
    """
    Generates data with more balanced S/I/R distribution for linear probing.
    Higher beta, gamma, and waning rate keeps all three states active.
    No actions — pure dynamics.
    """
    G = build_graph(num_nodes)
    region_features = generate_region_features(num_nodes)
    edge_index = from_networkx(G).edge_index

    # Start with a third of nodes in each state
    status = torch.zeros(num_nodes, dtype=torch.long)
    indices = list(range(num_nodes))
    random.shuffle(indices)
    for i in indices[:num_nodes//3]:
        status[i] = 1
    for i in indices[num_nodes//3:2*num_nodes//3]:
        status[i] = 2

    action = torch.zeros(num_nodes, 3)  # no actions

    data = []
    for step in range(num_steps):
        sir_onehot = status_to_onehot(status, num_nodes)
        graph = make_pyg_snapshot(G, sir_onehot, region_features)
        labels = graph.x[:, :3].argmax(dim=1)
        data.append({'graph': graph, 'labels': labels})

        status = sir_step(status, edge_index, region_features, action, base_beta, base_gamma, waning_rate)

        if (step + 1) % 100 == 0:
            counts = [(status == c).sum().item() for c in range(3)]
            print(f"Step {step+1}/{num_steps}  S={counts[0]} I={counts[1]} R={counts[2]}")

    return data


if __name__ == "__main__":
    data = generate_probe_data()
    torch.save(data, 'data/probe_data.pt')
    print(f"Generated {len(data)} samples saved to data/probe_data.pt")
