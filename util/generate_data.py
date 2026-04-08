import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import random
import torch
from torch_geometric.utils import from_networkx


def build_graph(num_nodes=200, p_edge=0.15):
    G = nx.erdos_renyi_graph(num_nodes, p_edge)
    return G


def init_sir_model(G, beta=0.4, gamma=0.05, initial_infected_frac=0.05):
    num_initial = max(1, int(initial_infected_frac * len(G.nodes)))
    initial_infected = random.sample(list(G.nodes), num_initial)

    model = ep.SIRModel(G)
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)
    config.add_model_initial_configuration("Infected", initial_infected)
    model.set_initial_status(config)
    return model


def status_to_features(status, num_nodes):
    features = torch.zeros(num_nodes, 3)
    for node, state in status.items():
        features[node, state] = 1
    return features


def apply_action(model, status, num_nodes, action_prob=0.3, vacc_rate=0.1):
    action = torch.zeros(num_nodes, dtype=torch.long)
    if random.random() < action_prob:
        susceptible = [n for n, s in status.items() if s == 0]
        num_to_vacc = min(int(vacc_rate * num_nodes), len(susceptible))
        if num_to_vacc > 0:
            for n in random.sample(susceptible, num_to_vacc):
                action[n] = 1
                model.status[n] = 2  # vaccinate → recovered
    return action


def make_pyg_snapshot(G, features):
    graph = from_networkx(G)
    graph.x = features
    return graph


def generate_epidemiology_data(num_nodes=200, num_steps=1000,
                               p_edge=0.15, beta=0.4, gamma=0.05,
                               action_prob=0.3, vacc_rate=0.1):
    """
    Generate a single SIR epidemic time series on a fixed graph.
    Returns a list of (graph_t, action, graph_t+1) transitions.
    """
    G = build_graph(num_nodes, p_edge)
    model = init_sir_model(G, beta, gamma)

    data = []
    for step in range(num_steps):
        current_status = model.status.copy()
        graph_t = make_pyg_snapshot(G, status_to_features(current_status, num_nodes))
        action = apply_action(model, current_status, num_nodes, action_prob, vacc_rate)

        next_status = model.iteration()['status']
        graph_t1 = make_pyg_snapshot(G, status_to_features(next_status, num_nodes))

        data.append({'graph_t': graph_t, 'action': action, 'graph_t1': graph_t1})

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{num_steps}")

    return data


if __name__ == "__main__":
    data = generate_epidemiology_data()
    torch.save(data, 'data/epidemiology_data.pt')
    print(f"Generated {len(data)} transitions and saved to data/epidemiology_data.pt")

    sample = data[0]
    print(f"\nSample:")
    print(f"  Nodes:    {sample['graph_t'].x.shape[0]}")
    print(f"  Edges:    {sample['graph_t'].edge_index.shape[1]}")
    print(f"  Features: {sample['graph_t'].x.shape}")
    print(f"  Action:   {sample['action'].shape}")
