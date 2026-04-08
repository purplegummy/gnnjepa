import networkx as nx
import random
import torch
from torch_geometric.utils import from_networkx


def build_graph(num_nodes=100):
    # Barabasi-Albert: scale-free network with hubs, like real travel/contact networks
    G = nx.barabasi_albert_graph(num_nodes, m=3)
    return G


def generate_region_features(num_nodes):
    """
    Static per-region features (fixed for the whole simulation):
      - population
      - hospital capacity per capita
      - population density
      - fraction elderly
    """
    population = torch.FloatTensor(num_nodes).uniform_(1e4, 5e6)
    hospitals = torch.FloatTensor(num_nodes).uniform_(1, 50)
    density = torch.FloatTensor(num_nodes).uniform_(10, 5000)
    elderly_frac = torch.FloatTensor(num_nodes).uniform_(0.05, 0.35)

    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)

    return torch.stack([
        norm(population),
        norm(hospitals / population),
        norm(density),
        norm(elderly_frac),
    ], dim=1)  # [num_nodes, 4]


def init_sir(num_nodes, initial_infected_frac=0.05):
    status = torch.zeros(num_nodes, dtype=torch.long)
    num_initial = max(1, int(initial_infected_frac * num_nodes))
    infected = random.sample(range(num_nodes), num_initial)
    status[infected] = 1
    return status


def sir_step(status, edge_index, region_features, action,
             base_beta=0.4, base_gamma=0.05):
    """
    One step of heterogeneous SIR with action effects:
      - vaccinate:          susceptible nodes move to recovered (applied before step)
      - increase hospitals: boosts gamma by 50% for targeted nodes
      - lockdown:           reduces beta by 50% for targeted nodes
    """
    num_nodes = status.shape[0]
    density  = region_features[:, 2]
    hosp_cap = region_features[:, 1]

    beta  = base_beta  * (0.5 + density)
    gamma = base_gamma * (0.5 + hosp_cap)

    # Action effects
    lockdown_mask   = action[:, 2].bool()
    hosp_boost_mask = action[:, 1].bool()
    beta[lockdown_mask]        *= 0.5
    gamma[hosp_boost_mask]     *= 1.5

    new_status = status.clone()
    src, dst = edge_index[0], edge_index[1]

    for node in range(num_nodes):
        if status[node] == 0:  # Susceptible
            neighbors = src[dst == node]
            infected_neighbors = (status[neighbors] == 1).sum().item()
            p_infect = 1 - (1 - beta[node].item()) ** infected_neighbors
            if random.random() < p_infect:
                new_status[node] = 1
        elif status[node] == 1:  # Infected
            if random.random() < gamma[node].item():
                new_status[node] = 2

    return new_status


def apply_actions(status, num_nodes, action_prob=0.3, vacc_rate=0.1, intervention_rate=0.1):
    """
    Returns action tensor of shape [num_nodes, 3]:
      col 0: vaccinate         (susceptible nodes only)
      col 1: increase hospitals (any node)
      col 2: lockdown           (any node)
    """
    action = torch.zeros(num_nodes, 3, dtype=torch.float)

    # Vaccinate
    if random.random() < action_prob:
        susceptible = (status == 0).nonzero(as_tuple=True)[0].tolist()
        num_to_vacc = min(int(vacc_rate * num_nodes), len(susceptible))
        if num_to_vacc > 0:
            for n in random.sample(susceptible, num_to_vacc):
                action[n, 0] = 1
                status[n] = 2

    # Increase hospital capacity
    if random.random() < action_prob:
        targets = random.sample(range(num_nodes), int(intervention_rate * num_nodes))
        for n in targets:
            action[n, 1] = 1

    # Lockdown
    if random.random() < action_prob:
        targets = random.sample(range(num_nodes), int(intervention_rate * num_nodes))
        for n in targets:
            action[n, 2] = 1

    return action


def status_to_onehot(status, num_nodes):
    onehot = torch.zeros(num_nodes, 3)
    for node in range(num_nodes):
        onehot[node, status[node]] = 1
    return onehot


def make_pyg_snapshot(G, sir_onehot, region_features):
    graph = from_networkx(G)
    graph.x = torch.cat([sir_onehot, region_features], dim=1)  # [num_nodes, 7]
    return graph


def generate_epidemiology_data(num_nodes=100, num_steps=1000,
                               base_beta=0.15, base_gamma=0.05,
                               action_prob=0.3, vacc_rate=0.1,
                               intervention_rate=0.1):
    """
    Region-level SIR time series on a fixed Barabasi-Albert graph.
    Node features: [S, I, R, population, hosp_capacity, density, elderly_frac]
    Actions per node: [vaccinate, increase_hospitals, lockdown]
    """
    G = build_graph(num_nodes)
    region_features = generate_region_features(num_nodes)
    status = init_sir(num_nodes)

    edge_index = from_networkx(G).edge_index

    data = []
    for step in range(num_steps):
        sir_onehot = status_to_onehot(status, num_nodes)
        graph_t = make_pyg_snapshot(G, sir_onehot, region_features)

        action = apply_actions(status, num_nodes, action_prob, vacc_rate, intervention_rate)

        next_status = sir_step(status, edge_index, region_features, action, base_beta, base_gamma)
        status = next_status

        # Reseed if epidemic dies out — simulates new outbreak arriving from outside
        if (status == 1).sum() == 0:
            susceptible = (status == 0).nonzero(as_tuple=True)[0].tolist()
            if len(susceptible) > 0:
                seeds = random.sample(susceptible, min(3, len(susceptible)))
                for n in seeds:
                    status[n] = 1

        sir_onehot_next = status_to_onehot(status, num_nodes)
        graph_t1 = make_pyg_snapshot(G, sir_onehot_next, region_features)

        data.append({'graph_t': graph_t, 'action': action, 'graph_t1': graph_t1})

        if (step + 1) % 100 == 0:
            n_infected = (status == 1).sum().item()
            print(f"Step {step+1}/{num_steps}  infected={n_infected}")

    return data


if __name__ == "__main__":
    data = generate_epidemiology_data()
    torch.save(data, 'data/epidemiology_data.pt')
    print(f"\nGenerated {len(data)} transitions saved to data/epidemiology_data.pt")

    sample = data[0]
    print(f"\nSample:")
    print(f"  Nodes:    {sample['graph_t'].x.shape[0]}")
    print(f"  Edges:    {sample['graph_t'].edge_index.shape[1]}")
    print(f"  Features: {sample['graph_t'].x.shape}  (3 SIR + 4 regional)")
    print(f"  Action:   {sample['action'].shape}  (vaccinate, hosp_boost, lockdown)")
