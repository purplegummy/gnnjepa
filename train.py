import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data

from model.encoder import Encoder
from model.predictor import Predictor
from model.jepa import JEPA


class CovidGraphDataset(Dataset):
    def __init__(self, path):
        # Each element is a PyG Data with x, edge_index, action, y
        # x      : [N, 3]  node features at t   — [daily_cases, daily_deaths, stringency]
        # action : [N, 4]  policy at t           — [C1, C2, C6, H6]
        # y      : [N, 3]  node features at t+1  — prediction target
        self.snapshots = torch.load(path, weights_only=False)

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        return self.snapshots[idx]


def collate_fn(batch):
    return batch


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 1e-3
    epochs = 50
    batch_size = 32
    ema_decay = 0.99

    dataset = CovidGraphDataset('data/covid_graphs.pt')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # in_dim=3  (daily_cases, daily_deaths, stringency)
    # predictor in_dim = out_dim(encoder) + action_dim = 16 + 4 = 20
    encoder = Encoder(in_dim=3, hidden_dim=32, out_dim=16)
    predictor = Predictor(in_dim=20, hidden_dim=32, out_dim=16)
    model = JEPA(encoder, predictor, ema_decay=ema_decay).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    pred_z = target_z = None  # for variance logging outside inner loop

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for snap in batch:
                graph_t = Data(x=snap.x, edge_index=snap.edge_index).to(device)
                # target graph uses t+1 node features, same topology
                graph_t1 = Data(x=snap.y, edge_index=snap.edge_index).to(device)
                action = snap.action.to(device)  # [N, 4]

                pred_z, target_z = model(graph_t, action, graph_t1)
                pred_z_n = nn.functional.normalize(pred_z, dim=-1)
                target_z_n = nn.functional.normalize(target_z, dim=-1)
                batch_loss = batch_loss + mse(pred_z_n, target_z_n)

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            model.update_target_encoder()
            total_loss += batch_loss.item()

        avg_train = total_loss / len(train_loader)
        if pred_z is not None and (epoch == 0 or (epoch + 1) % 10 == 0):
            pred_var = torch.var(pred_z, dim=0).mean().item()
            target_var = torch.var(target_z, dim=0).mean().item()
            print(f"  Pred var: {pred_var:.6f}, Target var: {target_var:.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_val = torch.tensor(0.0, device=device)
                for snap in batch:
                    graph_t = Data(x=snap.x, edge_index=snap.edge_index).to(device)
                    graph_t1 = Data(x=snap.y, edge_index=snap.edge_index).to(device)
                    action = snap.action.to(device)

                    pred_z, target_z = model(graph_t, action, graph_t1)
                    pred_z_n = nn.functional.normalize(pred_z, dim=-1)
                    target_z_n = nn.functional.normalize(target_z, dim=-1)
                    batch_val = batch_val + mse(pred_z_n, target_z_n)

                val_loss += (batch_val / len(batch)).item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

    torch.save(model.state_dict(), 'model_weights.pt')
    print("Saved model_weights.pt")


if __name__ == '__main__':
    train()
