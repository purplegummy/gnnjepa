import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model.encoder import Encoder
from model.predictor import Predictor
from model.jepa import JEPA


class CovidGraphDataset(Dataset):
    def __init__(self, path):
        self.snapshots = torch.load(path, weights_only=False)

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        return self.snapshots[idx]


def collate_fn(batch):
    return batch


def r2_score(pred, target):
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot.clamp(min=1e-8)


def probe():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained JEPA with updated dims
    encoder = Encoder(in_dim=3, hidden_dim=64, out_dim=32)
    predictor = Predictor(in_dim=36, hidden_dim=64, out_dim=32)
    model = JEPA(encoder, predictor).to(device)
    model.load_state_dict(torch.load('model_weights.pt', map_location=device))

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.eval()

    dataset = CovidGraphDataset('data/covid_graphs.pt')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Linear probe: embedding -> next-week node features [daily_cases, daily_deaths, stringency]
    probe_head = nn.Linear(32, 3).to(device)
    optimizer = torch.optim.Adam(probe_head.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(20):
        probe_head.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for snap in batch:
                x = snap.x.to(device)
                edge_index = snap.edge_index.to(device)
                with torch.no_grad():
                    z = model.encoder(x, edge_index)  # [N, 32]

                pred = probe_head(z)  # [N, 3]
                batch_loss = batch_loss + criterion(pred, x)  # predict current features

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_train_loss = total_loss / len(train_loader)

        probe_head.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                for snap in batch:
                    x = snap.x.to(device)
                    edge_index = snap.edge_index.to(device)
                    z = model.encoder(x, edge_index)
                    pred = probe_head(z)
                    all_preds.append(pred)
                    all_targets.append(x)  # predict current features

        all_preds   = torch.cat(all_preds,   dim=0)  # [total_nodes, 3]
        all_targets = torch.cat(all_targets, dim=0)

        val_mse = criterion(all_preds, all_targets).item()
        # Per-output R²: [daily_cases, daily_deaths, stringency]
        r2 = [r2_score(all_preds[:, i], all_targets[:, i]).item() for i in range(3)]

        print(f"Epoch {epoch+1}/20  train_mse={avg_train_loss:.4f}  val_mse={val_mse:.4f}"
              f"  R²: cases={r2[0]:.3f}  deaths={r2[1]:.3f}  stringency={r2[2]:.3f}")


if __name__ == '__main__':
    probe()
