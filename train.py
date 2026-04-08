import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model.encoder import Encoder
from model.predictor import Predictor
from model.jepa import JEPA


class EpidemiologyDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return batch


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 1e-3
    epochs = 50
    batch_size = 32
    ema_decay = 0.996

    dataset = EpidemiologyDataset('data/epidemiology_data.pt')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(in_dim=3, hidden_dim=64, out_dim=32)
    predictor = Predictor(in_dim=33, hidden_dim=64, out_dim=32)
    model = JEPA(encoder, predictor, ema_decay=ema_decay).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for sample in batch:
                graph_t = sample['graph_t'].to(device)
                graph_t1 = sample['graph_t1'].to(device)
                action = sample['action'].float().unsqueeze(-1).to(device)

                pred_z, target_z = model(graph_t, action, graph_t1)
                # L2 normalize before MSE to prevent collapse
                pred_z_n = nn.functional.normalize(pred_z, dim=-1)
                target_z_n = nn.functional.normalize(target_z, dim=-1)
                batch_loss = batch_loss + mse(pred_z_n, target_z_n)

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            model.update_target_encoder()
            total_loss += batch_loss.item()

        avg_train = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_val = torch.tensor(0.0, device=device)
                for sample in batch:
                    graph_t = sample['graph_t'].to(device)
                    graph_t1 = sample['graph_t1'].to(device)
                    action = sample['action'].float().unsqueeze(-1).to(device)

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
