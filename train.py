import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model.encoder import Encoder
from model.predictor import Predictor
from model.jepa import JEPA
from model.sigreg import SIGReg


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
    sigreg_weight = 0.01

    dataset = EpidemiologyDataset('data/epidemiology_data.pt')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(in_dim=3, hidden_dim=64, out_dim=32)
    predictor = Predictor(in_dim=33, hidden_dim=64, out_dim=32)
    model = JEPA(encoder, predictor).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    sigreg = SIGReg().to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        total_mse = 0.0
        total_reg = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            batch_mse = torch.tensor(0.0, device=device)
            batch_reg = torch.tensor(0.0, device=device)

            all_emb = []
            for sample in batch:
                graph_t = sample['graph_t'].to(device)
                graph_t1 = sample['graph_t1'].to(device)
                action = sample['action'].float().unsqueeze(-1).to(device)  # [num_nodes, 1]

                pred_z, target_z, z_t = model(graph_t, action, graph_t1)

                mse_loss = mse(pred_z, target_z.detach())
                batch_loss = batch_loss + mse_loss
                batch_mse = batch_mse + mse_loss
                all_emb.append(z_t)
                all_emb.append(target_z)

            # SIGReg over full sequence: [T, num_nodes, D] -> matches le-wm's (T, B, D)
            z_stack = torch.stack(all_emb, dim=0)
            reg_loss = sigreg(z_stack)
            batch_loss = batch_loss / len(batch) + sigreg_weight * reg_loss
            batch_reg = batch_reg + reg_loss
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            total_mse += (batch_mse / len(batch)).item()
            total_reg += (batch_reg / len(batch)).item()

        avg_train = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        avg_reg = total_reg / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_val = torch.tensor(0.0, device=device)
                all_emb_val = []
                for sample in batch:
                    graph_t = sample['graph_t'].to(device)
                    graph_t1 = sample['graph_t1'].to(device)
                    action = sample['action'].float().unsqueeze(-1).to(device)

                    pred_z, target_z, z_t = model(graph_t, action, graph_t1)
                    batch_val = batch_val + mse(pred_z, target_z)
                    all_emb_val.append(z_t)
                    all_emb_val.append(target_z)

                z_stack_val = torch.stack(all_emb_val, dim=0)
                reg_loss = sigreg(z_stack_val)
                val_loss += (batch_val / len(batch) + sigreg_weight * reg_loss).item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}  train={avg_train:.4f} (mse={avg_mse:.4f} reg={avg_reg:.4f})  val={avg_val:.4f}")

    torch.save(model.state_dict(), 'model_weights.pt')
    print("Saved model_weights.pt")


if __name__ == '__main__':
    train()
