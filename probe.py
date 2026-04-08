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


def probe():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained JEPA
    encoder = Encoder(in_dim=7, hidden_dim=64, out_dim=32)
    predictor = Predictor(in_dim=35, hidden_dim=64, out_dim=32)
    model = JEPA(encoder, predictor).to(device)
    model.load_state_dict(torch.load('model_weights.pt', map_location=device))

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.eval()

    # Linear probe: embedding -> SIR class (3-way per node)
    probe = nn.Linear(32, 3).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = EpidemiologyDataset('data/epidemiology_data.pt')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    for epoch in range(20):
        probe.train()
        total_loss = total_correct = total_nodes = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for sample in batch:
                graph_t = sample['graph_t'].to(device)

                with torch.no_grad():
                    z = model.encoder(graph_t.x, graph_t.edge_index)  # [num_nodes, 32]

                # Ground truth SIR: argmax of first 3 features
                labels = graph_t.x[:, :3].argmax(dim=1)  # [num_nodes]

                logits = probe(z)
                batch_loss = batch_loss + criterion(logits, labels)

                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_nodes += labels.shape[0]

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        train_acc = total_correct / total_nodes

        probe.eval()
        val_correct = val_nodes = 0
        class_correct = torch.zeros(3)
        class_total = torch.zeros(3)
        with torch.no_grad():
            for batch in val_loader:
                for sample in batch:
                    graph_t = sample['graph_t'].to(device)
                    z = model.encoder(graph_t.x, graph_t.edge_index)
                    labels = graph_t.x[:, :3].argmax(dim=1)
                    preds = probe(z).argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_nodes += labels.shape[0]
                    for c in range(3):
                        mask = labels == c
                        class_correct[c] += (preds[mask] == c).sum().item()
                        class_total[c] += mask.sum().item()

        val_acc = val_correct / val_nodes
        per_class = class_correct / class_total.clamp(min=1)
        print(f"Epoch {epoch+1}/20  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
              f"  S={per_class[0]:.3f}  I={per_class[1]:.3f}  R={per_class[2]:.3f}")


if __name__ == '__main__':
    probe()
