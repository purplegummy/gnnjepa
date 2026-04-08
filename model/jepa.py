import copy
import torch
import torch.nn as nn


class JEPA(nn.Module):
    def __init__(self, encoder, predictor, ema_decay=0.99):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.ema_decay = ema_decay

        # Target encoder is an EMA copy — no gradients
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data = self.ema_decay * target.data + (1 - self.ema_decay) * online.data

    def forward(self, graph_t, action, graph_t_next):
        z_t = self.encoder(graph_t.x, graph_t.edge_index)
        pred_z = self.predictor(z_t, action, graph_t.edge_index)

        # Target encoder provides stable targets — no gradients
        with torch.no_grad():
            target_z = self.target_encoder(graph_t_next.x, graph_t_next.edge_index)

        return pred_z, target_z
