
import torch.nn as nn
class JEPA(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, graph_t, action, graph_t_next):
        z_t = self.encoder(graph_t.x, graph_t.edge_index)
        z_t_next = self.encoder(graph_t_next.x, graph_t_next.edge_index)
        pred_z_t_next = self.predictor(z_t, action, graph_t.edge_index)
        return pred_z_t_next, z_t_next, z_t