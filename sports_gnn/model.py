import torch.nn as nn
from sports_gnn.common import *

class SportsGNN(nn.Module):
    def __init__(self):
        # TODO: add edge attribute
        super().__init__()
        self.gat_block = GATBlock(dim_in=3, dim_h=16, dim_out=8, edge_dim=2, num_layers=3, heads=2)
        self.sum_pool = SumPool(in_features=16, hidden_size=64)
        self.game_state_encoder = nn.Linear(in_features=20, out_features=16)
        self.res_mlp = ResMLP(in_features=16)
        self.lstm_block = LSTMBlock(in_features=16, hidden_size=16, num_classes=2, num_layers=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph_data, game_state, hn, cn):
        x, _ = self.gat_block(graph_data)
        x = self.sum_pool(x)
        x = torch.cat((x, game_state), dim=0)
        x = self.game_state_encoder(x)
        x = self.res_mlp(x)
        x = torch.unsqueeze(x, 0)
        x, hn, cn = self.lstm_block(x, hn, cn)
        x = self.softmax(x)
        return x, hn, cn
    
    def init(self, device=None):
        return self.lstm_block.init(device)
