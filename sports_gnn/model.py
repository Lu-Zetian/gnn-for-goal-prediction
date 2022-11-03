import torch.nn as nn
from sports_gnn.common import *

class SportsGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat_block = GATBlock(3, 16, 8, 3, 2)
        self.sum_pool = SumPool(16, 64)
        self.game_state_encoder = nn.Linear(16, 16)
        self.res_mlp = ResMLP(16)
        self.lstm_block = LSTMBlock(16, 16, 2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph_data, game_state, hn, cn):
        x, _ = self.gat_block(graph_data)
        x = self.sum_pool(x)
        x = self.game_state_encoder(x)
        x = self.res_mlp(x)
        x, hn, cn = self.lstm_block(x, hn, cn)
        x = self.softmax(x)
        return x, hn, cn
    
    def init(self, device=None):
        return self.lstm_block.init(device)
