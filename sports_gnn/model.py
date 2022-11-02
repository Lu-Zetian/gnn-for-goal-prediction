import torch.nn as nn
from sports_gnn.common import *

class SportsGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = GATBlock(3, 16, 8, 2)
        self.sum_pool = SumPool(8, 16)
        self.lstm = LSTMBlock(8, 16, 2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data, hn, cn):
        x, _ = self.gat(data)
        x = self.sum_pool(x)
        x = torch.unsqueeze(x, 0)
        x, hn, cn = self.lstm(x, hn, cn)
        x = self.softmax(x)
        return x, hn, cn
    
    def init(self, device=None):
        return self.lstm.init(device)
