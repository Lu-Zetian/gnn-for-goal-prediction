import torch.nn as nn
from sports_gnn.common import *

class SportsGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = GATBlock()
        self.lstm = LSTMBlock()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return
