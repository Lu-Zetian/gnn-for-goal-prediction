import torch.nn as nn
import torch.nn.functional as F
from sports_gnn.common import *

class SportsGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat_block = GATBlock(dim_in=3, dim_h=8, dim_out=4, edge_dim=2, num_layers=2, heads=3)
        self.sum_pool = SumPool(in_features=12, hidden_size=24)
        self.meta_data_encoder = MetaDataEncoder(in_features=6, out_features=4)
        self.res_fc = ResLinear(in_features=16)
        self.lstm_block = LSTMBlock(in_features=16, hidden_size=16, num_classes=3, num_layers=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph_data, meta_data, hn, cn):
        x, _ = self.gat_block(graph_data)
        x = self.sum_pool(x)
        
        meta_data = self.meta_data_encoder(meta_data)
        
        x = torch.cat((x, meta_data), dim=0)
        
        x = self.res_fc(x)
        
        x = torch.unsqueeze(x, 0)
        x, hn, cn = self.lstm_block(x, hn, cn)
        x = self.softmax(x)
        return x, hn, cn
    
    def init(self, device=None):
        return self.lstm_block.init(device)
