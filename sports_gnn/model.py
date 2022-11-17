import torch.nn as nn
import torch.nn.functional as F
from sports_gnn.common import *

class SportsGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = GATBlock(dim_in=3, dim_h=8, dim_out=8, edge_dim=2, num_layers=3, heads=3)
        # self.conv_block = GConvBlock(dim_in=3, dim_h=24, dim_out=24, num_layers=2)
        self.sum_pool = SumPool(in_features=24, hidden_size=32)
        self.meta_data_encoder = MetaDataEncoder(in_features=6, out_features=8)
        self.res_fc1 = ResLinear(in_features=32)
        self.res_fc2 = ResLinear(in_features=32)
        self.fc1 = nn.Linear(in_features=32, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph_data, meta_data):
        x, _ = self.conv_block(graph_data)
        x = self.sum_pool(x)
        
        meta_data = self.meta_data_encoder(meta_data)
        
        x = torch.cat((x, meta_data), dim=0)
        
        x = self.res_fc1(x)
        x = self.res_fc2(x)
        
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        
        x = torch.unsqueeze(x, dim=0)
        x = self.softmax(x)
        return x
    
