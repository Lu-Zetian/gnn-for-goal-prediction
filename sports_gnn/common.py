from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class GATBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_h=0, num_layers=1, heads=1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GATv2Conv(dim_in, dim_h, heads))
        for _ in range(self.num_layers-2):
            if (dim_h == 0):
                raise Exception("Missing dim_h in GATBlock.__init__()")
            self.convs.append(pyg_nn.GATv2Conv(dim_h*heads, dim_h, heads))
        self.convs.append(pyg_nn.GATv2Conv(dim_h*heads, dim_out, heads))
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, m in enumerate(self.convs):
            x = m(x, edge_index)
            if i != self.num_layers-1:
                x = F.leaky_relu(x, 0.1)
                x = F.dropout(x, p=0.5)
        return torch.sigmoid(x)
    

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, hn, cn):
        x = self.batch_norm(x)
        out, (hn, cn) = self.lstm(x, (hn, cn))
        print(out.size())
        final_out = self.fc(out)
        return final_out, hn, cn
    
    def init(self, device=None):
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.hidden_size)
        if device:
            h0, c0 = h0.to(device), c0.to(device)
        return h0, c0
    
    