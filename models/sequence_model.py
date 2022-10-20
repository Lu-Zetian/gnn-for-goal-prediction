import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
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
    
    