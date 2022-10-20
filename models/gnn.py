import torch.nn as nn
import feature_extractor
import sequence_model

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = feature_extractor.GATBlock()
        self.sequence_model = sequence_model.LSTMBlock()
        self.softmax = nn.Softmax(dim=1)

    def forward(self):
        pass
