import torch
import torch.nn.functional as F
import datetime as dt
import pandas_datareader


class StockPricePredictor(torch.nn.Module):
    def __init__(self, input_size=60, hidden_size=256, num_layers=1, dropout=0.):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.norm1 = torch.nn.LayerNorm(256)
        self.fc1 = torch.nn.Linear(256, 128)
        self.norm2 = torch.nn.LayerNorm(128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.norm3 = torch.nn.LayerNorm(32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x, h = self.gru(x)
        x = self.fc1(F.gelu(self.norm1(x)))
        x = self.fc2(F.gelu(self.norm2(x)))
        x = self.fc3(F.gelu(self.norm3(x)))
        return x

