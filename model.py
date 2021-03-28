import torch
import torch.nn.functional as F
import datetime as dt
import pandas_datareader


class FullyConnected(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.norm = torch.nn.LayerNorm(in_features)
        self.leakyrelu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.leakyrelu(x)
        return x


class StockPricePredictor(torch.nn.Module):
    def __init__(self, n_fc, input_size=60, dropout=0.05):
        super().__init__()
        self.fc = torch.nn.Sequential()
        self.fc.add_module("0", FullyConnected(input_size, n_fc[0], dropout))
        for i in range(len(n_fc) - 1):
            self.fc.add_module(str(i + 1), FullyConnected(n_fc[i], n_fc[i + 1], dropout))
        self.fc.add_module(str(len(n_fc)), FullyConnected(n_fc[-1], 1, dropout))

    def forward(self, x):
        x = x.reshape(-1, 60)
        return self.fc(x)

