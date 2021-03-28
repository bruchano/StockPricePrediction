import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import random
import pandas as pd
import pandas_datareader as web
import datetime
from math import *


class StockDataset(Dataset):
    def __init__(self, company, start, end=None, valid_size=0.1):
        self.train_data = []
        self.train_labels = []
        self.valid_data = []
        self.valid_labels = []

        self.data_inputs = []
        self.data_labels = []

        start_date = datetime.datetime(*start)
        if end:
            end_date = datetime.datetime(*end)
        else:
            end_date = datetime.datetime.today()

        data = web.DataReader(company, "yahoo", start_date, end_date)["Close"].values
        self.max = data.max()
        data /= self.max

        split = floor(len(data) * valid_size)

        for i in range(len(data) - 60):
            if i < split:

                self.valid_data.append(torch.from_numpy(data[i:i+60]).unsqueeze(0))
                self.valid_labels.append(torch.tensor(data[i+60].item()).unsqueeze(0))
            elif i >= split:
                self.train_data.append(torch.from_numpy(data[i:i+60]).unsqueeze(0))
                self.train_labels.append(torch.tensor(data[i+60].item()).unsqueeze(0))

        self.data_inputs = self.valid_data + self.train_data
        self.data_labels = self.valid_labels + self.train_labels
        self.valid_size = split

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, item):
        return self.data_inputs[item], self.data_labels[item]


def get_loaded_data(dataset, batch_size, num_workers=1):
    split = dataset.valid_size
    indice = [i for i in range(len(dataset))]
    valid_ind, train_ind = indice[:split], indice[split:]
    valid_sampler = SubsetRandomSampler(valid_ind)
    train_sampler = SubsetRandomSampler(train_ind)
    valid_data = DataLoader(dataset=dataset, batch_size=1, sampler=valid_sampler, num_workers=num_workers)
    training_data = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    return training_data, valid_data


if __name__ == "__main__":
    pass
