import torch
import torch.optim as optim
import numpy as np
import pandas_datareader as datareader
import datetime as dt
import matplotlib.pyplot as plt
from random import shuffle
from plot import *
from data import *
from model import StockPricePredictor

COMPANY = "TSLA"
START = [2011, 1, 1]
END = [2021, 2, 23]

epoch = 5
lr = 5e-4
MODEL_PATH = None
SAVE_PATH = COMPANY + "_model.pt"


def train(input_size, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StockPricePredictor().to(device)
    model.train()

    loss_plot = []
    MSELoss = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data = get_stock_close_price(COMPANY, START, END)
    data = torch.from_numpy(data)

    for e in range(epoch):
        for i in range(len(data) - input_size):
            optimizer.zero_grad()

            highest = data[i: i + input_size + 1].max().float()
            price = data[i: i + input_size].float() / highest
            price = price.reshape(1, 1, -1)
            true = data[i + input_size].float() / highest

            predicted = model(price)
            loss = MSELoss(predicted, true)

            loss_plot.append(loss.item())
            print("predicted:", predicted.item())
            print("true:", true.item())
            print("loss:", loss.item())

            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), SAVE_PATH)
    plot_loss(loss_plot)


def evaluate(model_path, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StockPricePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predicted_plot = []
    true_plot = []

    data = get_stock_close_price(COMPANY, START, END)
    data = torch.from_numpy(data)

    for i in range(len(data) - input_size):
        highest = data[i: i + input_size + 1].max().float()
        price = data[i: i + input_size].float() / highest
        price = price.reshape(1, 1, -1)
        true = data[i + input_size].float()

        true_plot.append(true)
        predicted = model(price)
        print(predicted)
        predicted *= highest
        predicted_plot.append(predicted)

    plot_accuracy(predicted_plot, true_plot)




train(60)
evaluate(SAVE_PATH, 60)



