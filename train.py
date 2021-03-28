import torch
import torch.optim as optim
import numpy as np
import pandas_datareader as datareader
import datetime as dt
import matplotlib.pyplot as plt
from random import shuffle
import argparse
from plot import *
from model import StockPricePredictor
from data_loader import *

COMPANY = "GOOG"
START = [2017, 1, 1]
END = None

N_FC = [1024, 512, 256]
BATCH_SIZE = 4
epoch = 10
lr = 1e-4
VER = "2"
SAVE_PATH = os.path.join("models", COMPANY + "_lr_" + str(lr) + "_ver_" + VER + ".pt")
MODEL_PATH = None

ACCURACY_PATH = os.path.join("comparison", COMPANY + "_lr_" + str(lr) + "_ver_" + VER + "_accuracy" + ".png")
LOSS_PATH = os.path.join("loss", COMPANY + "_lr_" + str(lr) + "_ver_" + VER + "_loss" + ".png")


def train():
    print("Training Start")
    print("company:", COMPANY)
    print("version:", VER)
    print("learning rate:", lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = StockPricePredictor(N_FC).to(device)
    if MODEL_PATH:
        model.load_state_dict(torch.load(MODEL_PATH))
    model.train()

    MSELoss = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

    dataset = StockDataset(COMPANY, START)
    training_dataset, valid_dataset = get_loaded_data(dataset=dataset, batch_size=BATCH_SIZE)

    loss_plot = []
    for e in range(epoch):
        print("epoch:", e + 1)
        for i, (input_data, label) in enumerate(training_dataset):
            optimizer.zero_grad()

            input_data = input_data.to(device).float()
            label = label.to(device).float()
            output = model(input_data)

            loss = MSELoss(output, label)
            loss.backward()
            loss_plot.append(loss.item())
            optimizer.step()
            print("training progress: %4d / %4d\r" % (i + 1, len(training_dataset)), end="")

        scheduler.step()
        print("\n")

    print("Evaluation Start")
    model.eval()
    predicted_plot = []
    true_plot = []
    for i, (input_data, label) in enumerate(valid_dataset):
        input_data = input_data.to(device).float()
        label = label.to(device).float()
        output = model(input_data)

        predicted_plot.append(output.item() * dataset.max)
        true_plot.append(label.item() * dataset.max)

    torch.save(model.state_dict(), SAVE_PATH)

    model_name = COMPANY + " ver " + VER
    plot_loss(loss_plot, model_name, LOSS_PATH)
    plot_accuracy(predicted_plot, true_plot, model_name, ACCURACY_PATH)


if __name__ == "__main__":
    train()
