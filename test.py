import torch
import torch.optim as optim
import numpy as np
import pandas_datareader as datareader
import datetime as dt
import matplotlib.pyplot as plt
import random
from random import shuffle
from model import StockPricePredictor

x = [random.random() for i in range(10)]
print(x)
shuffle(x)
print(x)