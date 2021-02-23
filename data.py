import torch
import numpy as np
import pandas_datareader as datareader
import datetime as dt


def get_stock_close_price(company, start, end):
    start = dt.datetime(*start)
    end = dt.datetime(*end)
    data = datareader.DataReader(company, "yahoo", start, end)
    return data["Close"].values


def get_stock_open_price(company, start, end):
    start = dt.datetime(*start)
    end = dt.datetime(*end)
    data = datareader.DataReader(company, "yahoo", start, end)
    return data["Open"].values

