import numpy as np
import yfinance as yf
import pandas as pd

def read_csv(filename):
    data=pd.read_csv(filename, header=None)
    l= data.values.tolist()
    list=[]
    for i in l:
        list.append(i[0])
    return list


def check_ticker(list):
    valid_tickers=[]
    invalid_tickers=[]

    for ticker in list:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")

        if data.empty:
            invalid_tickers.append(ticker)
        else:
            valid_tickers.append(ticker)
    return valid_tickers, invalid_tickers


def main():
    list=read_csv("Tickers.csv")
    valid, invalid = check_ticker(list)
    for i in valid:
        stock = yf.Ticker(i)
        start = "2024-10-01"
        end = "2025-10-01"
        data = stock.history(start=start, end=end, interval="1d")
        avg_volume = data["Volume"].mean()
        print(avg_volume)
        if avg_volume < 5000:
            invalid.append(i)
            valid.remove(i)

    print("Valid:", valid)
    print("Invalid:", invalid)

if __name__ == "__main__":
    main()