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
    start = "2024-10-01"
    end = "2025-10-01"

    market = yf.Ticker("^GSPC")
    market_data = market.history(start=start, end=end, interval="1d")
    market_data["Market_Return"] = market_data["Close"].pct_change()


    for ticker in list:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end, interval="1d")

        if data.empty:
            invalid_tickers.append(ticker)
            continue

        avg_volume = data["Volume"].mean()


        if avg_volume < 5000:
            invalid_tickers.append(ticker)
            continue

        valid_tickers.append(ticker)

        # For Companies in S&P 500/TSX, add them to the final list
        market_of_ticker = stock.info.get("market")
        if market_of_ticker not in ["us_market", "ca_market"]:
            invalid_tickers.append(ticker)
            valid_tickers.remove(ticker)
            continue

    return valid_tickers, invalid_tickers

def main():
    tickers_list = read_csv("Tickers.csv")
    valid, invalid = check_ticker(tickers_list)

    print("Valid:", valid)
    print("Invalid:", invalid)

if __name__ == "__main__":
    main()

# calculating beta function
def compute_beta(ticker, benchmark="^GSPC", start="2025-01-01", end="2025-11-21"):
    # Download prices
    data = yf.download([ticker, benchmark], start=start, end=end)["Close"]
    
    # Compute daily returns
    rets = data.pct_change().dropna()
    
    # Extract series
    stock_ret = rets[ticker]
    bench_ret = rets[benchmark]
    
    # Compute covariance(stock, market) / variance(market)
    cov = np.cov(stock_ret, bench_ret)[0][1]
    var = np.var(bench_ret)
    
    beta = cov / var
    return beta

print(compute_beta("AAPL"))



