import yfinance as yf
import numpy as np
import pandas as pd


# ============================================================
# 1. BLENDED BENCHMARK (S&P500 + TSX Composite)
# ============================================================
def blended_benchmark(start, end):
    data = yf.download(["^GSPC", "^GSPTSE"], start=start, end=end)["Close"]

    # 50% / 50% blended price index
    blended_price = 0.5 * data["^GSPC"] + 0.5 * data["^GSPTSE"]
    blended_price.name = "Benchmark"

    return blended_price


# ============================================================
# 2. PORTFOLIO BACKTEST VS BENCHMARK
# ============================================================
def test_portfolio_vs_benchmark(final, start="2025-11-01", end="2025-11-15"):

    if not final:
        print("Portfolio is empty.")
        return

    # Normalize weights
    total_w = sum(v["Weight_Percent"] for v in final.values())
    weights = {t: v["Weight_Percent"] / total_w for t, v in final.items()}
    tickers = list(weights.keys())

    print("\nDownloading portfolio data...")
    price_data = yf.download(tickers, start=start, end=end)["Close"]

    # --- returns ---
    stock_rets = price_data.pct_change().dropna()

    # ============================================================
    # FIXED: Align weights EXACTLY with price_data.columns
    # Avoids incorrect multiplication & fake returns
    # ============================================================
    aligned_weights = np.array([weights[t] for t in stock_rets.columns])

    # Portfolio return
    port_ret = stock_rets.mul(aligned_weights).sum(axis=1)

    # ============================================================
    # Benchmark
    # ============================================================
    print("Downloading benchmark data...")
    bench_price = blended_benchmark(start, end)
    bench_ret = bench_price.pct_change().dropna()

    # Match date index to avoid timezone gaps
    port_ret = port_ret.loc[bench_ret.index]

    # Cumulative return
    port_cum = float((1 + port_ret).prod() - 1)
    bench_cum = float((1 + bench_ret).prod() - 1)

    # ============================================================
    # Print Results
    # ============================================================
    print("\n📈 PERFORMANCE TEST — Portfolio vs Blended Benchmark (S&P500 + TSX)")
    print(f"Portfolio Return: {port_cum * 100:.2f}%")
    print(f"Benchmark Return: {bench_cum * 100:.2f}%")
    print(f"Outperformance:   {(port_cum - bench_cum) * 100:.2f}%")

    return port_cum, bench_cum



# ============================================================
# 3. YOUR PORTFOLIO (paste anything here)
# ============================================================
portfolio = {'JPM': {'Score': 0.65023, 'Weight_Percent': 4.58158, 'Sector': 'Financial Services'}, 'PG': {'Score': 0.55431, 'Weight_Percent': 4.55533, 'Sector': 'Consumer Defensive'}, 'WMT': {'Score': 0.54301, 'Weight_Percent': 4.55533, 'Sector': 'Consumer Defensive'}, 'ABBV': {'Score': 0.50152, 'Weight_Percent': 4.55533, 'Sector': 'Healthcare'}, 'MSFT': {'Score': 0.63964, 'Weight_Percent': 4.50695, 'Sector': 'Technology'}, 'UNP': {'Score': 0.61701, 'Weight_Percent': 4.34751, 'Sector': 'Industrials'}, 'V': {'Score': 0.61623, 'Weight_Percent': 4.34201, 'Sector': 'Financial Services'}, 'HD': {'Score': 0.61157, 'Weight_Percent': 4.30918, 'Sector': 'Consumer Cyclical'}, 'MA': {'Score': 0.60979, 'Weight_Percent': 4.29662, 'Sector': 'Financial Services'}, 'CSCO': {'Score': 0.60228, 'Weight_Percent': 4.2437, 'Sector': 'Technology'}, 'HON': {'Score': 0.59252, 'Weight_Percent': 4.17495, 'Sector': 'Industrials'}, 'LOW': {'Score': 0.57909, 'Weight_Percent': 4.08032, 'Sector': 'Consumer Cyclical'}, 'AAPL': {'Score': 0.55528, 'Weight_Percent': 3.91255, 'Sector': 'Technology'}, 'RTX': {'Score': 0.52986, 'Weight_Percent': 3.73343, 'Sector': 'Industrials'}, 'MCD': {'Score': 0.5271, 'Weight_Percent': 3.71399, 'Sector': 'Consumer Cyclical'}, 'JNJ': {'Score': 0.52577, 'Weight_Percent': 3.70462, 'Sector': 'Healthcare'}, 'CVX': {'Score': 0.52529, 'Weight_Percent': 3.70124, 'Sector': 'Energy'}, 'COST': {'Score': 0.52296, 'Weight_Percent': 3.68482, 'Sector': 'Consumer Defensive'}, 'IBM': {'Score': 0.50727, 'Weight_Percent': 3.57427, 'Sector': 'Technology'}, 'ADBE': {'Score': 0.50457, 'Weight_Percent': 3.55524, 'Sector': 'Technology'}, 'ACN': {'Score': 0.50332, 'Weight_Percent': 3.54643, 'Sector': 'Technology'}, 'VZ': {'Score': 0.50125, 'Weight_Percent': 3.53184, 'Sector': 'Communication Services'}, 'ABT': {'Score': 0.49899, 'Weight_Percent': 3.51594, 'Sector': 'Healthcare'}, 'GOOGL': {'Score': 0.49882, 'Weight_Percent': 3.51474, 'Sector': 'Communication Services'}, 'PEP': {'Score': 0.49844, 'Weight_Percent': 3.51206, 'Sector': 'Consumer Defensive'}}

# ============================================================
# 4. RUN TEST
# ============================================================
if __name__ == "__main__":
    test_portfolio_vs_benchmark(portfolio)
