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
portfolio = {'CNQ.TO': {'Score': 0.52084, 'Weight_Percent': 15.00001, 'Sector': 'Energy'}, 'RY.TO': {'Score': 0.71168, 'Weight_Percent': 12.57558, 'Sector': 'Financial Services'}, 'TD.TO': {'Score': 0.66958, 'Weight_Percent': 12.57558, 'Sector': 'Financial Services'}, 'SHOP.TO': {'Score': 0.24392, 'Weight_Percent': 7.70143, 'Sector': 'Technology'}, 'MSFT': {'Score': 0.63964, 'Weight_Percent': 7.47116, 'Sector': 'Technology'}, 'AAPL': {'Score': 0.55528, 'Weight_Percent': 6.48581, 'Sector': 'Technology'}, 'CVX': {'Score': 0.52529, 'Weight_Percent': 6.13553, 'Sector': 'Energy'}, 'GOOGL': {'Score': 0.49882, 'Weight_Percent': 5.82633, 'Sector': 'Communication Services'}, 'XOM': {'Score': 0.49851, 'Weight_Percent': 5.82272, 'Sector': 'Energy'}, 'AMZN': {'Score': 0.45262, 'Weight_Percent': 5.28671, 'Sector': 'Consumer Cyclical'}, 'NVDA': {'Score': 0.45121, 'Weight_Percent': 5.27025, 'Sector': 'Technology'}, 'JPM': {'Score': 0.65023, 'Weight_Percent': 5.10586, 'Sector': 'Financial Services'}, 'BAC': {'Score': 0.60402, 'Weight_Percent': 4.743, 'Sector': 'Financial Services'}}

# ============================================================
# 4. RUN TEST
# ============================================================
if __name__ == "__main__":
    test_portfolio_vs_benchmark(portfolio)
