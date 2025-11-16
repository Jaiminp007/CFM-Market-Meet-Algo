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
portfolio = {
    'T.TO': {'Score': 0.56972, 'Weight_Percent': 10.00001, 'Sector': 'Communication Services'},
    'BB.TO': {'Score': 0.34352, 'Weight_Percent': 10.00001, 'Sector': 'Technology'},
    'RY.TO': {'Score': 0.71168, 'Weight_Percent': 9.15758, 'Sector': 'Financial Services'},
    'TD.TO': {'Score': 0.66958, 'Weight_Percent': 9.15758, 'Sector': 'Financial Services'},
    'SHOP.TO': {'Score': 0.24392, 'Weight_Percent': 8.47683, 'Sector': 'Technology'},
    'UNP': {'Score': 0.61701, 'Weight_Percent': 3.40706, 'Sector': 'Industrials'},
    'AAPL': {'Score': 0.55528, 'Weight_Percent': 3.06618, 'Sector': 'Technology'},
    'ACN': {'Score': 0.50332, 'Weight_Percent': 2.77927, 'Sector': 'Technology'},
    'ABT': {'Score': 0.49899, 'Weight_Percent': 2.75537, 'Sector': 'Healthcare'},
    'PEP': {'Score': 0.49844, 'Weight_Percent': 2.7523,  'Sector': 'Consumer Defensive'},
    'BK': {'Score': 0.6673,  'Weight_Percent': 2.74063, 'Sector': 'Financial Services'},
    'LMT': {'Score': 0.49517, 'Weight_Percent': 2.73426, 'Sector': 'Industrials'},
    'PG': {'Score': 0.55431, 'Weight_Percent': 2.72625, 'Sector': 'Consumer Defensive'},
    'CL': {'Score': 0.53219, 'Weight_Percent': 2.72625, 'Sector': 'Consumer Defensive'},
    'ABBV': {'Score': 0.50152, 'Weight_Percent': 2.72625, 'Sector': 'Healthcare'},
    'KO': {'Score': 0.4934,  'Weight_Percent': 2.72452, 'Sector': 'Consumer Defensive'},
    'BA': {'Score': 0.49328, 'Weight_Percent': 2.72383, 'Sector': 'Industrials'},
    'PFE': {'Score': 0.49158, 'Weight_Percent': 2.71444, 'Sector': 'Healthcare'},
    'BMY': {'Score': 0.48663, 'Weight_Percent': 2.68713, 'Sector': 'Healthcare'},
    'BAC': {'Score': 0.60402, 'Weight_Percent': 2.48071, 'Sector': 'Financial Services'},
    'USB': {'Score': 0.58776, 'Weight_Percent': 2.41394, 'Sector': 'Financial Services'},
    'BLK': {'Score': 0.56864, 'Weight_Percent': 2.33541, 'Sector': 'Financial Services'},
    'AIG': {'Score': 0.55707, 'Weight_Percent': 2.2879,  'Sector': 'Financial Services'},
    'C':   {'Score': 0.53974, 'Weight_Percent': 2.21669, 'Sector': 'Financial Services'},
    'AXP': {'Score': 0.53801, 'Weight_Percent': 2.20961, 'Sector': 'Financial Services'},
    'GTII': {'Score': 0.5, 'Weight_Percent': 0.0, 'Sector': 'Industrials'}
}


# ============================================================
# 4. RUN TEST
# ============================================================
if __name__ == "__main__":
    test_portfolio_vs_benchmark(portfolio)
