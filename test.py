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
# ...existing code...
def test_portfolio_vs_benchmark(final, start="2025-11-01", end="2025-11-15"):
    if not final:
        print("Portfolio is empty.")
        return

    total_w = sum(v["Weight_Percent"] for v in final.values())
    weights = {t: v["Weight_Percent"] / total_w for t, v in final.items()}
    tickers = list(weights.keys())

    print(f"\nDownloading portfolio data ({start} → {end})...")
    price_data = yf.download(tickers, start=start, end=end)["Close"]
    stock_rets = price_data.pct_change().dropna()

    aligned_weights = np.array([weights[t] for t in stock_rets.columns])
    port_ret = stock_rets.mul(aligned_weights).sum(axis=1)

    print("Downloading benchmark data...")
    bench_price = blended_benchmark(start, end)
    bench_ret = bench_price.pct_change().dropna()

    # Strict intersection
    common = port_ret.index.intersection(bench_ret.index)
    port_ret = port_ret.loc[common]
    bench_ret = bench_ret.loc[common]

    assert len(port_ret) == len(bench_ret), "Date alignment failed"

    port_cum = float((1 + port_ret).prod() - 1)
    bench_cum = float((1 + bench_ret).prod() - 1)

    print(f"\nCommon trading days: {len(common)} (first {common[0].date()} / last {common[-1].date()})")
    print("\n📈 PERFORMANCE TEST — Portfolio vs Blended Benchmark (S&P500 + TSX)")
    print(f"Portfolio Return: {port_cum * 100:.2f}%")
    print(f"Benchmark Return: {bench_cum * 100:.2f}%")
    print(f"Outperformance:   {(port_cum - bench_cum) * 100:.2f}%")
    return port_cum, bench_cum
# ...existing code...

# ============================================================
# 3. YOUR PORTFOLIO (paste anything here)
# ============================================================
portfolio = {'WMT': {'Score': 0.57143, 'Weight_Percent': 6.14759, 'Sector': 'Consumer Defensive'}, 'PG': {'Score': 0.56812, 'Weight_Percent': 6.14759, 'Sector': 'Consumer Defensive'}, 'V': {'Score': 0.68833, 'Weight_Percent': 3.98038, 'Sector': 'Financial Services'}, 'AAPL': {'Score': 0.68776, 'Weight_Percent': 3.9771, 'Sector': 'Technology'}, 'AMZN': {'Score': 0.68466, 'Weight_Percent': 3.95917, 'Sector': 'Consumer Cyclical'}, 'HON': {'Score': 0.68063, 'Weight_Percent': 3.93587, 'Sector': 'Industrials'}, 'JPM': {'Score': 0.6782, 'Weight_Percent': 3.9218, 'Sector': 'Financial Services'}, 'CSCO': {'Score': 0.67344, 'Weight_Percent': 3.8943, 'Sector': 'Technology'}, 'LOW': {'Score': 0.66824, 'Weight_Percent': 3.86424, 'Sector': 'Consumer Cyclical'}, 'QCOM': {'Score': 0.66739, 'Weight_Percent': 3.85932, 'Sector': 'Technology'}, 'IBM': {'Score': 0.66464, 'Weight_Percent': 3.8434, 'Sector': 'Technology'}, 'GOOGL': {'Score': 0.66255, 'Weight_Percent': 3.83133, 'Sector': 'Communication Services'}, 'NKE': {'Score': 0.66204, 'Weight_Percent': 3.82839, 'Sector': 'Consumer Cyclical'}, 'UPS': {'Score': 0.65787, 'Weight_Percent': 3.80427, 'Sector': 'Industrials'}, 'HD': {'Score': 0.65728, 'Weight_Percent': 3.80086, 'Sector': 'Consumer Cyclical'}, 'AVGO': {'Score': 0.65603, 'Weight_Percent': 3.79364, 'Sector': 'Technology'}, 'ADBE': {'Score': 0.65477, 'Weight_Percent': 3.78636, 'Sector': 'Technology'}, 'UNH': {'Score': 0.65332, 'Weight_Percent': 3.77795, 'Sector': 'Healthcare'}, 'ACN': {'Score': 0.6485, 'Weight_Percent': 3.75007, 'Sector': 'Technology'}, 'MA': {'Score': 0.6449, 'Weight_Percent': 3.72926, 'Sector': 'Financial Services'}, 'UNP': {'Score': 0.63792, 'Weight_Percent': 3.68891, 'Sector': 'Industrials'}, 'INTC': {'Score': 0.63631, 'Weight_Percent': 3.67958, 'Sector': 'Technology'}, 'BA': {'Score': 0.63026, 'Weight_Percent': 3.64458, 'Sector': 'Industrials'}, 'DHR': {'Score': 0.61477, 'Weight_Percent': 3.55503, 'Sector': 'Healthcare'}, 'LLY': {'Score': 0.61373, 'Weight_Percent': 3.54901, 'Sector': 'Healthcare'}}

# ============================================================
# 4. RUN TEST
# ============================================================
if __name__ == "__main__":
    test_portfolio_vs_benchmark(portfolio)
