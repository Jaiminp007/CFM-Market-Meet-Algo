import yfinance as yf
import numpy as np
import pandas as pd
from test import blended_benchmark

def test_portfolio_historical(final, start="2024-09-01", end="2024-11-15"):
    """Test portfolio on REAL historical data"""
    if not final:
        print("Portfolio is empty.")
        return

    total_w = sum(v["Weight_Percent"] for v in final.values())
    weights = {t: v["Weight_Percent"] / total_w for t, v in final.items()}
    tickers = list(weights.keys())

    print(f"\n📊 Testing on HISTORICAL data ({start} → {end})...")

    # Download real price data
    price_data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    stock_rets = price_data.pct_change().dropna()

    aligned_weights = np.array([weights[t] for t in stock_rets.columns])
    port_ret = stock_rets.mul(aligned_weights).sum(axis=1)

    # Get benchmark
    bench_price = blended_benchmark(start, end)
    bench_ret = bench_price.pct_change().dropna()

    # Align dates
    common = port_ret.index.intersection(bench_ret.index)
    port_ret = port_ret.loc[common]
    bench_ret = bench_ret.loc[common]

    # Calculate cumulative returns
    port_cum = float((1 + port_ret).prod() - 1)
    bench_cum = float((1 + bench_ret).prod() - 1)

    print(f"Trading days: {len(common)} (from {common[0].date()} to {common[-1].date()})")
    print("\n📈 PERFORMANCE TEST — Portfolio vs Benchmark")
    print(f"Portfolio Return: {port_cum * 100:.2f}%")
    print(f"Benchmark Return: {bench_cum * 100:.2f}%")
    print(f"Outperformance:   {(port_cum - bench_cum) * 100:.2f}%")

    # Risk metrics
    port_vol = port_ret.std() * np.sqrt(252)
    bench_vol = bench_ret.std() * np.sqrt(252)
    sharpe = (port_cum * 252/len(common)) / port_vol if port_vol > 0 else 0

    print(f"\n📊 Risk Metrics:")
    print(f"Portfolio Volatility: {port_vol * 100:.2f}%")
    print(f"Benchmark Volatility: {bench_vol * 100:.2f}%")
    print(f"Sharpe Ratio (approx): {sharpe:.2f}")

    return port_cum, bench_cum

# Test with your portfolio
portfolio = {'RY.TO': {'Score': 0.77802, 'Weight_Percent': 14.41286, 'Sector': 'Financial Services'}, 'TD.TO': {'Score': 0.7508, 'Weight_Percent': 13.90865, 'Sector': 'Financial Services'}, 'BB.TO': {'Score': 0.66671, 'Weight_Percent': 13.65909, 'Sector': 'Technology'}, 'SHOP.TO': {'Score': 0.46297, 'Weight_Percent': 9.48495, 'Sector': 'Technology'}, 'T.TO': {'Score': 0.4496, 'Weight_Percent': 9.21107, 'Sector': 'Communication Services'}, 'UNH': {'Score': 0.72015, 'Weight_Percent': 2.50939, 'Sector': 'Healthcare'}, 'CAT': {'Score': 0.70086, 'Weight_Percent': 2.44216, 'Sector': 'Industrials'}, 'BLK': {'Score': 0.77492, 'Weight_Percent': 2.44162, 'Sector': 'Financial Services'}, 'BK': {'Score': 0.76005, 'Weight_Percent': 2.39477, 'Sector': 'Financial Services'}, 'AAPL': {'Score': 0.68015, 'Weight_Percent': 2.37002, 'Sector': 'Technology'}, 'LLY': {'Score': 0.67858, 'Weight_Percent': 2.36457, 'Sector': 'Healthcare'}, 'C': {'Score': 0.74226, 'Weight_Percent': 2.33872, 'Sector': 'Financial Services'}, 'LMT': {'Score': 0.6651, 'Weight_Percent': 2.31756, 'Sector': 'Industrials'}, 'BIIB': {'Score': 0.66363, 'Weight_Percent': 2.31244, 'Sector': 'Healthcare'}, 'QCOM': {'Score': 0.66155, 'Weight_Percent': 2.30517, 'Sector': 'Technology'}, 'BA': {'Score': 0.65729, 'Weight_Percent': 2.2903, 'Sector': 'Industrials'}, 'BAC': {'Score': 0.72672, 'Weight_Percent': 2.28976, 'Sector': 'Financial Services'}, 'TXN': {'Score': 0.64995, 'Weight_Percent': 2.26474, 'Sector': 'Technology'}, 'USB': {'Score': 0.70256, 'Weight_Percent': 2.21362, 'Sector': 'Financial Services'}, 'AMZN': {'Score': 0.62016, 'Weight_Percent': 2.16099, 'Sector': 'Consumer Cyclical'}, 'ACN': {'Score': 0.62004, 'Weight_Percent': 2.16056, 'Sector': 'Technology'}, 'UPS': {'Score': 0.61615, 'Weight_Percent': 2.14696, 'Sector': 'Industrials'}}

if __name__ == "__main__":
    # Test on historical out-of-sample period (after your training period)
    test_portfolio_historical(portfolio, start="2024-09-01", end="2024-11-15")
