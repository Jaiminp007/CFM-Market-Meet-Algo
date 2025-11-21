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
# MARKET SIMULATION (for future dates)
# ============================================================
def simulate_future_market(tickers, start_date, num_days=5, seed=42):
    """
    Simulate future market returns for testing purposes

    Args:
        tickers: List of stock tickers
        start_date: Starting date for simulation
        num_days: Number of trading days to simulate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with simulated price data
    """
    np.random.seed(seed)

    # Generate realistic daily returns (mean ~0%, std ~1-2%)
    # Market typically has slight positive drift
    market_drift = 0.0005  # ~0.05% daily
    market_vol = 0.015     # ~1.5% daily volatility

    # Create date range (business days only)
    dates = pd.bdate_range(start=start_date, periods=num_days)

    # Simulate market returns with some correlation
    market_returns = np.random.normal(market_drift, market_vol, num_days)

    # Initialize price data dictionary
    price_data = {}

    for ticker in tickers:
        # Each stock has correlation with market + idiosyncratic risk
        stock_beta = np.random.uniform(0.7, 1.3)  # Beta between 0.7-1.3
        idio_vol = np.random.uniform(0.01, 0.025)  # Stock-specific volatility

        # Stock returns = beta * market + idiosyncratic
        stock_returns = (stock_beta * market_returns +
                        np.random.normal(0, idio_vol, num_days))

        # Get last known price (try to fetch real data)
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                last_price = hist['Close'].iloc[-1]
            else:
                last_price = 100.0  # Default if no data
        except:
            last_price = 100.0

        # Generate price series
        prices = [last_price]
        for ret in stock_returns:
            prices.append(prices[-1] * (1 + ret))

        price_data[ticker] = prices[1:]  # Skip initial price

    # Create DataFrame
    df = pd.DataFrame(price_data, index=dates)
    return df

def simulate_benchmark_future(start_date, num_days=5, seed=42):
    """Simulate future benchmark returns"""
    np.random.seed(seed + 1)  # Different seed for independence

    dates = pd.bdate_range(start=start_date, periods=num_days)

    # S&P 500 simulation
    sp_drift = 0.0004
    sp_vol = 0.012
    sp_returns = np.random.normal(sp_drift, sp_vol, num_days)

    # TSX simulation (slightly different characteristics)
    tsx_drift = 0.0003
    tsx_vol = 0.011
    tsx_returns = np.random.normal(tsx_drift, tsx_vol, num_days)

    # Get last known benchmark prices
    try:
        sp_hist = yf.Ticker("^GSPC").history(period="5d")
        tsx_hist = yf.Ticker("^GSPTSE").history(period="5d")

        sp_last = sp_hist['Close'].iloc[-1] if not sp_hist.empty else 5000.0
        tsx_last = tsx_hist['Close'].iloc[-1] if not tsx_hist.empty else 22000.0
    except:
        sp_last = 5000.0
        tsx_last = 22000.0

    # Generate price series
    sp_prices = [sp_last]
    tsx_prices = [tsx_last]

    for sp_ret, tsx_ret in zip(sp_returns, tsx_returns):
        sp_prices.append(sp_prices[-1] * (1 + sp_ret))
        tsx_prices.append(tsx_prices[-1] * (1 + tsx_ret))

    # Blended benchmark (50/50)
    blended = 0.5 * np.array(sp_prices[1:]) + 0.5 * np.array(tsx_prices[1:])

    return pd.Series(blended, index=dates, name="Benchmark")


# ============================================================
# 2. PORTFOLIO BACKTEST VS BENCHMARK
# ============================================================
# ...existing code...
def test_portfolio_vs_benchmark(final, start="2025-11-05", end="2025-11-15", simulate=False):
    if not final:
        print("Portfolio is empty.")
        return

    total_w = sum(v["Weight_Percent"] for v in final.values())
    weights = {t: v["Weight_Percent"] / total_w for t, v in final.items()}
    tickers = list(weights.keys())

    if simulate:
        print(f"\n🔮 SIMULATING future market ({start} → {end})...")
        print("Note: Using simulated data for future dates\n")

        # Simulate 5 trading days
        price_data = simulate_future_market(tickers, start, num_days=5)
        stock_rets = price_data.pct_change().dropna()

        aligned_weights = np.array([weights[t] for t in stock_rets.columns])
        port_ret = stock_rets.mul(aligned_weights).sum(axis=1)

        # Simulate benchmark
        bench_price = simulate_benchmark_future(start, num_days=5)
        bench_ret = bench_price.pct_change().dropna()

    else:
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

    print(f"Common trading days: {len(common)} (first {common[0].date()} / last {common[-1].date()})")
    print("\n📈 PERFORMANCE TEST — Portfolio vs Blended Benchmark (S&P500 + TSX)")
    print(f"Portfolio Return: {port_cum * 100:.2f}%")
    print(f"Benchmark Return: {bench_cum * 100:.2f}%")
    print(f"Outperformance:   {(port_cum - bench_cum) * 100:.2f}%")
    return port_cum, bench_cum
# ...existing code...

# ============================================================
# 3. YOUR PORTFOLIO (paste anything here)
# ============================================================
portfolio = {'JPM': {'Score': 0.72595, 'Weight_Percent': 4.62059, 'Sector': 'Financial Services'}, 'V': {'Score': 0.72034, 'Weight_Percent': 4.58488, 'Sector': 'Financial Services'}, 'MA': {'Score': 0.70519, 'Weight_Percent': 4.48847, 'Sector': 'Financial Services'}, 'HD': {'Score': 0.68797, 'Weight_Percent': 4.37885, 'Sector': 'Consumer Cyclical'}, 'AMZN': {'Score': 0.68772, 'Weight_Percent': 4.37725, 'Sector': 'Consumer Cyclical'}, 'LOW': {'Score': 0.68247, 'Weight_Percent': 4.34384, 'Sector': 'Consumer Cyclical'}, 'HON': {'Score': 0.68066, 'Weight_Percent': 4.33232, 'Sector': 'Industrials'}, 'UNP': {'Score': 0.67014, 'Weight_Percent': 4.26537, 'Sector': 'Industrials'}, 'META': {'Score': 0.6541, 'Weight_Percent': 4.16327, 'Sector': 'Communication Services'}, 'GOOGL': {'Score': 0.64862, 'Weight_Percent': 4.12838, 'Sector': 'Communication Services'}, 'NKE': {'Score': 0.64638, 'Weight_Percent': 4.11414, 'Sector': 'Consumer Cyclical'}, 'TMO': {'Score': 0.64596, 'Weight_Percent': 4.11147, 'Sector': 'Healthcare'}, 'DHR': {'Score': 0.63597, 'Weight_Percent': 4.04788, 'Sector': 'Healthcare'}, 'BA': {'Score': 0.63523, 'Weight_Percent': 4.04316, 'Sector': 'Industrials'}, 'CSCO': {'Score': 0.72002, 'Weight_Percent': 3.93179, 'Sector': 'Technology'}, 'QCOM': {'Score': 0.69012, 'Weight_Percent': 3.76854, 'Sector': 'Technology'}, 'AAPL': {'Score': 0.67434, 'Weight_Percent': 3.68238, 'Sector': 'Technology'}, 'IBM': {'Score': 0.67103, 'Weight_Percent': 3.66428, 'Sector': 'Technology'}, 'MSFT': {'Score': 0.67015, 'Weight_Percent': 3.65948, 'Sector': 'Technology'}, 'CRM': {'Score': 0.66523, 'Weight_Percent': 3.6326, 'Sector': 'Technology'}, 'ADBE': {'Score': 0.65714, 'Weight_Percent': 3.58842, 'Sector': 'Technology'}, 'ACN': {'Score': 0.65377, 'Weight_Percent': 3.57003, 'Sector': 'Technology'}, 'TXN': {'Score': 0.64996, 'Weight_Percent': 3.54922, 'Sector': 'Technology'}, 'NVDA': {'Score': 0.63689, 'Weight_Percent': 3.47785, 'Sector': 'Technology'}, 'AMD': {'Score': 0.63645, 'Weight_Percent': 3.47544, 'Sector': 'Technology'}}# 4. RUN TEST
# ============================================================
if __name__ == "__main__":
    # Test with simulated future market (Nov 21-28, 2025)
    # Use simulate=True for future dates, simulate=False for historical data
    test_portfolio_vs_benchmark(portfolio, simulate=False)
