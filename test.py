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
def test_portfolio_vs_benchmark(final, start="2025-11-21", end="2025-11-28", simulate=True):
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
portfolio = {'RY.TO': {'Score': 0.83732, 'Weight_Percent': 7.61908, 'Sector': 'Financial Services'}, 'CNQ.TO': {'Score': 0.77947, 'Weight_Percent': 7.09267, 'Sector': 'Energy'}, 'MG.TO': {'Score': 0.7549, 'Weight_Percent': 6.8691, 'Sector': 'Consumer Cyclical'}, 'SU.TO': {'Score': 0.75412, 'Weight_Percent': 6.86199, 'Sector': 'Energy'}, 'AEM.TO': {'Score': 0.74698, 'Weight_Percent': 6.79703, 'Sector': 'Basic Materials'}, 'CNR.TO': {'Score': 0.73978, 'Weight_Percent': 6.73153, 'Sector': 'Industrials'}, 'BMO.TO': {'Score': 0.73561, 'Weight_Percent': 6.69357, 'Sector': 'Financial Services'}, 'TRP.TO': {'Score': 0.7348, 'Weight_Percent': 6.68621, 'Sector': 'Energy'}, 'ENB.TO': {'Score': 0.70873, 'Weight_Percent': 6.44899, 'Sector': 'Energy'}, 'BNS.TO': {'Score': 0.68291, 'Weight_Percent': 6.21404, 'Sector': 'Financial Services'}, 'TD.TO': {'Score': 0.68228, 'Weight_Percent': 6.20832, 'Sector': 'Financial Services'}, 'NTR.TO': {'Score': 0.66159, 'Weight_Percent': 6.02004, 'Sector': 'Basic Materials'}, 'WCN.TO': {'Score': 0.62513, 'Weight_Percent': 5.68828, 'Sector': 'Industrials'}, 'CCO.TO': {'Score': 0.60369, 'Weight_Percent': 5.49319, 'Sector': 'Energy'}, 'BCE.TO': {'Score': 0.4891, 'Weight_Percent': 4.45049, 'Sector': 'Communication Services'}, 'SHOP.TO': {'Score': 0.45338, 'Weight_Percent': 4.12547, 'Sector': 'Technology'}}# ============================================================
# 4. RUN TEST
# ============================================================
if __name__ == "__main__":
    # Test with simulated future market (Nov 21-28, 2025)
    # Use simulate=True for future dates, simulate=False for historical data
    test_portfolio_vs_benchmark(portfolio, simulate=True)
