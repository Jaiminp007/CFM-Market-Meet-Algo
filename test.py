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
portfolio = {'HMC': {'Score': 0.7577, 'Weight_Percent': 4.90282, 'Sector': 'Consumer Cyclical'}, 'NEM': {'Score': 0.74309, 'Weight_Percent': 4.80828, 'Sector': 'Basic Materials'}, 'FOXA': {'Score': 0.73412, 'Weight_Percent': 4.75025, 'Sector': 'Communication Services'}, 'L': {'Score': 0.72584, 'Weight_Percent': 4.69665, 'Sector': 'Financial Services'}, 'AMCR': {'Score': 0.71609, 'Weight_Percent': 4.63357, 'Sector': 'Consumer Cyclical'}, 'PLUG': {'Score': 0.65319, 'Weight_Percent': 4.22657, 'Sector': 'Industrials'}, 'CYBN': {'Score': 0.63143, 'Weight_Percent': 4.08576, 'Sector': 'Healthcare'}, 'ADIL': {'Score': 0.63043, 'Weight_Percent': 4.07929, 'Sector': 'Healthcare'}, 'EVRG': {'Score': 0.62611, 'Weight_Percent': 4.05135, 'Sector': 'Utilities'}, 'IMPP': {'Score': 0.62423, 'Weight_Percent': 4.03918, 'Sector': 'Energy'}, 'SNTI': {'Score': 0.62393, 'Weight_Percent': 4.03724, 'Sector': 'Healthcare'}, 'SIDU': {'Score': 0.61366, 'Weight_Percent': 3.97078, 'Sector': 'Industrials'}, 'MDIA': {'Score': 0.61343, 'Weight_Percent': 3.9693, 'Sector': 'Communication Services'}, 'AGRI': {'Score': 0.60903, 'Weight_Percent': 3.94082, 'Sector': 'Financial Services'}, 'MNMD': {'Score': 0.58885, 'Weight_Percent': 3.81024, 'Sector': 'Healthcare'}, 'DGICA': {'Score': 0.57943, 'Weight_Percent': 3.7493, 'Sector': 'Financial Services'}, 'KHC': {'Score': 0.57811, 'Weight_Percent': 3.74075, 'Sector': 'Consumer Defensive'}, 'SELF': {'Score': 0.57783, 'Weight_Percent': 3.73894, 'Sector': 'Real Estate'}, 'OPXS': {'Score': 0.5755, 'Weight_Percent': 3.72387, 'Sector': 'Industrials'}, 'BMY': {'Score': 0.56841, 'Weight_Percent': 3.67798, 'Sector': 'Healthcare'}, 'BLTE': {'Score': 0.55316, 'Weight_Percent': 3.5793, 'Sector': 'Healthcare'}, 'SPCE': {'Score': 0.54466, 'Weight_Percent': 3.52432, 'Sector': 'Industrials'}, 'TTOO': {'Score': 0.53155, 'Weight_Percent': 3.43947, 'Sector': 'Healthcare'}, 'VZ': {'Score': 0.52852, 'Weight_Percent': 3.41988, 'Sector': 'Communication Services'}, 'KGS': {'Score': 0.52608, 'Weight_Percent': 3.40408, 'Sector': 'Energy'}}
# ============================================================
# 4. RUN TEST
# ============================================================
if __name__ == "__main__":
    # Test with simulated future market (Nov 21-28, 2025)
    # Use simulate=True for future dates, simulate=False for historical data
    test_portfolio_vs_benchmark(portfolio, simulate=True)
