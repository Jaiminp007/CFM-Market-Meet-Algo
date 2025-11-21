"""
STANDALONE BACKTEST SCRIPT
--------------------------
This script can test any portfolio without importing your main code.
Just paste your final_portfolio dictionary and run!

Usage:
1. Run your main portfolio code to generate final_portfolio
2. Copy the final_portfolio dictionary 
3. Paste it into the PORTFOLIO_TO_TEST variable below
4. Run this script
"""

import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# PASTE YOUR PORTFOLIO HERE
# ============================================================================
# After running your main code, copy the final_portfolio dictionary
# It should look like:
# {
#     'AAPL': {'Score': 0.85, 'Weight_Percent': 5.2, 'Sector': 'Technology'},
#     'MSFT': {'Score': 0.82, 'Weight_Percent': 4.8, 'Sector': 'Technology'},
#     ...
# }

PORTFOLIO_TO_TEST = {'JPM': {'Score': 0.73261, 'Weight_Percent': 4.33069, 'Sector': 'Financial Services'}, 'V': {'Score': 0.72721, 'Weight_Percent': 4.29877, 'Sector': 'Financial Services'}, 'CSCO': {'Score': 0.7269, 'Weight_Percent': 4.29693, 'Sector': 'Technology'}, 'MA': {'Score': 0.71228, 'Weight_Percent': 4.21052, 'Sector': 'Financial Services'}, 'HD': {'Score': 0.69511, 'Weight_Percent': 4.10901, 'Sector': 'Consumer Cyclical'}, 'AMZN': {'Score': 0.69168, 'Weight_Percent': 4.08874, 'Sector': 'Consumer Cyclical'}, 'QCOM': {'Score': 0.69137, 'Weight_Percent': 4.0869, 'Sector': 'Technology'}, 'LOW': {'Score': 0.68988, 'Weight_Percent': 4.07811, 'Sector': 'Consumer Cyclical'}, 'HON': {'Score': 0.68778, 'Weight_Percent': 4.06569, 'Sector': 'Industrials'}, 'AAPL': {'Score': 0.68147, 'Weight_Percent': 4.02838, 'Sector': 'Technology'}, 'IBM': {'Score': 0.67856, 'Weight_Percent': 4.01118, 'Sector': 'Technology'}, 'UNP': {'Score': 0.67663, 'Weight_Percent': 3.99977, 'Sector': 'Industrials'}, 'MSFT': {'Score': 0.67625, 'Weight_Percent': 3.99753, 'Sector': 'Technology'}, 'CRM': {'Score': 0.67272, 'Weight_Percent': 3.97667, 'Sector': 'Technology'}, 'ADBE': {'Score': 0.66487, 'Weight_Percent': 3.93025, 'Sector': 'Technology'}, 'META': {'Score': 0.66171, 'Weight_Percent': 3.91159, 'Sector': 'Communication Services'}, 'ACN': {'Score': 0.66144, 'Weight_Percent': 3.90999, 'Sector': 'Technology'}, 'TXN': {'Score': 0.65749, 'Weight_Percent': 3.88663, 'Sector': 'Technology'}, 'GOOGL': {'Score': 0.65628, 'Weight_Percent': 3.87949, 'Sector': 'Communication Services'}, 'TMO': {'Score': 0.65374, 'Weight_Percent': 3.86446, 'Sector': 'Healthcare'}, 'NKE': {'Score': 0.65263, 'Weight_Percent': 3.8579, 'Sector': 'Consumer Cyclical'}, 'DHR': {'Score': 0.64374, 'Weight_Percent': 3.80535, 'Sector': 'Healthcare'}, 'BA': {'Score': 0.64288, 'Weight_Percent': 3.80027, 'Sector': 'Industrials'}, 'NFLX': {'Score': 0.64207, 'Weight_Percent': 3.79548, 'Sector': 'Communication Services'}, 'AMD': {'Score': 0.6394, 'Weight_Percent': 3.7797, 'Sector': 'Technology'}}

# ============================================================================
# BACKTEST FUNCTIONS
# ============================================================================

def blended_benchmark_returns(start, end):
    """Calculate benchmark as simple average of TSX and S&P 500 returns - OPTIMIZED"""
    try:
        # Download both at once
        data = yf.download(["^GSPC", "^GSPTSE"], start=start, end=end, progress=False, auto_adjust=True)
        
        if data.empty or len(data) < 2:
            return None
        
        # Extract close prices - handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data['Close']
        else:
            close_data = data
        
        # Check we have both indices
        if '^GSPC' not in close_data.columns or '^GSPTSE' not in close_data.columns:
            return None
        
        # Calculate returns
        sp500_returns = close_data['^GSPC'].pct_change()
        tsx_returns = close_data['^GSPTSE'].pct_change()
        
        # Blend
        benchmark_returns = (sp500_returns + tsx_returns) / 2
        return benchmark_returns.dropna()
    except Exception as e:
        # Suppress errors during batch processing
        return None

def calculate_portfolio_returns(portfolio_dict, start, end):
    """Calculate weighted portfolio returns - OPTIMIZED"""
    try:
        # Download ALL tickers at once (much faster!)
        tickers = list(portfolio_dict.keys())
        all_data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        
        # Check if we got any data
        if all_data.empty or len(all_data) < 2:
            return None
        
        # Extract Close prices - handle different data structures
        if len(tickers) == 1:
            # Single ticker returns simpler structure
            if 'Close' in all_data.columns:
                close_prices = all_data[['Close']].copy()
                close_prices.columns = [tickers[0]]
            else:
                close_prices = all_data.to_frame(name=tickers[0])
        else:
            # Multiple tickers
            if isinstance(all_data.columns, pd.MultiIndex):
                close_prices = all_data['Close']
            else:
                close_prices = all_data
        
        # Ensure we have a DataFrame
        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame()
        
        if close_prices.empty or len(close_prices) < 2:
            return None
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        if returns.empty:
            return None
        
        # Weight the returns
        weighted_returns = pd.Series(0.0, index=returns.index)
        for ticker in tickers:
            if ticker in returns.columns:
                weight = portfolio_dict[ticker]['Weight_Percent'] / 100.0
                weighted_returns += returns[ticker] * weight
        
        return weighted_returns
    except Exception as e:
        # Suppress errors during batch processing
        return None

def run_single_backtest(portfolio_dict, start_date, end_date):
    """Run backtest for a single period"""
    portfolio_returns = calculate_portfolio_returns(portfolio_dict, start_date, end_date)
    benchmark_returns = blended_benchmark_returns(start_date, end_date)
    
    if portfolio_returns is None or benchmark_returns is None:
        return None
    
    combined = pd.DataFrame({
        'Portfolio': portfolio_returns,
        'Benchmark': benchmark_returns
    }).dropna()
    
    if len(combined) < 5:
        return None
    
    portfolio_cumret = (1 + combined['Portfolio']).cumprod() - 1
    benchmark_cumret = (1 + combined['Benchmark']).cumprod() - 1
    
    portfolio_total_return = portfolio_cumret.iloc[-1]
    benchmark_total_return = benchmark_cumret.iloc[-1]
    
    tracking_diff = portfolio_total_return - benchmark_total_return
    tracking_error_abs = abs(tracking_diff)
    
    daily_diff = combined['Portfolio'] - combined['Benchmark']
    tracking_error_vol = daily_diff.std() * np.sqrt(252)
    
    correlation = combined['Portfolio'].corr(combined['Benchmark'])
    
    covariance = combined['Portfolio'].cov(combined['Benchmark'])
    benchmark_var = combined['Benchmark'].var()
    beta = covariance / benchmark_var if benchmark_var != 0 else np.nan
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'portfolio_return': portfolio_total_return * 100,
        'benchmark_return': benchmark_total_return * 100,
        'tracking_diff': tracking_diff * 100,
        'tracking_error_abs': tracking_error_abs * 100,
        'tracking_error_vol': tracking_error_vol * 100,
        'correlation': correlation,
        'beta': beta,
        'num_days': len(combined)
    }

def generate_random_periods(start_date, end_date, num_periods=50):
    """Generate random 10-day periods (more reliable than 5-day)"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    periods = []
    max_attempts = num_periods * 5
    attempts = 0
    
    while len(periods) < num_periods and attempts < max_attempts:
        attempts += 1
        days_between = (end - start).days
        if days_between < 20:
            break
        
        random_days = random.randint(0, days_between - 25)
        period_start = start + timedelta(days=random_days)
        
        try:
            temp_data = yf.download("^GSPC", 
                                   start=period_start.strftime("%Y-%m-%d"),
                                   end=(period_start + timedelta(days=30)).strftime("%Y-%m-%d"),
                                   progress=False,
                                   auto_adjust=True)
            
            if len(temp_data) >= 10:
                trading_days = temp_data.index[:10]
                period_start_str = trading_days[0].strftime("%Y-%m-%d")
                period_end_str = (trading_days[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
                periods.append((period_start_str, period_end_str))
        except:
            continue
    
    return periods
    
    return periods

def run_backtest(portfolio_dict, num_periods=50):
    """Run comprehensive backtest"""
    
    if not portfolio_dict:
        print("ERROR: No portfolio provided!")
        print("Please paste your portfolio dictionary into PORTFOLIO_TO_TEST variable")
        return None
    
    print("=" * 80)
    print("PORTFOLIO BACKTEST - MARKET MEET STRATEGY")
    print("=" * 80)
    print(f"\nPortfolio: {len(portfolio_dict)} stocks")
    print(f"Test periods: {num_periods} random 10-day periods")
    print(f"Lookback: 12 months")
    
    # Validate portfolio
    total_weight = sum(d['Weight_Percent'] for d in portfolio_dict.values())
    print(f"Total weight: {total_weight:.2f}%")
    
    if abs(total_weight - 100) > 0.1:
        print(f"⚠️  WARNING: Weights don't sum to 100%!")
    
    print("\n" + "=" * 80)
    
    # Generate test periods
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    periods = generate_random_periods(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        num_periods
    )
    
    print(f"\nGenerated {len(periods)} test periods\n")
    
    # Run backtests
    results = []
    for i, (start, end) in enumerate(periods):
        print(f"Testing period {i+1}/{len(periods)}: {start} to {end}", end="\r")
        result = run_single_backtest(portfolio_dict, start, end)
        if result is not None:
            results.append(result)
    
    print("\n" + "=" * 80)
    
    if not results:
        print("ERROR: No valid results")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nSuccessful backtests: {len(results_df)}")
    print(f"\nAverage Returns:")
    print(f"  Portfolio: {results_df['portfolio_return'].mean():.4f}%")
    print(f"  Benchmark: {results_df['benchmark_return'].mean():.4f}%")
    
    print(f"\n🎯 MARKET MEET PERFORMANCE (closer to 0 = better):")
    print(f"  Mean Tracking Difference: {results_df['tracking_diff'].mean():.4f}%")
    print(f"  Median Tracking Difference: {results_df['tracking_diff'].median():.4f}%")
    print(f"  Std Dev: {results_df['tracking_diff'].std():.4f}%")
    print(f"  Mean Absolute Error: {results_df['tracking_error_abs'].mean():.4f}%")
    
    print(f"\nCorrelation with Benchmark:")
    print(f"  Mean: {results_df['correlation'].mean():.4f}")
    print(f"  Median: {results_df['correlation'].median():.4f}")
    print(f"  Range: [{results_df['correlation'].min():.4f}, {results_df['correlation'].max():.4f}]")
    
    print(f"\nPortfolio Beta:")
    print(f"  Mean: {results_df['beta'].mean():.4f}")
    print(f"  Median: {results_df['beta'].median():.4f}")
    
    # Accuracy metrics
    within_1pct = (results_df['tracking_error_abs'] <= 1.0).sum()
    within_2pct = (results_df['tracking_error_abs'] <= 2.0).sum()
    within_3pct = (results_df['tracking_error_abs'] <= 3.0).sum()
    
    print(f"\n📊 Tracking Accuracy:")
    print(f"  Within ±1.0%: {within_1pct}/{len(results_df)} ({100*within_1pct/len(results_df):.1f}%)")
    print(f"  Within ±2.0%: {within_2pct}/{len(results_df)} ({100*within_2pct/len(results_df):.1f}%)")
    print(f"  Within ±3.0%: {within_3pct}/{len(results_df)} ({100*within_3pct/len(results_df):.1f}%)")
    
    # Best/worst
    best_idx = results_df['tracking_error_abs'].idxmin()
    worst_idx = results_df['tracking_error_abs'].idxmax()
    
    print(f"\n✅ BEST Tracking:")
    best = results_df.loc[best_idx]
    print(f"  Period: {best['start_date']} to {best['end_date']}")
    print(f"  Difference: {best['tracking_diff']:.4f}%")
    
    print(f"\n❌ WORST Tracking:")
    worst = results_df.loc[worst_idx]
    print(f"  Period: {worst['start_date']} to {worst['end_date']}")
    print(f"  Difference: {worst['tracking_diff']:.4f}%")
    
    print("\n" + "=" * 80)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Portfolio Backtest Results - Market Meet', fontsize=16, fontweight='bold')
    
    # Tracking difference histogram
    ax1 = axes[0, 0]
    ax1.hist(results_df['tracking_diff'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Tracking')
    ax1.axvline(results_df['tracking_diff'].mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["tracking_diff"].mean():.3f}%')
    ax1.set_xlabel('Tracking Difference (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Tracking Differences')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(results_df['benchmark_return'], results_df['portfolio_return'], alpha=0.6)
    min_val = min(results_df['benchmark_return'].min(), results_df['portfolio_return'].min())
    max_val = max(results_df['benchmark_return'].max(), results_df['portfolio_return'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Tracking')
    ax2.set_xlabel('Benchmark Return (%)')
    ax2.set_ylabel('Portfolio Return (%)')
    ax2.set_title('Portfolio vs Benchmark')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Correlation
    ax3 = axes[1, 0]
    ax3.hist(results_df['correlation'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax3.axvline(results_df['correlation'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["correlation"].mean():.3f}')
    ax3.set_xlabel('Correlation')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Beta
    ax4 = axes[1, 1]
    ax4.hist(results_df['beta'], bins=20, edgecolor='black', alpha=0.7, color='green')
    ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Beta = 1.0')
    ax4.axvline(results_df['beta'].mean(), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["beta"].mean():.3f}')
    ax4.set_xlabel('Beta')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Beta')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
    print("\n📈 Plot saved: backtest_results.png")
    
    return results_df

def analyze_portfolio(portfolio_dict):
    """Quick portfolio analysis"""
    print("\n" + "=" * 80)
    print("PORTFOLIO ANALYSIS")
    print("=" * 80)
    
    # Basic stats
    print(f"\nNumber of stocks: {len(portfolio_dict)}")
    total_weight = sum(d['Weight_Percent'] for d in portfolio_dict.values())
    print(f"Total weight: {total_weight:.2f}%")
    
    # Currency split
    cad = [t for t in portfolio_dict if t.endswith(".TO")]
    usd = [t for t in portfolio_dict if not t.endswith(".TO")]
    cad_w = sum(portfolio_dict[t]["Weight_Percent"] for t in cad)
    usd_w = sum(portfolio_dict[t]["Weight_Percent"] for t in usd)
    
    print(f"\nCurrency Split:")
    print(f"  CAD: {len(cad)} stocks ({cad_w:.2f}%)")
    print(f"  USD: {len(usd)} stocks ({usd_w:.2f}%)")
    
    # Sectors
    sectors = {}
    for ticker, data in portfolio_dict.items():
        sector = data.get('Sector', 'Unknown')
        sectors[sector] = sectors.get(sector, 0) + data['Weight_Percent']
    
    print(f"\nTop Sectors:")
    for sector, weight in sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {sector}: {weight:.2f}%")
    
    # Position sizes
    weights = [d['Weight_Percent'] for d in portfolio_dict.values()]
    print(f"\nPosition Sizes:")
    print(f"  Largest: {max(weights):.2f}%")
    print(f"  Smallest: {min(weights):.2f}%")
    print(f"  Average: {np.mean(weights):.2f}%")
    
    # Top holdings
    print(f"\nTop 10 Holdings:")
    sorted_holdings = sorted(portfolio_dict.items(), 
                           key=lambda x: x[1]['Weight_Percent'], 
                           reverse=True)[:10]
    for i, (ticker, data) in enumerate(sorted_holdings, 1):
        print(f"  {i:2d}. {ticker:6s} {data['Weight_Percent']:5.2f}%  {data.get('Sector', 'Unknown')}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STANDALONE PORTFOLIO BACKTEST")
    print("=" * 80)
    
    if not PORTFOLIO_TO_TEST:
        print("\n⚠️  NO PORTFOLIO PROVIDED!")
        print("\nInstructions:")
        print("1. Run your main portfolio code")
        print("2. Copy the final_portfolio dictionary")
        print("3. Paste it into PORTFOLIO_TO_TEST at the top of this file")
        print("4. Run this script again")
        print("\nExample format:")
        print("PORTFOLIO_TO_TEST = {")
        print("    'AAPL': {'Score': 0.85, 'Weight_Percent': 5.2, 'Sector': 'Technology'},")
        print("    'MSFT': {'Score': 0.82, 'Weight_Percent': 4.8, 'Sector': 'Technology'},")
        print("    # ... more stocks")
        print("}")
    else:
        # Analyze portfolio
        analyze_portfolio(PORTFOLIO_TO_TEST)
        
        # Run backtest
        print("\n" + "=" * 80)
        input("Press Enter to start backtest (or Ctrl+C to cancel)...")
        
        results_df = run_backtest(PORTFOLIO_TO_TEST, num_periods=50)
        
        if results_df is not None:
            print("\n✅ BACKTEST COMPLETE!")
            print(f"\nKey takeaway for Market Meet:")
            avg_error = results_df['tracking_error_abs'].mean()
            if avg_error < 1.5:
                print(f"  🎯 Excellent! Avg tracking error: {avg_error:.3f}%")
            elif avg_error < 2.5:
                print(f"  👍 Good! Avg tracking error: {avg_error:.3f}%")
            elif avg_error < 4.0:
                print(f"  ⚠️  Moderate. Avg tracking error: {avg_error:.3f}%")
            else:
                print(f"  ❌ High tracking error: {avg_error:.3f}%")
                print("     Consider adjusting weights toward high-correlation stocks")
