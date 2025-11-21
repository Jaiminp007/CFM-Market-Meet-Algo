# Import essential libraries for numerical computation, financial data retrieval, and data processing
import numpy as np
import yfinance as yf
import pandas as pd

def read_csv(filename):
    data = pd.read_csv(filename, header=None)
    l= data.values.tolist()
    list=[]
    for i in l:
        list.append(i[0])
    return list
    
# Checks each ticker to determine if it is valid or invalid based on data availability,
# trading volume, and market listing. It returns two lists: valid_tickers and invalid_tickers
# Function that checks every ticker to determine if it is valid or invalid 
# It returns two lists: valid_tickers and invalid_tickers   
def check_ticker(list):
    valid_tickers=[]
    invalid_tickers=[]
    start = "2024-10-01"
    end = "2025-10-01"
    # Retrieve S&P 500 history (used to validate data availability)
    market = yf.Ticker("^GSPC")
    market_data = market.history(start=start, end=end, interval="1d")
    market_data["Market_Return"] = market_data["Close"].pct_change()

    info_cache = {}
    for ticker in list:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end, interval="1d")

        # If there's no historical price data that exists, the stock is marked as invalid
        if data.empty:
            invalid_tickers.append(ticker)
            continue

        avg_volume = data["Volume"].mean()

        # If the volume of the ticker is less than 5000, the ticker is considered invalid
        if avg_volume < 5000:
            invalid_tickers.append(ticker)
            continue
        try:
            if ticker not in info_cache:
                info_cache[ticker] = stock.get_info()
            market_of_ticker = info_cache[ticker].get("market")
        except Exception:
            market_of_ticker = None
        
        if market_of_ticker not in ["us_market", "ca_market"]:
            invalid_tickers.append(ticker)
            continue

        valid_tickers.append(ticker)

    return valid_tickers, invalid_tickers

# Calculates the combined benchmark of both S&P 500 and TSX, creating a blended benchmark
def blended_benchmark(start, end):
    data = yf.download(["^GSPC", "^GSPTSE"], start=start, end=end)["Close"]
    blended_price = 0.5 * data["^GSPC"] + 0.5 * data["^GSPTSE"]
    returns = blended_price.pct_change().dropna()
    return returns.rename("Benchmark")

# Function that takes a list of valid tickers + sets time range

def score_data(valid_tickers):
    start="2025-05-15"
    end="2025-11-15"

# Retrieves the sector from yfinance 
    def get_sector_safe(ticker):
        try:
            info = yf.Ticker(ticker).get_info()  # Retrieves a dictionary of data from the company
    
        except Exception:    # This makes sure that if theres missing data or any errors, we run an empty dictionary instead of just crashing 
            info = {}
        sector = ( 
            info.get("sector")
            or info.get("industry")
            or info.get("categoryName")
            or "Unknown"
        )
        # Returns the best available sector field 
        # Will move onto industry if there's no sector for example
        return sector 

    # Empty lists to store results for each ticker then generates the blended benchmark
    valid_stocks_with_data = []
    bench = blended_benchmark(start, end)

# Formula for converting the daily volatility to the annual volatility
    bench_vol_ann = float(bench.std() * np.sqrt(252))
    if bench_vol_ann <= 0 or not np.isfinite(bench_vol_ann):
        bench_vol_ann = bench.std() * np.sqrt(252) # If the value is invalid, it's replaced with np.nan
   
    # Sets the number of days in the rolling window (as we have a rolling beta and rolling correlation)
    # to roughly 3 trading months
    window = 63 


    # Looping through all valid tickers and extracting daily returns
    for i in valid_tickers:
        stock_ret = yf.download(i, start=start, end=end, auto_adjust=True)["Close"].pct_change().dropna()

    # Name return series as the ticker and combine stock's daily returns with blended benchmark daily returns
        stock_ret.name = i 
        df = pd.concat([stock_ret, bench], axis=1).dropna().rename(columns={"Benchmark": "Bench"}) #            Is renaming to "Bench" really necessary???
        if len(df) < window: # If there's not at least 63 days of overlapping data, we skip the ticker
            continue

        # This calculates the rolling beta between the stock and blended benchmark over the given time period
        # A rolling beta recalculates the beta every time new data comes in for a 63 day window, adding more accuracy than a standard beta calculation
        rolling_cov = df[i].rolling(window).cov(df["Bench"])
        rolling_var = df["Bench"].rolling(window).var()
        rolling_beta = rolling_cov / rolling_var

        # Average of all rolling beta values to get an overall beta estimate
        beta_mean = rolling_beta.dropna().mean()

        # Calculates rolling correlation between stock and benchmark and finds average of all values 
        rolling_corr = df[i].rolling(window).corr(df["Bench"])
        corr_mean = rolling_corr.dropna().mean()

        # Volatility Calculation
        vol_ann = stock_ret.std() * np.sqrt(252)
        raw_ratio = (vol_ann / bench_vol_ann) if np.isfinite(bench_vol_ann) else np.nan
        sigma_rel = float(raw_ratio / (1 + raw_ratio)) if (isinstance(raw_ratio, (int, float)) and np.isfinite(raw_ratio)) else np.nan


        # Retrieves info by calling the function
        sector = get_sector_safe(i)

        # Stores all the metrics for given ticker
        valid_stocks_with_data.append([i, {'Beta': float(np.round(beta_mean, 5)), 'Correlation': float(np.round(corr_mean, 5)), 'Volatility_Ann': float(np.round(vol_ann, 5)), 'Sigma_Rel': float(np.round(sigma_rel, 5)), 'Sector': sector}])
        
    return valid_stocks_with_data


# Cleans up the scored tickers, removing stocks with zero/negative weight and 
# sorts them from highest weight to lowest

def filter_out_low_weight_stocks(final_stocks):
    final_portfolio = {}
    for ticker, vals in final_stocks.items():
        # vals = [score, sector, weight_percent]
        if vals[2] > 0.0: # Dropping tickers with zero or negative weight
            final_portfolio[ticker] = {
                "Score": vals[0],
                "Weight_Percent": vals[2],
                "Sector": vals[1]
            }
 # Then, sort the tickers by weight from highest to lowest 

    return dict(sorted(final_portfolio.items(),
                       key=lambda kv: kv[1]["Weight_Percent"],
                       reverse=True))

# Function that finds low-beta/low-volatility stocks to reduce risk for the portfolio in case the market drops
def add_defensive_layer(final, scored_data, defensive_ratio=0.05):
    if not final:
        return final # Stops the function if the portfolio is empty (no portfolio to modify)

    # Sectors that are relatively safe even during recessions 
    allowed_sectors = {"Utilities", "Consumer Defensive", "Healthcare"}

    # A criteria for selecting low-risk stocks 
    defensives = [
        (t, m)
        for t, m in scored_data
        if isinstance(m.get("Beta"), (int, float))
        and isinstance(m.get("Volatility_Ann"), (int, float))
        and m["Beta"] < 0.9 # beta needs to be <0.9
        and m["Volatility_Ann"] < 0.25 #Volatility needs to be <0.25
        and m.get("Sector") in allowed_sectors # stock needs to be in one of the chosen sectors
    ]

# don't add defensive layer if nothing qualifies
    if not defensives:
        return final

# Sort the qualified stocks by correlation with benchmark and take the top 3
    defensives = sorted(
        defensives,
        key=lambda x: (x[1].get("Correlation") if pd.notna(x[1].get("Correlation")) else -np.inf),
        reverse=True
    )[:3]

# Makes a set for the 3 chosen stocks
    selected = {t for t, _ in defensives}

# Make sure the portfolio isn't empty/corrupted
    total_before = sum(v["Weight_Percent"] for v in final.values())
    if total_before <= 0:
        return final

# Distribute 5% of the total portfolio to defensive stocks
    def_total = defensive_ratio * 100.0
# How much weight do the selected stocks already occupy? 
    selected_sum = sum(final[t]["Weight_Percent"] for t in final if t in selected)
# How much of the weight is not taken up by the safety stocks?
    nondef_sum = total_before - selected_sum
# In case defensive stocks are >= 100% (stops an edge case)
    if nondef_sum <= 0:
        return final

# Find how much we need to adjust stock weightings so defensive ones take up 5%
    scale = (total_before - def_total) / nondef_sum
    for t, data in final.items():
        if t not in selected: # If a stock is NOT defensive, multiply it by "scale"
            data["Weight_Percent"] = float(np.round(data["Weight_Percent"] * scale, 5)) # This shifts the rest of the portfolio to 95%

    # Distribute an equal weight to each defensive stock
    each = float(np.round(def_total / len(defensives), 5))
    for t, m in defensives:
        if t in final:
            final[t]["Weight_Percent"] = each
            final[t]["Sector"] = m.get("Sector", final[t]["Sector"])
        else:
            # If a defensive pick wasn't in the top 3, add it with a score of 0
            final[t] = {"Score": 0.0, "Weight_Percent": each, "Sector": m.get("Sector", "Unknown")}

# Make sure that the total portfolio still sums to exactly 100% (might be some rounding errors)
    total_after = sum(v["Weight_Percent"] for v in final.values())
    if total_after > 0 and abs(total_after - 100.0) > 1e-6:
        norm = 100.0 / total_after
        for t in final:
            final[t]["Weight_Percent"] = float(np.round(final[t]["Weight_Percent"] * norm, 5))

# Return the completed portfolio from highest -> lowest weight
    return dict(sorted(final.items(), key=lambda kv: kv[1]["Weight_Percent"], reverse=True))

# Function that tries to keep CAD stocks and USD stocks equal 
def rebalance_currency_mix(final, target_ratio=0.5, tolerance=0.1):
    if not final:
        return final # Like before if portfolio is empty, stop
    
    cad_tickers = [t for t in final if t.endswith(".TO")]  #TSX (ends in .TO)
    usd_tickers = [t for t in final if not t.endswith(".TO")]

    # Find current weights
    cad_weight = sum(final[t]["Weight_Percent"] for t in cad_tickers)
    usd_weight = sum(final[t]["Weight_Percent"] for t in usd_tickers)
    total = cad_weight + usd_weight
    if total == 0:
        return final

    # Compute CAD stock ratio
    cad_ratio = cad_weight / total

    # Stop if there's no CAD or USD stocks
    if cad_weight == 0 or usd_weight == 0:
        return final

    # If the difference between the ratios is <= 10%, don't bother rebalancing
    if abs(cad_ratio - target_ratio) > tolerance:
        # Find the target weightings
        target_cad = target_ratio * 100.0
        target_usd = (1 - target_ratio) * 100.0

        # Ratio to multiply with each weighting to fix it
        scale_cad = target_cad / cad_weight
        scale_usd = target_usd / usd_weight

        # Now we loop through each CAD and US stock and multiply the scale to each
        for t in cad_tickers:
            final[t]["Weight_Percent"] = round(final[t]["Weight_Percent"] * scale_cad, 5)
        for t in usd_tickers:
            final[t]["Weight_Percent"] = round(final[t]["Weight_Percent"] * scale_usd, 5)

        # Make sure the whole portfolio sums to exactly 100% (avoids rounding errors)
        new_total = sum(v["Weight_Percent"] for v in final.values())
        if new_total > 0:
            norm = 100.0 / new_total
            for t in final:
                final[t]["Weight_Percent"] = round(final[t]["Weight_Percent"] * norm, 5)

    # Return the final dictionary by weightings
    return dict(sorted(final.items(), key=lambda kv: kv[1]["Weight_Percent"], reverse=True))

# Makes sure the portfolio doesn't break the rules
def apply_risk_constraints(final, max_position=15.0, max_sector=40.0, max_iters=50):
    if not final:
        return final #Nothing in the portfolio? Skip it

    # Loop through until all rules are fulfilled 
    for _ in range(max_iters):
        changed = False

        # Loop through all stocks to make sure they take up max 15%
        for t, d in final.items():
            if d["Weight_Percent"] > max_position:
                d["Weight_Percent"] = float(np.round(max_position, 5))
                changed = True # Report that something in the portfolio was fixed - loop again

        # Builds a dictionary with weight for each sector
        sector_totals = {}
        for t, d in final.items():
            s = d.get("Sector", "Unknown")
            sector_totals[s] = sector_totals.get(s, 0.0) + d["Weight_Percent"]

        # Check if sector weights exceed 40%
        for s, tot in sector_totals.items():
            if tot > max_sector:
                scale = max_sector / tot #If they do, shrink them using this variable
                for t, d in final.items():
                    if d.get("Sector", "Unknown") == s:
                        d["Weight_Percent"] = float(np.round(d["Weight_Percent"] * scale, 5)) # Apply the scale variable to all the stocks in that sector
                        changed = True # Report a change has been made to the portfolio - Loop again

        total_w = sum(v["Weight_Percent"] for v in final.values())
        if total_w > 0 and abs(total_w - 100.0) > 1e-6:         # If the total isn't exactly 100%, adjust 
            norm = 100.0 / total_w # Scale factor to fix the portfolio
            for t in final:
                final[t]["Weight_Percent"] = float(np.round(final[t]["Weight_Percent"] * norm, 5)) # Apply the change to all weights
            changed = True # Report something changed

        if not changed: # If no more rules have been violated, break the loop
            break

    return dict(sorted(final.items(), key=lambda kv: kv[1]["Weight_Percent"], reverse=True)) # Sort stocks by weight from highest to lowest

# Make sure portolio isn't bigger than 25 stocks or less than 10 stocks
def limit_portfolio_size(final, max_size=25, min_size=10):
    if not final:
        return final 
        
    items = sorted(final.items(), key=lambda kv: kv[1]["Weight_Percent"], reverse=True) # Sorts a list ranking all stocks by weighting
    n = len(items) # Count how many tickers are in the portfolio
   
    if n <= max_size: # If n is <= 25, leave everything as is
        return dict(items)
    trimmed = dict(items[:max_size]) # Otherwise, only keep the top 25 tickers
   
    # Adjust the portfolio so it sums to exactly 100%
    total = sum(v["Weight_Percent"] for v in trimmed.values())
    if total > 0:
        norm = 100.0 / total
        for t in trimmed:
            trimmed[t]["Weight_Percent"] = float(np.round(trimmed[t]["Weight_Percent"] * norm, 5))

    # Run the previous helpers to make sure the trimmed portfolio doesn't violate prior rules
    trimmed = apply_risk_constraints(trimmed, max_position=15.0, max_sector=40.0)
    trimmed = rebalance_currency_mix(trimmed)

    return trimmed

# 
def enforce_min_weight(final):
    if not final:
        return final
    
    n = len(final)
    min_weight = 100.0 / (2 * n)
    
    # Remove stocks below minimum weight
    to_remove = [t for t, d in final.items() if d["Weight_Percent"] < min_weight]
    
    for ticker in to_remove:
        del final[ticker]
    
    # Renormalize if we removed anything
    if to_remove:
        total = sum(v["Weight_Percent"] for v in final.values())
        if total > 0:
            norm = 100.0 / total
            for t in final:
                final[t]["Weight_Percent"] = float(np.round(final[t]["Weight_Percent"] * norm, 5))
    
    return dict(sorted(final.items(), key=lambda kv: kv[1]["Weight_Percent"], reverse=True))

def market_cap_filtering(final, scored_data):
    if not final:
        return final

    CAD_PER_USD = 1.38

    def get_mc_in_cad(ticker):
        try:
            info = yf.Ticker(ticker).get_info()
            mc = info.get("marketCap", None)
            if mc is None:
                return None
            return mc if ticker.endswith(".TO") else mc * CAD_PER_USD
        except:
            return None

    def get_sector_of(t):
        for tick, metrics in scored_data:
            if tick == t:
                return metrics.get("Sector", "Unknown")
        return "Unknown"

    caps = {t: get_mc_in_cad(t) for t in final}

    has_large = any(mc and mc > 10_000_000_000 for mc in caps.values())
    has_small = any(mc and mc < 2_000_000_000 for mc in caps.values())

    if has_large and has_small:
        return final

    scored_caps = {}
    for t, m in scored_data:
        mc = get_mc_in_cad(t)
        if mc is not None:
            scored_caps[t] = mc

    if not has_large:
        candidates = [t for t, mc in scored_caps.items() if mc > 10_000_000_000]
        if candidates:
            t = candidates[0]
            final[t] = {"Score": 0.5, "Weight_Percent": 0.0, "Sector": get_sector_of(t)}

    if not has_small:
        candidates = [t for t, mc in scored_caps.items() if mc < 2_000_000_000]
        if candidates:
            t = candidates[0]
            final[t] = {"Score": 0.5, "Weight_Percent": 0.0, "Sector": get_sector_of(t)}

    total = sum(v["Weight_Percent"] for v in final.values())
    if total > 0:
        for t in final:
            final[t]["Weight_Percent"] = round(final[t]["Weight_Percent"] * (100 / total), 5)

    return final

def shrink_weights_for_fees(final, cash_buffer_bps=25):
    if not final:
        return final
    buffer_pct = cash_buffer_bps / 100.0
    scale = (100.0 - buffer_pct) / 100.0
    for t in final:
        final[t]["Weight_Percent"] = float(np.round(final[t]["Weight_Percent"] * scale, 5))
    return dict(sorted(final.items(), key=lambda kv: kv[1]["Weight_Percent"], reverse=True))

def net_returns_after_mgmt_fee(daily_returns, annual_fee_bps=50):
    fee_daily = annual_fee_bps / 10000.0 / 252.0
    return daily_returns - fee_daily

def score_calculate(valid_tickers):
    x = score_data(valid_tickers)
    final = {}
    total_weight = 0.0

    w1, w2, w3 = 0.45, 0.45, 0.1

    for ticker, m in x:
        beta = m['Beta']
        corr = m['Correlation']
        sigma_rel = m['Sigma_Rel']
        
        vol_ratio = sigma_rel / (1 - sigma_rel) if sigma_rel < 1 else 1.0
        
        distance = np.sqrt(
            w1 * (beta - 1) ** 2 +
            w2 * (1 - corr) ** 2 +
            w3 * (vol_ratio - 1) ** 2
        )

        score = 1 / (1 + distance)

        if not np.isfinite(score):
            continue

        weight = score
        total_weight += weight

        final[ticker] = [float(np.round(score, 5)), m['Sector']]

    for i in final:
        score_value = final[i][0]
        weight_in_percent = (score_value / total_weight) * 100 if total_weight != 0 else 0.0
        final[i].append(float(np.round(weight_in_percent, 5)))


    final = filter_out_low_weight_stocks(final)

    final = add_defensive_layer(final, x, defensive_ratio=0.05)

    final = rebalance_currency_mix(final)

    final = market_cap_filtering(final, x)

    final = limit_portfolio_size(final, max_size=25, min_size=10)

    final = enforce_min_weight(final)
    
    final = apply_risk_constraints(final, max_position=15.0, max_sector=40.0)

    final = shrink_weights_for_fees(final, cash_buffer_bps=25)

    return final


def create_portfolio_dataframe(final_portfolio, total_value_cad=1000000):
    """Create the final portfolio DataFrame with all required columns"""
    
    CAD_PER_USD = 1.38
    
    portfolio_data = []
    
    for ticker, data in final_portfolio.items():
        # Get current price
        try:
            stock = yf.Ticker(ticker)
            info = stock.get_info()
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if price is None:
                hist = stock.history(period="5d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                else:
                    continue
        except:
            continue
        
        # Determine currency
        currency = "CAD" if ticker.endswith(".TO") else "USD"
        
        # Calculate allocation in CAD
        weight_decimal = data['Weight_Percent'] / 100
        allocation_cad = total_value_cad * weight_decimal
        
        # Convert to appropriate currency for shares calculation
        if currency == "USD":
            allocation_in_currency = allocation_cad / CAD_PER_USD
        else:
            allocation_in_currency = allocation_cad
        
        # Calculate shares
        shares = round(allocation_in_currency / price, 4)
        
        # Calculate actual value in original currency
        value_in_currency = shares * price
        
        # Convert value to CAD
        value_cad = value_in_currency if currency == "CAD" else value_in_currency * CAD_PER_USD
        
        portfolio_data.append({
            'Ticker': ticker,
            'Price': round(price, 2),
            'Currency': currency,
            'Shares': shares,
            'Value': round(value_cad, 2),
            'Weight': round(data['Weight_Percent'], 2)
        })
    
    df = pd.DataFrame(portfolio_data)
    df.index = range(1, len(df) + 1)
    
    return df

def save_stocks_csv(portfolio_df, group_number, directory="."):
    """Save Ticker and Shares to CSV"""
    stocks_df = portfolio_df[['Ticker', 'Shares']].copy()
    filename = f"{directory}/Stocks_Group_{group_number:02d}.csv"
    stocks_df.to_csv(filename, index=False)
    print(f"\nStocks CSV saved to: {filename}")
    return filename

def main():
    tickers_list = read_csv("Tickers.csv")
    valid, invalid = check_ticker(tickers_list)

    final_portfolio = score_calculate(valid)
    
    print(f"Final portfolio contains {len(final_portfolio)} stocks\n")
    print(final_portfolio)
    
    # Account for fees (0.25% buffer already applied in shrink_weights_for_fees)
    TOTAL_PORTFOLIO_VALUE = 1000000  # $1M CAD
    FEES_BPS = 25  # 0.25% = 25 basis points
    available_to_invest = TOTAL_PORTFOLIO_VALUE * (1 - FEES_BPS/10000)
    
    # Create portfolio DataFrame
    portfolio_df = create_portfolio_dataframe(final_portfolio, available_to_invest)
    
    # Display Portfolio_Final DataFrame
    print("="*80)
    print("PORTFOLIO_FINAL")
    print("="*80)
    print(portfolio_df.to_string())
    print("\n" + "="*80)
    
    # Calculate actual totals
    total_value = portfolio_df['Value'].sum()
    total_weight = portfolio_df['Weight'].sum()
    
    print(f"\nTotal Portfolio Value: ${total_value:,.2f} CAD")
    print(f"Total Weight: {total_weight:.2f}%")
    print(f"Cash Reserve (fees): ${TOTAL_PORTFOLIO_VALUE - total_value:,.2f} CAD")
    print(f"Portfolio + Cash: ${total_value + (TOTAL_PORTFOLIO_VALUE - total_value):,.2f} CAD")
    
    # Save stocks CSV
    save_stocks_csv(portfolio_df, 13)
    
    # Save full portfolio to CSV as well (optional)
    portfolio_df.to_csv(f"Portfolio_Group_{13:02d}.csv")
    print(f"Full portfolio saved to: Portfolio_Group_{13:02d}.csv")
if __name__ == "__main__":
    main()
