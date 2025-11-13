import numpy as np
import yfinance as yf
import pandas as pd

def read_csv(filename):
    data=pd.read_csv(filename, header=None)
    l= data.values.tolist()
    list=[]
    for i in l:
        list.append(i[0])
    return list

def check_ticker(list):
    valid_tickers=[]
    invalid_tickers=[]
    start = "2024-10-01"
    end = "2025-10-01"

    market = yf.Ticker("^GSPC")
    market_data = market.history(start=start, end=end, interval="1d")
    market_data["Market_Return"] = market_data["Close"].pct_change()


    for ticker in list:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end, interval="1d")

        if data.empty:
            invalid_tickers.append(ticker)
            continue

        avg_volume = data["Volume"].mean()


        if avg_volume < 5000:
            invalid_tickers.append(ticker)
            continue

        valid_tickers.append(ticker)

        # For Companies in S&P 500/TSX, add them to the final list
        market_of_ticker = stock.info.get("market")
        if market_of_ticker not in ["us_market", "ca_market"]:
            invalid_tickers.append(ticker)
            valid_tickers.remove(ticker)
            continue

    return valid_tickers, invalid_tickers

def blended_benchmark(start, end):
    data = yf.download(["^GSPC", "^GSPTSE"], start=start, end=end)["Close"]
    rets = data.pct_change().dropna()
    blended = rets.mean(axis=1)
    return blended.rename("Benchmark")



def score_data(valid_tickers):
    start="2025-05-15"
    end="2025-11-15"


    def get_sector_safe(ticker):
        try:
            info = yf.Ticker(ticker).get_info()  # more reliable than .info on newer yfinance
        except Exception:
            info = {}
        sector = (
            info.get("sector")
            or info.get("industry")
            or info.get("categoryName")
            or "Unknown"
        )
        return sector
    
    valid_stocks_with_data = []
    bench = blended_benchmark(start, end)


    bench_vol_ann = float(bench.std() * np.sqrt(252))
    if not np.isfinite(bench_vol_ann) or bench_vol_ann <= 0:
        bench_vol_ann = np.nan
    
    window = 63

    for i in valid_tickers:
        stock_ret = yf.download(i, start=start, end=end, auto_adjust=True)["Close"].pct_change().dropna()

        stock_ret.name = i 
        df = pd.concat([stock_ret, bench], axis=1).dropna().rename(columns={"Benchmark": "Bench"})
        if len(df) < window:
            continue

        rolling_cov = df[i].rolling(window).cov(df["Bench"])
        rolling_var = df["Bench"].rolling(window).var()
        rolling_beta = rolling_cov / rolling_var
        beta_mean = rolling_beta.dropna().mean()

        rolling_corr = df[i].rolling(window).corr(df["Bench"])
        corr_mean = rolling_corr.dropna().mean()

        # Volatility Calculation
        vol_ann = float(stock_ret.std() * np.sqrt(252))
        raw_ratio = (vol_ann / bench_vol_ann) if np.isfinite(bench_vol_ann) else np.nan
        sigma_rel = float(raw_ratio / (1 + raw_ratio)) if np.isfinite(raw_ratio) else np.nan


        sector = get_sector_safe(i)


        valid_stocks_with_data.append([i, {'Beta': float(np.round(beta_mean, 5)), 'Correlation': float(np.round(corr_mean, 5)), 'Volatility_Ann': float(np.round(vol_ann, 5)), 'Sigma_Rel': float(np.round(sigma_rel, 5)), 'Sector': sector}])
        
    return valid_stocks_with_data

def filter_out_low_weight_stocks(final_stocks):
    final_portfolio = {}
    for ticker, vals in final_stocks.items():
        if vals[2] > 0.0:
            final_portfolio[ticker] = {
                "Score": vals[0],
                "Weight_Percent": vals[2],
                "Sector": vals[1]
            }
    return dict(sorted(final_portfolio.items(),
                       key=lambda kv: kv[1]["Weight_Percent"],
                       reverse=True))

def add_defensive_layer(final, scored_data, defensive_ratio=0.08):
    if not final:
        return final
    
    allowed_sectors = {"Utilities", "Consumer Defensive", "Healthcare"}

    defensives = [
        (t, m)
        for t, m in scored_data
        if isinstance(m.get("Beta"), (int, float))
        and isinstance(m.get("Volatility_Ann"), (int, float))
        and m["Beta"] < 0.9
        and m["Volatility_Ann"] < 0.25
        and m.get("Sector") in allowed_sectors
    ]
    if not defensives:
        return final
    
    defensives = sorted(
        defensives,
        key=lambda x: (x[1].get("Correlation") if pd.notna(x[1].get("Correlation")) else -np.inf),
        reverse=True
    )[:3]

    selected = {t for t, _ in defensives}
    total_before = sum(v["Weight_Percent"] for v in final.values())
    if total_before <= 0:
        return final
    
    def_total = defensive_ratio * 100.0
    selected_sum = sum(final[t]["Weight_Percent"] for t in final if t in selected)
    nondef_sum = total_before - selected_sum
    if nondef_sum <= 0:
        return final
    
    scale = (total_before - def_total) / nondef_sum
    for t, data in final.items():
        if t not in selected:
            data["Weight_Percent"] = float(np.round(data["Weight_Percent"] * scale, 5))

    each = float(np.round(def_total / len(defensives), 5))
    for t, m in defensives:
        if t in final:
            final[t]["Weight_Percent"] = each
            final[t]["Sector"] = m.get("Sector", final[t]["Sector"])
        else:
            # If a defensive pick wasn't in the core, add it with a neutral score
            final[t] = {"Score": 0.0, "Weight_Percent": each, "Sector": m.get("Sector", "Unknown")}
        
    total_after = sum(v["Weight_Percent"] for v in final.values())
    if total_after > 0 and abs(total_after - 100.0) > 1e-6:
        norm = 100.0 / total_after
        for t in final:
            final[t]["Weight_Percent"] = float(np.round(final[t]["Weight_Percent"] * norm, 5))

    return dict(sorted(final.items(), key=lambda kv: kv[1]["Weight_Percent"], reverse=True))

def rebalance_currency_mix(final, target_ratio=0.5, tolerance=0.1):
    if not final:
        return final
    
    cad_tickers = [t for t in final if t.endswith(".TO")]
    usd_tickers = [t for t in final if not t.endswith(".TO")]

    cad_weight = sum(final[t]["Weight_Percent"] for t in cad_tickers)
    usd_weight = sum(final[t]["Weight_Percent"] for t in usd_tickers)
    total = cad_weight + usd_weight
    if total == 0:
        return final
    
    cad_ratio = cad_weight / total

    if cad_weight == 0 or usd_weight == 0:
        return final
    
    if abs(cad_ratio - target_ratio) > tolerance:
        target_cad = target_ratio * 100.0
        target_usd = (1 - target_ratio) * 100.0

        scale_cad = target_cad / cad_weight
        scale_usd = target_usd / usd_weight

        for t in cad_tickers:
            final[t]["Weight_Percent"] = round(final[t]["Weight_Percent"] * scale_cad, 5)
        for t in usd_tickers:
            final[t]["Weight_Percent"] = round(final[t]["Weight_Percent"] * scale_usd, 5)

        new_total = sum(v["Weight_Percent"] for v in final.values())
        if new_total > 0:
            norm = 100.0 / new_total
            for t in final:
                final[t]["Weight_Percent"] = round(final[t]["Weight_Percent"] * norm, 5)

    return dict(sorted(final.items(), key=lambda kv: kv[1]["Weight_Percent"], reverse=True))

def apply_risk_constraints(final, max_position=10.0, max_sector=35.0, max_iters=3):
    if not final:
        return final
    for _ in range(max_iters):
        changed = False

        for t, d in final.items():
            if d["Weight_Percent"] > max_position:
                d["Weight_Percent"] = float(np.round(max_position, 5))
                changed = True
        sector_totals = {}
        for t, d in final.items():
            s = d.get("Sector", "Unknown")
            sector_totals[s] = sector_totals.get(s, 0.0) + d["Weight_Percent"]

        for s, tot in sector_totals.items():
            if tot > max_sector:
                scale = max_sector / tot
                for t, d in final.items():
                    if d.get("Sector", "Unknown") == s:
                        d["Weight_Percent"] = float(np.round(d["Weight_Percent"] * scale, 5))
                        changed = True

        total_w = sum(v["Weight_Percent"] for v in final.values())
        if total_w > 0 and abs(total_w - 100.0) > 1e-6:
            norm = 100.0 / total_w
            for t in final:
                final[t]["Weight_Percent"] = float(np.round(final[t]["Weight_Percent"] * norm, 5))
            changed = True

        if not changed:
            break

    return dict(sorted(final.items(), key=lambda kv: kv[1]["Weight_Percent"], reverse=True))


def score_calculate(valid_tickers):
    x = score_data(valid_tickers)
    final = {}
    total_weight = 0.0

    w1, w2, w3 = 0.4, 0.4, 0.2

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

    final = add_defensive_layer(final, x, defensive_ratio=0.08)

    final = apply_risk_constraints(final, max_position=10.0, max_sector=35.0)

    final = rebalance_currency_mix(final)

    return final



def main():
    tickers_list = read_csv("Tickers.csv")
    valid, invalid = check_ticker(tickers_list)

    x = score_calculate(valid)

    print("Valid:", valid)
    print()
    
    print("Invalid:", invalid)
    print()

    print(x)
    print()

if __name__ == "__main__":
    main()



