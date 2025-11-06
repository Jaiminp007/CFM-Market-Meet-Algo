# The Game Plan

# Objective

## Competition Goals (choose one)

## Market Meet -- Return closest to the benchmark average (above or below)

## Benchmark Average: Simple arithmetic average of the total returns of the TSX Composite and S&P 500 over the contest period

# Brainstorming How to Do

- Recalculate benchmark average in each iteration
- Check stocks to make sure they're moving the same way as the benchmark
- Weigh each stock differently depending on similarity of different factors, to the movement of benchmark average

First analyze how the benchmark is moving (is it stable, growing fast or falling quickly etc)

- Based on this movement, pick stocks that would be best suited to mimic this in the 5 days for every scenario

If stock is stable and valuation is > 10B, (i.e. less than a certain volatility) -- Immediately add it to the list

If stock is stable but valuation is < 2B, Goes through another if statement. -- calculate a short-term (e.g. 5 day) **correlation (directional similarity)** between the benchmark and the stock, and the **beta (magnitudinal similarity), and the volatility**

If stable but not following benchmark average movement

If a stock is risky, add ONLY if you can find another stock to offset that movement

- Get BETA as close to 1.0 as possible (That means it's in line with the overall market)
- Run a loop to store all the values closest to 1. If the beta value of one stock is closer than a previous one it replaces it
- If beta is > 1.5, send it to secondary checking

If beta > 1.5 -- get sent to pairing list

If beta ~1 BUT volatility is high -- get sent to pairing list

## Edge Cases

- Non-existent tickers
- Non-existent date for a ticker
- A huge list of super risky stocks that aren't stable
- Gives us penny stocks
- Currency change because S&P stocks are in USD

## Team Member Assignments

| Team Member | Jaimin | Elliot | Frank |
|-------------|--------|--------|-------|
| Task        |        |        |       |

# Different Strategies We Can Use

## 1. Index Mirroring (Best Approach)

- **Split allocation**: ~50% TSX stocks, ~50% S&P 500 stocks
- **Replicate top holdings**: Choose the largest companies from each index
- **Match sector weights**: If TSX is heavy in financials/energy and S&P 500 is heavy in tech, reflect that in your portfolio

## 2. Diversification Strategy

- **Spread across sectors**: No more than 40% in one sector (it's a rule anyway)
- **Mix of 15-20 stocks**: Not too few (risky), not too many (hard to manage)
- **Balance volatility**: Include both stable blue-chips and some moderate-growth stocks

## 3. Geographic Balance

- **Canadian stocks (TSX)**: Banks (RBC, TD), Energy (Suncor, Enbridge), Shopify
- **US stocks (S&P 500)**: FAANG stocks, major industrials, consumer staples
- **Currency consideration**: Mix USD and CAD stocks roughly equally

## 4. Market Cap Mix Strategy

- Include **large-caps** (>$10B): Apple, Microsoft, RBC, TD - these move with the market
- Include **1-2 small-caps** (<$2B): Just to meet the requirement, keep their weight minimal
- **Avoid mid-caps dominating**: They're more volatile

## 5. Low Volatility Focus

- Choose stocks with **beta close to 1.0** (moves with the market)
- Avoid high-beta stocks (>1.5) - they amplify movements
- Look for stocks with steady historical returns

## 6. Sector Allocation

Based on typical index weights:

- **Financials**: 15-20% (banks, insurance)
- **Technology**: 15-20% (FAANG, Shopify)
- **Energy**: 10-15% (oil & gas)
- **Industrials**: 10-15%
- **Consumer**: 10-15%
- **Healthcare**: 5-10%
- **Others**: Fill remaining

# Pseudo Code

Calculate Benchmark Average for S and P 500

Calculate Benchmark Average for TSX

List of all valid Stock_ticker = []

While(stock_ticker)

> If stock.volume < 5000 for Oct 1 2024 to Sept 30, 2025:
>
> Drop months with < 18 trading days
>
> If yfinance can't find the ticker && the ticker doesnt exist for the date:
>
> Drop stock
