# rsi.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's method (TradingView style)."""
    prices = prices.squeeze()  # ensure Series
    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Step 1: initial average gain/loss (simple average over first period)
    avg_gain = pd.Series(np.nan, index=prices.index)
    avg_loss = pd.Series(np.nan, index=prices.index)

    avg_gain.iloc[period] = gain.iloc[1:period+1].mean()
    avg_loss.iloc[period] = loss.iloc[1:period+1].mean()

    # Step 2: Wilder's smoothing for the rest
    for i in range(period+1, len(prices)):
        avg_gain.iat[i] = (avg_gain.iat[i-1] * (period - 1) + gain.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i-1] * (period - 1) + loss.iat[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def main():
    ticker = "NVDA"   # change this to any stock symbol
    period = 14       # RSI lookback period
    years = 1       # how many years of history

    # Download price data
    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        print("No data downloaded. Check ticker symbol.")
        return

    # Compute RSI and its moving average
    df["RSI"] = compute_rsi(df["Close"], period=period)
    df["RSI_MA"] = df["RSI"].rolling(window=period).mean()

    # Save to CSV
    out_csv = f"{ticker}_RSI_{period}_{years}y.csv"
    df.to_csv(out_csv)
    print(f"Saved RSI values to {out_csv}")

    # Plot RSI
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["RSI"], label=f"RSI-{period}", color="blue")
    plt.plot(df.index, df["RSI_MA"], label=f"RSI-{period} MA", color="gold")
    plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    plt.title(f"{ticker} RSI-{period} over {years} years")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
