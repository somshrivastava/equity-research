# bollinger.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def compute_bollinger(prices: pd.Series, window: int = 20, num_std: int = 2):
    """Compute Bollinger Bands (SMA ± num_std * stddev)."""
    sma = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = sma + num_std * rolling_std
    lower_band = sma - num_std * rolling_std
    return sma, upper_band, lower_band

def main():
    ticker = "NVDA"   # change to any ticker
    years = 1
    window = 20
    num_std = 2

    # Download data
    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        print("No data downloaded. Check ticker symbol.")
        return

    close = df["Close"]
    df["SMA"], df["Upper"], df["Lower"] = compute_bollinger(close, window, num_std)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, close, label=f"{ticker} Price", color="blue")
    plt.plot(df.index, df["SMA"], label=f"SMA-{window}", color="orange")
    plt.plot(df.index, df["Upper"], label="Upper Band", color="green", linestyle="--")
    plt.plot(df.index, df["Lower"], label="Lower Band", color="red", linestyle="--")
    plt.fill_between(df.index, df["Lower"], df["Upper"], color="gray", alpha=0.1)

    plt.title(f"{ticker} Bollinger Bands ({window}-day, ±{num_std}σ)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
