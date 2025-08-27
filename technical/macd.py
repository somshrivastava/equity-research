# macd.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def compute_macd(prices: pd.Series, short: int = 12, long: int = 26, signal: int = 9):
    """Compute MACD line, Signal line, and Histogram."""
    ema_short = prices.ewm(span=short, adjust=False).mean()
    ema_long = prices.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def main():
    ticker = "NVDA"   # default ticker
    years = 1

    # Download data
    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        print("No data downloaded. Check ticker symbol.")
        return

    # Compute MACD
    df["MACD"], df["Signal"], df["Hist"] = compute_macd(df["Close"])

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Price chart
    ax1.plot(df.index, df["Close"], label=f"{ticker} Price", color="blue")
    ax1.set_title(f"{ticker} Price with MACD Indicator")
    ax1.set_ylabel("Price (USD)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # MACD chart
    ax2.plot(df.index, df["MACD"], label="MACD", color="purple")
    ax2.plot(df.index, df["Signal"], label="Signal Line", color="orange")
    ax2.bar(df.index, df["Hist"], label="Histogram", color="gray", alpha=0.6)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("MACD")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
