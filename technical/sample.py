import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

today = pd.Timestamp.today().normalize()
start_date = today - pd.DateOffset(years=4)
cutoff_date = today - pd.DateOffset(months=3) 

df = yf.download(
    "NVDA",
    start=start_date.date(),
    end=cutoff_date.date(), 
    interval="1d",
    auto_adjust=True,
    progress=False
)

if df.empty:
    print("No data found for the specified date range.")
else:

    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    else:
        df = df.rename(columns={"Close": "AdjClose"})

    df["SMA_90"] = df["AdjClose"].rolling(window=90, min_periods=90).mean()
    df["SMA_30"] = df["AdjClose"].rolling(window=30, min_periods=30).mean()
    df["SMA_10"] = df["AdjClose"].rolling(window=10, min_periods=10).mean()

    df["Label"] = ((df["AdjClose"].shift(-1) > df["AdjClose"]).replace({True: 1, False: -1}))

    df_out = df[["AdjClose", "SMA_90", "SMA_30", "SMA_10", "Label"]].dropna()
    df_out.index.name = "Date"

    df_out.to_csv("nvda_4y_excl_last3mo.csv")
    df_out.to_csv("nvda_4y_excl_last3mo.txt", sep=" ", header=False, index=False, float_format="%.6f")

    plt.figure(figsize=(10, 5))
    plt.plot(df_out.index, df_out["AdjClose"], label="NVDA Price (Adj Close)")
    plt.plot(df_out.index, df_out["SMA_90"], label="90-Day Moving Average")
    plt.title("NVDA Price vs 90-Day SMA (4 Years, excluding last 3 months)")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
