import pandas as pd
import yfinance as yf

today = pd.Timestamp.today().normalize()

df = yf.download(
    "NVDA",
    start=fetch_start.date(),
    end=(today + pd.Timedelta(days=1)).date(),
    interval="1d",
    auto_adjust=True,
    progress=False
)

if df.empty:
    print("No data found for NVDA.")
else:
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    else:
        df = df.rename(columns={"Close": "AdjClose"})

    df["SMA_90"] = df["AdjClose"].rolling(window=90, min_periods=90).mean()
    df["SMA_30"] = df["AdjClose"].rolling(window=30, min_periods=30).mean()
    df["SMA_10"] = df["AdjClose"].rolling(window=10, min_periods=10).mean()

    df["Label"] = (df["AdjClose"].shift(-1) > df["AdjClose"]).astype(int)

    df_out = df.loc[df.index >= output_start,
                    ["AdjClose", "SMA_90", "SMA_30", "SMA_10", "Label"]].dropna()

    df_out.to_csv("nvda_last3mo.txt", sep=" ", header=False, index=False, float_format="%.6f")

    print(f"Saved {len(df_out)} rows to nvda_last3mo.txt")
