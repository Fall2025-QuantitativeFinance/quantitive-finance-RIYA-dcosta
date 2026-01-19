import pandas as pd
import yfinance as yf

YEARS = 5
ETF_TICKER = "QQQ"


def load_holdings(path="data/holdings.csv"):
    rows = []
    start = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # detect start of actual table
            if line.lower().startswith("holding,symbol"):
                start = True
                continue

            if not start or not line:
                continue

            parts = line.rsplit(",", 2)
            if len(parts) != 3:
                continue

            name, symbol, weight = parts
            symbol = symbol.strip().upper()
            weight = weight.replace("%", "").strip()

            try:
                weight = float(weight)
            except ValueError:
                continue

            rows.append({"Symbol": symbol, "Weight": weight})

    return pd.DataFrame(rows)


def main():
    print("Loading holdings...")
    holdings = load_holdings()
    symbols = holdings["Symbol"].unique().tolist()

    # add ETF itself
    symbols = [ETF_TICKER] + [s for s in symbols if s != ETF_TICKER]

    print("Total symbols:", len(symbols))

    print("Downloading prices...")
    data = yf.download(
        symbols,
        period=f"{YEARS}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    close = data["Close"].reset_index()
    prices = close.melt(id_vars="Date", var_name="symbol", value_name="close").dropna()

    prices.rename(columns={"Date": "date"}, inplace=True)
    prices.to_csv("data/prices_daily.csv", index=False)

    print("Saved: data/prices_daily.csv")
    print(prices.head())


if __name__ == "__main__":
    main()
