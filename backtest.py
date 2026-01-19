import pandas as pd
import matplotlib.pyplot as plt

ETF = "QQQ"
LONG_N = 15
SHORT_N = 15


def make_monthly(prices):
    df = prices.copy()

    # clean columns
    df.columns = df.columns.str.lower().str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].str.upper()

    # create month-end
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    # keep last price of each month
    df = df.sort_values(["symbol", "date"])
    monthly = df.groupby(["symbol", "month"]).last().reset_index()

    monthly = monthly.drop(columns=["date"])
    monthly = monthly.rename(columns={"month": "date"})

    return monthly[["symbol", "date", "close"]]


def add_momentum(df):
    df = df.sort_values(["symbol", "date"]).copy()

    df["ret_1m"] = df.groupby("symbol")["close"].pct_change()
    df["mom_3m"] = df.groupby("symbol")["close"].pct_change(3)
    df["mom_6m"] = df.groupby("symbol")["close"].pct_change(6)
    df["mom_12m"] = df.groupby("symbol")["close"].pct_change(12)

    df["momentum"] = df[["mom_3m", "mom_6m", "mom_12m"]].mean(axis=1)

    return df


def backtest(df):
    df = df.sort_values(["symbol", "date"]).copy()
    df["next_return"] = df.groupby("symbol")["ret_1m"].shift(-1)

    etf = df[df["symbol"] == ETF][["date", "next_return"]]
    stocks = df[df["symbol"] != ETF]

    rows = []

    for date, group in stocks.groupby("date"):
        group = group.dropna(subset=["momentum", "next_return"])
        if len(group) < LONG_N + SHORT_N:
            continue

        group = group.sort_values("momentum", ascending=False)

        longs = group.head(LONG_N)
        shorts = group.tail(SHORT_N)

        rows.append(
            {
                "date": date,
                "long_return": longs["next_return"].mean(),
                "short_return": shorts["next_return"].mean(),
                "long_short_return": longs["next_return"].mean()
                - shorts["next_return"].mean(),
            }
        )

    result = pd.DataFrame(rows)
    result = result.merge(etf, on="date", how="left")
    result = result.rename(columns={"next_return": "etf_return"})

    result["ls_cumulative"] = (1 + result["long_short_return"]).cumprod() - 1
    result["etf_cumulative"] = (1 + result["etf_return"].fillna(0)).cumprod() - 1

    return result


import matplotlib.pyplot as plt


def plot_charts(bt):
    bt = bt.dropna().copy()
    bt["date"] = pd.to_datetime(bt["date"])

    # CHART 1: Monthly long-short bar vs ETF line
    plt.figure(figsize=(10, 5))
    colors = ["green" if x >= 0 else "red" for x in bt["long_short_return"]]
    plt.bar(bt["date"], bt["long_short_return"], color=colors, label="Long-Short")
    plt.plot(bt["date"], bt["etf_return"], marker="o", label="ETF")
    plt.title("Monthly Long-Short Return (bars) vs ETF (line)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # CHART 2: Monthly long vs short vs ETF
    plt.figure(figsize=(10, 5))
    plt.plot(bt["date"], bt["long_return"], label="Long basket")
    plt.plot(bt["date"], bt["short_return"], label="Short basket")
    plt.plot(bt["date"], bt["etf_return"], label="ETF")
    plt.title("Monthly Returns: Long vs Short vs ETF")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # CHART 3: Cumulative long-short vs ETF
    plt.figure(figsize=(10, 5))
    plt.plot(bt["date"], bt["ls_cumulative"], label="Long-Short cumulative")
    plt.plot(bt["date"], bt["etf_cumulative"], label="ETF cumulative")
    plt.title("Cumulative Return: Long-Short vs ETF")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show ALL figures
    plt.show()


def main():
    prices = pd.read_csv("data/prices_daily.csv")

    monthly = make_monthly(prices)
    monthly.to_csv("data/monthly_backtest.csv", index=False)
    print("Saved monthly_backtest.csv")

    monthly = add_momentum(monthly)
    bt = backtest(monthly)

    bt.to_csv("data/backtest_monthly.csv", index=False)
    print("Saved backtest_monthly.csv")
    print(bt.tail())

    plot_charts(bt)


if __name__ == "__main__":
    main()
