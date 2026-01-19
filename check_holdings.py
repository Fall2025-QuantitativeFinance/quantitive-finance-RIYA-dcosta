import pandas as pd

HOLDINGS_PATH = "data/holdings_clean.csv"
PRICES_PATH = "data/prices.csv"

LOOKBACKS = {"mom_3m": 63, "mom_6m": 126, "mom_12m": 252}

N_PICK = 10


def main():
    # ---- holdings ----
    hold = pd.read_csv(HOLDINGS_PATH)
    hold.columns = hold.columns.str.strip().str.lower()
    hold["symbol"] = hold["symbol"].astype(str).str.strip().str.upper()

    # ---- prices (ROBUST LOAD) ----
    px = pd.read_csv(PRICES_PATH, sep=",", engine="python")
    px.columns = px.columns.str.strip().str.lower()
    px["symbol"] = px["symbol"].astype(str).str.strip().str.upper()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["symbol", "date"])

    # keep only symbols in holdings
    px = px[px["symbol"].isin(hold["symbol"])]

    # ---- momentum ----
    g = px.groupby("symbol", group_keys=False)
    for name, days in LOOKBACKS.items():
        px[name] = g["close"].apply(lambda s: s / s.shift(days) - 1)

    last = px.groupby("symbol").tail(1)

    last["mom_avg"] = last[list(LOOKBACKS.keys())].mean(axis=1)
    last["z"] = (last["mom_avg"] - last["mom_avg"].mean()) / last["mom_avg"].std(ddof=0)

    last = last.sort_values("z", ascending=False)

    print("\nLONG:")
    print(last.head(N_PICK)[["symbol", "z"]])

    print("\nSHORT:")
    print(last.tail(N_PICK)[["symbol", "z"]])

    last.to_csv("data/signals_latest.csv", index=False)
    print("\nSaved data/signals_latest.csv")


if __name__ == "__main__":
    main()
