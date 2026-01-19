import pandas as pd


def load_holdings(path="data/holdings.csv") -> pd.DataFrame:
    """
    holdings.csv lines look like:
      Amazon.com, Inc.,AMZN,5.0756%
    Name can contain commas, so we split from the RIGHT into 3 parts.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue

            # Split from right: [name..., symbol, weight]
            parts = ln.rsplit(",", 2)
            if len(parts) != 3:
                continue

            name, symbol, weight = parts
            rows.append([name.strip(), symbol.strip(), weight.strip()])

    df = pd.DataFrame(rows, columns=["Name", "Symbol", "Weight"])

    # Clean weight: remove % and convert to number
    df["Weight"] = (
        df["Weight"].astype(str).str.replace("%", "", regex=False).str.strip()
    )
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    # Clean symbol
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()

    # Keep only real rows
    df = df.dropna(subset=["Symbol", "Weight"])
    return df


def to_months(period: str) -> int:
    period = period.strip().lower()
    if period.endswith("m"):
        return int(period[:-1])
    raise ValueError("Period must look like '3m', '6m', '12m'.")


def calc_momentum(close_series: pd.Series, months: int) -> pd.Series:
    # approx 21 trading days per month
    lookback = months * 21
    return (close_series / close_series.shift(lookback)) - 1


def main():
    print("momentum_calc starting...")

    # ---- holdings ----
    holdings: pd.DataFrame = load_holdings("data/holdings.csv")

    # ---- prices ----
    prices = pd.read_csv("data/prices.csv")
    prices.columns = prices.columns.str.strip().str.lower()

    required = {"symbol", "date", "close"}
    if not required.issubset(prices.columns):
        raise ValueError(
            f"prices.csv must have columns {required}. Found: {prices.columns.tolist()}"
        )

    prices["symbol"] = prices["symbol"].astype(str).str.strip().str.upper()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["symbol", "date", "close"]).sort_values(
        ["symbol", "date"]
    )

    windows = ["3m", "6m", "12m"]

    # compute latest momentum per symbol
    out = []
    for sym, g in prices.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        for w in windows:
            m = to_months(w)
            g[w + "_mom"] = calc_momentum(g["close"], m)

        last = g.iloc[-1][["symbol", "date"] + [w + "_mom" for w in windows]]
        out.append(last)

    mom_df = pd.DataFrame(out)

    # ---- merge ----
    merged = mom_df.merge(
        holdings.rename(
            columns={"Symbol": "symbol", "Weight": "weight", "Name": "name"}
        ),
        on="symbol",
        how="inner",
    )

    if merged.empty:
        print(
            "WARNING: merged is empty (symbols in holdings don’t match symbols in prices)."
        )

    # ---- portfolio weighted momentum (weighted average) ----
    print("\n---- Portfolio Momentum (weighted avg) ----")
    portfolio = {}

    for w in windows:
        valid = merged.dropna(subset=[w + "_mom", "weight"])
        if valid.empty:
            portfolio[w] = pd.NA
        else:
            portfolio[w] = (valid[w + "_mom"] * valid["weight"]).sum() / valid[
                "weight"
            ].sum()
        print(f"{w}: {portfolio[w]}")

    # save outputs
    merged.to_csv("data/momentum_results.csv", index=False)
    print("\nSaved: data/momentum_results.csv")
    print("\nPreview:")
    print(merged.head())


if __name__ == "__main__":
    main()
