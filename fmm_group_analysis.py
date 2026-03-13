#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FMM Group Project: NVDA Market Microstructure Analysis

Usage:
    python fmm_group_analysis.py --input NDVA_week.csv.gz --output-dir outputs
    python fmm_group_analysis.py --input NDVA_week.csv.gz --output-dir outputs --run-regression
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_white
    from statsmodels.stats.stattools import jarque_bera
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False


def parse_args():
    parser = argparse.ArgumentParser(description="NVDA market microstructure analysis")
    parser.add_argument("--input", required=True, help="Path to raw csv or csv.gz file")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Chunk size for pd.read_csv")
    parser.add_argument("--timezone", default="America/New_York", help="Timezone for timestamps")
    parser.add_argument("--start-time", default="09:30", help="Market open time")
    parser.add_argument("--end-time", default="16:00", help="Market close time")
    parser.add_argument("--run-regression", action="store_true", help="Run optional HAC regression")
    return parser.parse_args()


def weighted_avg(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def load_and_clean_data(
    input_path: str,
    chunk_size: int = 1_000_000,
    timezone: str = "America/New_York",
    start_time: str = "09:30",
    end_time: str = "16:00",
):
    use_cols = [
        "Date-Time",
        "Type",
        "Price",
        "Volume",
        "Bid Price",
        "Ask Price",
        "Tick Dir."
    ]

    chunks = []

    print(f"Loading file: {input_path}")
    for i, chunk in enumerate(pd.read_csv(input_path, usecols=use_cols, chunksize=chunk_size, low_memory=False)):
        print(f"Processing chunk {i + 1} ...")

        chunk["Date_Time"] = pd.to_datetime(chunk["Date-Time"], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["Date_Time"])
        chunk["Date_Time"] = chunk["Date_Time"].dt.tz_convert(timezone)
        chunk = chunk.set_index("Date_Time").sort_index()
        chunk = chunk.between_time(start_time, end_time)

        for col in ["Price", "Volume", "Bid Price", "Ask Price"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk[(chunk["Price"].isna()) | (chunk["Price"] > 0)]
        chunk = chunk[(chunk["Volume"].isna()) | (chunk["Volume"] >= 0)]

        chunks.append(chunk)

    df = pd.concat(chunks).sort_index()
    print("Shape after basic cleaning:", df.shape)

    print("\nType value counts:")
    print(df["Type"].value_counts(dropna=False))

    return df


def identify_trade_labels(df: pd.DataFrame):
    candidates = {"Trade", "TRADE", "Transaction", "TRANSACTION"}
    existing = set(df["Type"].dropna().astype(str).unique().tolist())
    matched = candidates & existing
    if matched:
        return matched

    common_names = [x for x in existing if "trade" in x.lower() or "trans" in x.lower()]
    if common_names:
        return set(common_names)

    raise ValueError(
        "Could not infer trade labels from Type column. "
        f"Observed values: {sorted(existing)}"
    )


def prepare_quotes_and_trades(df: pd.DataFrame):
    trade_labels = identify_trade_labels(df)
    print("\nUsing trade labels:", trade_labels)

    df["Best_Bid"] = df["Bid Price"].ffill()
    df["Best_Ask"] = df["Ask Price"].ffill()

    df = df[(df["Best_Bid"].notna()) & (df["Best_Ask"].notna())].copy()
    df = df[df["Best_Bid"] < df["Best_Ask"]].copy()

    df["Midpoint"] = (df["Best_Bid"] + df["Best_Ask"]) / 2
    df["Quoted_Spread_Dollar"] = df["Best_Ask"] - df["Best_Bid"]
    df["Quoted_Spread_Rel"] = np.where(
        df["Midpoint"] > 0,
        df["Quoted_Spread_Dollar"] / df["Midpoint"],
        np.nan
    )
    df["Quoted_Spread_bps"] = df["Quoted_Spread_Rel"] * 10000

    print("\nShape after quote cleaning:", df.shape)

    trades = df[df["Type"].isin(trade_labels)].copy()
    print("Trades before abnormal trade filter:", trades.shape)

    trades = trades[
        (trades["Price"].notna()) &
        (trades["Price"] >= trades["Best_Bid"]) &
        (trades["Price"] <= trades["Best_Ask"])
    ].copy()

    trades["Volume"] = trades["Volume"].fillna(0)
    trades["Dollar_Volume"] = trades["Price"] * trades["Volume"]

    print("Trades after abnormal trade filter:", trades.shape)

    trades["Direction"] = np.nan
    if "Tick Dir." in trades.columns:
        td = trades["Tick Dir."].astype(str).str.strip().str.lower()
        buy_set = {"buy", "b", "1", "+1", "up", "buyer", "buyer-initiated"}
        sell_set = {"sell", "s", "-1", "down", "seller", "seller-initiated"}

        trades.loc[td.isin(buy_set), "Direction"] = 1
        trades.loc[td.isin(sell_set), "Direction"] = -1

    trades.loc[trades["Direction"].isna() & (trades["Price"] > trades["Midpoint"]), "Direction"] = 1
    trades.loc[trades["Direction"].isna() & (trades["Price"] < trades["Midpoint"]), "Direction"] = -1

    trades["Effective_Spread_Dollar"] = 2 * np.abs(trades["Price"] - trades["Midpoint"])
    trades["Effective_Spread_Rel"] = np.where(
        trades["Midpoint"] > 0,
        trades["Effective_Spread_Dollar"] / trades["Midpoint"],
        np.nan
    )
    trades["Effective_Spread_bps"] = trades["Effective_Spread_Rel"] * 10000

    trades["Signed_Effective_Spread_Rel"] = np.where(
        trades["Direction"].notna() & (trades["Midpoint"] > 0),
        2 * trades["Direction"] * (trades["Price"] - trades["Midpoint"]) / trades["Midpoint"],
        np.nan
    )

    trades["LogPrice"] = np.log(trades["Price"])
    trades["LogReturn"] = trades["LogPrice"].diff()

    trades["Next_Midpoint"] = trades["Midpoint"].shift(-1)
    trades["Price_Impact_Rel"] = np.where(
        trades["Direction"].notna() &
        (trades["Midpoint"] > 0) &
        (trades["Next_Midpoint"].notna()),
        2 * trades["Direction"] * (trades["Next_Midpoint"] - trades["Midpoint"]) / trades["Midpoint"],
        np.nan
    )

    return df, trades


def build_interval_metrics(df_all: pd.DataFrame, trades_df: pd.DataFrame, freq: str = "5min"):
    all_part = df_all.resample(freq).agg(
        Message_Count=("Type", "count"),
        Avg_Quoted_Spread_Dollar=("Quoted_Spread_Dollar", "mean"),
        Median_Quoted_Spread_Dollar=("Quoted_Spread_Dollar", "median"),
        Avg_Quoted_Spread_Rel=("Quoted_Spread_Rel", "mean"),
        Avg_Quoted_Spread_bps=("Quoted_Spread_bps", "mean"),
        Avg_Midpoint=("Midpoint", "mean"),
        Midpoint_STD=("Midpoint", "std")
    )

    trade_part = trades_df.resample(freq).agg(
        Trade_Count=("Type", "count"),
        Total_Volume=("Volume", "sum"),
        Total_Dollar_Volume=("Dollar_Volume", "sum"),
        Avg_Trade_Price=("Price", "mean"),
        Avg_Effective_Spread_Dollar=("Effective_Spread_Dollar", "mean"),
        Median_Effective_Spread_Dollar=("Effective_Spread_Dollar", "median"),
        Avg_Effective_Spread_Rel=("Effective_Spread_Rel", "mean"),
        Avg_Effective_Spread_bps=("Effective_Spread_bps", "mean"),
        Avg_Signed_Effective_Spread_Rel=("Signed_Effective_Spread_Rel", "mean"),
        Avg_Price_Impact_Rel=("Price_Impact_Rel", "mean"),
        Return_STD=("LogReturn", "std")
    )

    vw_eff_dollar = trades_df.groupby(pd.Grouper(freq=freq)).apply(
        lambda g: weighted_avg(g["Effective_Spread_Dollar"], g["Volume"])
    )
    vw_eff_dollar.name = "VW_Effective_Spread_Dollar"

    vw_eff_rel = trades_df.groupby(pd.Grouper(freq=freq)).apply(
        lambda g: weighted_avg(g["Effective_Spread_Rel"], g["Volume"])
    )
    vw_eff_rel.name = "VW_Effective_Spread_Rel"

    vw_eff_bps = trades_df.groupby(pd.Grouper(freq=freq)).apply(
        lambda g: weighted_avg(g["Effective_Spread_bps"], g["Volume"])
    )
    vw_eff_bps.name = "VW_Effective_Spread_bps"

    interval_ret = trades_df["Price"].resample(freq).apply(
        lambda x: np.log(x.iloc[-1] / x.iloc[0]) if len(x) >= 2 and x.iloc[0] > 0 and x.iloc[-1] > 0 else np.nan
    )
    interval_ret.name = "Interval_Log_Return"

    rv = trades_df["LogReturn"].resample(freq).apply(
        lambda x: np.sqrt(np.nansum(np.square(x.dropna())))
    )
    rv.name = "Realized_Volatility"

    out = all_part.join(trade_part, how="outer")
    out = out.join(vw_eff_dollar, how="outer")
    out = out.join(vw_eff_rel, how="outer")
    out = out.join(vw_eff_bps, how="outer")
    out = out.join(interval_ret, how="outer")
    out = out.join(rv, how="outer")

    out["AT_Proxy"] = np.where(
        out["Trade_Count"] > 0,
        out["Message_Count"] / out["Trade_Count"],
        np.nan
    )

    out["Avg_Trade_Size"] = np.where(
        out["Trade_Count"] > 0,
        out["Total_Volume"] / out["Trade_Count"],
        np.nan
    )

    out["Amihud_ILLIQ"] = np.where(
        out["Total_Dollar_Volume"] > 0,
        np.abs(out["Interval_Log_Return"]) / out["Total_Dollar_Volume"],
        np.nan
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def save_summary(df: pd.DataFrame, path: Path):
    df.describe(include="all").to_csv(path)


def make_plots(metrics_5m: pd.DataFrame, metrics_10m: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(14, 6))
    plt.plot(metrics_5m.index, metrics_5m["Avg_Quoted_Spread_Dollar"], label="5m Avg Quoted Spread ($)")
    plt.plot(metrics_5m.index, metrics_5m["Avg_Effective_Spread_Dollar"], label="5m Avg Effective Spread ($)")
    plt.xlabel("Time")
    plt.ylabel("Spread")
    plt.title("5-Minute Quoted Spread vs Effective Spread")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_5m_spreads.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(metrics_10m.index, metrics_10m["Avg_Quoted_Spread_bps"], label="10m Quoted Spread (bps)")
    plt.plot(metrics_10m.index, metrics_10m["Avg_Effective_Spread_bps"], label="10m Effective Spread (bps)")
    plt.xlabel("Time")
    plt.ylabel("bps")
    plt.title("10-Minute Quoted Spread vs Effective Spread")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_10m_spreads_bps.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(metrics_5m.index, metrics_5m["Amihud_ILLIQ"], label="Amihud ILLIQ")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amihud ILLIQ")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(metrics_5m.index, metrics_5m["AT_Proxy"], linestyle="--", label="AT Proxy")
    ax2.set_ylabel("AT Proxy")

    plt.title("5-Minute Liquidity vs AT Activity")
    fig.tight_layout()
    plt.savefig(output_dir / "plot_5m_illiquidity_vs_at.png", dpi=200, bbox_inches="tight")
    plt.close()


def run_regression(metrics_5m: pd.DataFrame, output_dir: Path):
    if not STATSMODELS_OK:
        print("statsmodels not available; regression skipped.")
        return

    dep_var = "Avg_Effective_Spread_bps"
    reg_df = metrics_5m[[dep_var, "AT_Proxy", "Realized_Volatility", "Total_Dollar_Volume", "Trade_Count"]].dropna().copy()
    reg_df["Log_Dollar_Volume"] = np.log1p(reg_df["Total_Dollar_Volume"])
    reg_df["Log_Trade_Count"] = np.log1p(reg_df["Trade_Count"])

    X = reg_df[["AT_Proxy", "Realized_Volatility", "Log_Dollar_Volume", "Log_Trade_Count"]]
    X = sm.add_constant(X)
    y = reg_df[dep_var]

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    with open(output_dir / "regression_summary.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
        f.write("\n\nDiagnostics:\n")
        resid = model.resid

        jb_stat, jb_pvalue, skew, kurt = jarque_bera(resid)
        f.write(f"Jarque-Bera p-value: {jb_pvalue:.6f}\n")

        white = het_white(resid, X)
        f.write(f"White test p-value: {white[1]:.6f}\n")

    plt.figure(figsize=(10, 4))
    plt.plot(model.resid.values)
    plt.title("Regression Residuals")
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "regression_residuals.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Regression completed.")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data(
        input_path=args.input,
        chunk_size=args.chunk_size,
        timezone=args.timezone,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    df, trades = prepare_quotes_and_trades(df)

    metrics_5m = build_interval_metrics(df, trades, freq="5min")
    metrics_10m = build_interval_metrics(df, trades, freq="10min")

    metrics_5m.to_csv(output_dir / "NVDA_market_quality_5min.csv")
    metrics_10m.to_csv(output_dir / "NVDA_market_quality_10min.csv")

    save_summary(metrics_5m, output_dir / "summary_5min.csv")
    save_summary(metrics_10m, output_dir / "summary_10min.csv")

    make_plots(metrics_5m, metrics_10m, output_dir)

    metrics_5m.isna().sum().sort_values(ascending=False).to_csv(output_dir / "missing_5min.csv")
    metrics_10m.isna().sum().sort_values(ascending=False).to_csv(output_dir / "missing_10min.csv")

    if args.run_regression:
        run_regression(metrics_5m, output_dir)

    print("\nDone.")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
