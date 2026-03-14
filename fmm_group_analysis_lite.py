#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FMM Group Project - Lite Version

Faster version for large NVDA tick-by-tick data.
Keeps the core workflow:
1. Load raw data
2. Clean data
3. Construct prevailing bid/ask
4. Build 5-minute and 10-minute liquidity/market quality metrics
5. Save csv outputs and a few plots

Usage:
    python fmm_group_analysis_lite.py --input NDVA_week.csv.gz --output-dir outputs
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Fast NVDA market microstructure analysis")
    parser.add_argument("--input", required=True, help="Path to raw csv or csv.gz file")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Chunk size for pd.read_csv")
    parser.add_argument("--timezone", default="America/New_York", help="Timezone for timestamps")
    parser.add_argument("--start-time", default="09:30", help="Market open time")
    parser.add_argument("--end-time", default="16:00", help="Market close time")
    return parser.parse_args()


def weighted_avg(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def load_data(input_path, chunk_size, timezone, start_time, end_time):
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
        print(f"Processing chunk {i+1} ...")

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


def infer_trade_labels(df):
    candidates = {"Trade", "TRADE", "Transaction", "TRANSACTION"}
    existing = set(df["Type"].dropna().astype(str).unique().tolist())
    matched = candidates & existing
    if matched:
        return matched

    common_names = [x for x in existing if "trade" in x.lower() or "trans" in x.lower()]
    if common_names:
        return set(common_names)

    raise ValueError(f"Could not infer trade labels. Observed Type values: {sorted(existing)}")


def prepare_data(df):
    trade_labels = infer_trade_labels(df)
    print("\nUsing trade labels:", trade_labels)

    # prevailing quotes
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

    print("Shape after quote cleaning:", df.shape)

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

    # effective spread
    trades["Effective_Spread_Dollar"] = 2 * np.abs(trades["Price"] - trades["Midpoint"])
    trades["Effective_Spread_Rel"] = np.where(
        trades["Midpoint"] > 0,
        trades["Effective_Spread_Dollar"] / trades["Midpoint"],
        np.nan
    )
    trades["Effective_Spread_bps"] = trades["Effective_Spread_Rel"] * 10000

    # returns
    trades["LogPrice"] = np.log(trades["Price"])
    trades["LogReturn"] = trades["LogPrice"].diff()

    return df, trades


def build_metrics(df_all, trades_df, freq="5min"):
    all_part = df_all.resample(freq).agg(
        Message_Count=("Type", "count"),
        Avg_Quoted_Spread_Dollar=("Quoted_Spread_Dollar", "mean"),
        Avg_Quoted_Spread_Rel=("Quoted_Spread_Rel", "mean"),
        Avg_Quoted_Spread_bps=("Quoted_Spread_bps", "mean"),
        Avg_Midpoint=("Midpoint", "mean")
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
        Return_STD=("LogReturn", "std")
    )

    vw_eff_dollar = trades_df.groupby(pd.Grouper(freq=freq)).apply(
        lambda g: weighted_avg(g["Effective_Spread_Dollar"], g["Volume"])
    )
    vw_eff_dollar.name = "VW_Effective_Spread_Dollar"

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


def make_plots(metrics_5m, metrics_10m, output_dir):
    plt.figure(figsize=(12, 5))
    plt.plot(metrics_5m.index, metrics_5m["Avg_Quoted_Spread_Dollar"], label="5m Quoted Spread")
    plt.plot(metrics_5m.index, metrics_5m["Avg_Effective_Spread_Dollar"], label="5m Effective Spread")
    plt.xlabel("Time")
    plt.ylabel("Spread ($)")
    plt.title("5-Minute Spreads")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_5m_spreads.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(metrics_10m.index, metrics_10m["Avg_Quoted_Spread_bps"], label="10m Quoted Spread (bps)")
    plt.plot(metrics_10m.index, metrics_10m["Avg_Effective_Spread_bps"], label="10m Effective Spread (bps)")
    plt.xlabel("Time")
    plt.ylabel("bps")
    plt.title("10-Minute Spreads (bps)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_10m_spreads_bps.png", dpi=180)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(
        args.input,
        args.chunk_size,
        args.timezone,
        args.start_time,
        args.end_time
    )

    df, trades = prepare_data(df)

    metrics_5m = build_metrics(df, trades, freq="5min")
    metrics_10m = build_metrics(df, trades, freq="10min")

    metrics_5m.to_csv(output_dir / "NVDA_market_quality_5min.csv")
    metrics_10m.to_csv(output_dir / "NVDA_market_quality_10min.csv")

    metrics_5m.describe().to_csv(output_dir / "summary_5min.csv")
    metrics_10m.describe().to_csv(output_dir / "summary_10min.csv")

    metrics_5m.isna().sum().sort_values(ascending=False).to_csv(output_dir / "missing_5min.csv")
    metrics_10m.isna().sum().sort_values(ascending=False).to_csv(output_dir / "missing_10min.csv")

    make_plots(metrics_5m, metrics_10m, output_dir)

    print("\nDone.")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
