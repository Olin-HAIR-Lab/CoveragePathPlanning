#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_bool_series(s: pd.Series) -> pd.Series:
    """Handle bools or strings like 'True', 'false', '1', etc."""
    if s.dtype == bool:
        return s

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def load_runs(csv_dir: Path) -> pd.DataFrame:
    rows = []

    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)

        required = {"dist", "nrmse", "path", "presample"}
        missing = required - set(df.columns)
        if missing:
            print(f"Skipping {csv_path.name}: missing {missing}")
            continue

        df = df.copy()
        df["presample"] = parse_bool_series(df["presample"])
        df["presample_count"] = int(df["presample"].sum())
        df["run_name"] = csv_path.stem
        df["source_csv"] = str(csv_path)

        rows.append(df)

    if not rows:
        raise RuntimeError(f"No valid CSV files found in {csv_dir}")

    return pd.concat(rows, ignore_index=True)


def plot_all_runs(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10, 6))

    for (run_name, presample_count), g in df.groupby(["run_name", "presample_count"]):
        g = g.sort_values("dist")
        plt.plot(
            g["dist"],
            g["nrmse"],
            alpha=0.45,
            drawstyle="steps-post",
            label=f"{run_name}, presample={presample_count}",
        )

    plt.xlabel("Distance traveled")
    plt.ylabel("NRMSE")
    plt.title("NRMSE over distance, individual runs")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_averaged_by_presample(df: pd.DataFrame, out_path: Path, bins: int):
    """
    Average runs with the same presample count.

    Because different runs may have different distance values, this bins distance
    and averages NRMSE within each bin.
    """
    df = df.copy()

    min_dist = df["dist"].min()
    max_dist = df["dist"].max()

    df["dist_bin"] = pd.cut(
        df["dist"],
        bins=bins,
        labels=False,
        include_lowest=True,
    )

    bin_centers = (
        df.groupby("dist_bin")["dist"]
        .mean()
        .rename("dist_center")
        .reset_index()
    )

    avg = (
        df.groupby(["presample_count", "dist_bin"], as_index=False)
        .agg(
            nrmse_mean=("nrmse", "mean"),
            nrmse_std=("nrmse", "std"),
            n=("nrmse", "count"),
        )
        .merge(bin_centers, on="dist_bin", how="left")
        .sort_values(["presample_count", "dist_center"])
    )

    plt.figure(figsize=(10, 6))

    for presample_count, g in avg.groupby("presample_count"):
        line, = plt.step(
            g["dist_center"],
            g["nrmse_mean"],
            where="post",
            marker="o",
            label=f"{presample_count} presampled points",
        )

        plt.fill_between(
            g["dist_center"],
            g["nrmse_mean"] - g["nrmse_std"],
            g["nrmse_mean"] + g["nrmse_std"],
            step="post",
            alpha=0.2,
            color=line.get_color(),
        )

    plt.xlabel("Distance traveled")
    plt.ylabel("Mean NRMSE")
    plt.title("Average NRMSE over distance by presampling amount")
    plt.ylim([0.1,0.25])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir", type=Path, help="Directory containing simulation CSV files")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    csv_dir = args.csv_dir
    out_dir = args.out_dir or csv_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(csv_dir)

    plot_all_runs(df, out_dir / "nrmse_over_distance_all_runs.png")
    plot_averaged_by_presample(
        df,
        out_dir / "nrmse_over_distance_avg_by_presample.png",
        bins=args.bins,
    )

    print("Saved:")
    print(out_dir / "nrmse_over_distance_all_runs.png")
    print(out_dir / "nrmse_over_distance_avg_by_presample.png")


if __name__ == "__main__":
    main()