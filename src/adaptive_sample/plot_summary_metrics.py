#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILES = [
    "adaptive_sample_diff_small_lengths_with_grad_summary.csv",
    "adaptive_sample_acq_and_min_spacing_summary.csv",
    "adaptive_sample_diff_reward_funcs_summary.csv",
    "adaptive_sample_diff_small_lengths_summary.csv",
    "adaptive_sample_grad_weight_increment_summary.csv",
    "adaptive_sample_grad_weight_increment_summary.csv",
    "adaptive_sample_mean_weight_summary.csv",
    "adaptive_sample_mean_weight_with_grad_weight_summary.csv",
    "adaptive_sample_n_steps_summary.csv"
]

# Optional labels for distinguishing CSV groups in plots.
# Leave empty to infer labels from filenames.
CSV_LABELS = {
    # "results/summary_run_1.csv": "baseline",
    # "results/summary_run_2.csv": "adaptive A",
}

# Metrics you want to plot
METRICS = [
    "nrmse_mean"
]

# Parameters you want on the x-axis
PARAMETERS = [
    "n_candidates",
    "n_steps",
    "min_spacing",
    "w_mean",
    "w_std",
    "w_dist",
    "w_grad_mean",
]

OUTPUT_DIR = Path("metric_plots")


def load_all_csvs(csv_files):
    dfs = []

    for path in csv_files:
        path = Path(path)
        df = pd.read_csv(path)

        label = CSV_LABELS.get(str(path), path.stem)
        df["source_csv"] = label

        dfs.append(df)

    if not dfs:
        raise ValueError("CSV_FILES is empty.")

    return pd.concat(dfs, ignore_index=True)


def plot_metric_vs_param(df, metric, param, output_dir, metric_limits):
    if metric not in df.columns:
        print(f"Skipping {metric} vs {param}: missing metric column '{metric}'")
        return

    if param not in df.columns:
        print(f"Skipping {metric} vs {param}: missing parameter column '{param}'")
        return

    plot_df = df[[param, metric, "source_csv"]].dropna()

    if plot_df.empty:
        print(f"Skipping {metric} vs {param}: no valid data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, group in plot_df.groupby("source_csv"):
        grouped = (
            group.groupby(param, as_index=False)[metric]
            .mean()
            .sort_values(param)
        )

        ax.plot(
            grouped[param],
            grouped[metric],
            marker="o",
            label=label,
        )
    
    if metric in metric_limits:
        ax.set_ylim(*metric_limits[metric])

    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {param}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = output_dir / f"{metric}_vs_{param}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_csvs(CSV_FILES)

    metric_limits = {}

    for metric in METRICS:
        if metric not in df.columns:
            continue

        vals = df[metric].dropna()

        if len(vals) == 0:
            continue

        ymin = vals.min()
        ymax = vals.max()

        # Add 5% padding
        pad = 0.05 * (ymax - ymin)
        if pad == 0:
            pad = 0.01 * abs(ymax) if ymax != 0 else 1.0

        metric_limits[metric] = (ymin - pad, ymax + pad)
    
    for metric in METRICS:
        for param in PARAMETERS:
            plot_metric_vs_param(df, metric, param, OUTPUT_DIR, metric_limits=metric_limits)

if __name__ == "__main__":
    main()