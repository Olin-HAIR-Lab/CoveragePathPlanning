import sys
import pandas as pd


CONFIG_COLS = [
    "presample_pts",
    "n_steps",
    "budget",
    "n_candidates",
    "min_spacing",
    "min_length_scale",
    "w_mean",
    "w_std",
    "w_dist",
    "w_grad_mean",
    "w_far_from_mean",
]

METRIC_COLS = [
    "rmse",
    "nrmse",
    "rmse_over_std",
    "length_scale",
    "num_pts",
    "data_range",
    "data_std",
    "region_area",
]

# Used to identify which dataset/region a run came from.
DATASET_COL_CANDIDATES = [
    "path"
]


def add_relative_scores(df, metric_cols, dataset_cols):
    """
    Adds within-dataset relative scores.

    For each dataset:
      metric_rel_best = metric / best metric on that dataset
      metric_centered = metric - dataset mean metric
      metric_z = centered / dataset std
      metric_rank = rank of this run within that dataset
    """

    group_key = dataset_cols

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        dataset_best = df.groupby(group_key, dropna=False)[metric].transform("min")
        dataset_mean = df.groupby(group_key, dropna=False)[metric].transform("mean")
        dataset_std = df.groupby(group_key, dropna=False)[metric].transform("std")

        df[f"{metric}_rel_best"] = df[metric] / dataset_best
        df[f"{metric}_centered"] = df[metric] - dataset_mean
        df[f"{metric}_z"] = (df[metric] - dataset_mean) / dataset_std
        df[f"{metric}_rank"] = df.groupby(group_key, dropna=False)[metric].rank(
            method="average",
            ascending=True,
        )

    return df


def main(csv_path):
    df = pd.read_csv(csv_path)

    config_cols = [c for c in CONFIG_COLS if c in df.columns]
    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    dataset_cols = [c for c in DATASET_COL_CANDIDATES if c in df.columns]

    if not dataset_cols:
        raise ValueError(
            "Could not find a dataset/region identifier column. "
            "Add one of these columns to DATASET_COL_CANDIDATES: "
            f"{DATASET_COL_CANDIDATES}"
        )

    # Convert numeric columns
    for col in config_cols + metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add per-run relative scores before aggregation
    df = add_relative_scores(df, metric_cols, dataset_cols)

    # Include original metrics plus new relative metrics in summary
    relative_metric_cols = [
        c
        for c in df.columns
        if any(
            c.startswith(f"{metric}_")
            for metric in metric_cols
        )
        and c not in metric_cols
    ]

    summary_metric_cols = metric_cols + relative_metric_cols

    grouped = (
        df.groupby(config_cols, dropna=False)[summary_metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    # Prefer relative score for sorting, falling back to raw nrmse
    sort_col = (
        "nrmse_rel_best_mean"
        if "nrmse_rel_best_mean" in grouped.columns
        else "nrmse_mean"
    )
    grouped = grouped.sort_values(sort_col)

    output_path = csv_path.replace(".csv", "_summary.csv")
    grouped.to_csv(output_path, index=False)

    print(f"Saved summary to {output_path}")
    print()
    print(f"Top 20 configurations by {sort_col}:")

    display_cols = [
        c for c in config_cols
        + [
            "nrmse_mean",
            "nrmse_std",
            "nrmse_rel_best_mean",
            "nrmse_rel_best_std",
            "nrmse_centered_mean",
            "nrmse_z_mean",
            "nrmse_rank_mean",
            "rmse_mean",
            "rmse_rel_best_mean",
            "rmse_over_std_mean",
            "rmse_over_std_rel_best_mean",
            "length_scale_mean",
            "num_pts_mean",
            "nrmse_count",
        ]
        if c in grouped.columns
    ]

    print(grouped[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])