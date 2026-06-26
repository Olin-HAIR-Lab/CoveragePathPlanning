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


def main(csv_path):
    df = pd.read_csv(csv_path)

    # Keep only columns that exist, in case some are missing
    config_cols = [c for c in CONFIG_COLS if c in df.columns]
    metric_cols = [c for c in METRIC_COLS if c in df.columns]

    # Convert numeric columns
    for col in config_cols + metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(config_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    # Flatten multi-index columns
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    grouped = grouped.sort_values("nrmse_mean")

    output_path = csv_path.replace(".csv", "_summary.csv")
    grouped.to_csv(output_path, index=False)

    print(f"Saved summary to {output_path}")
    print()
    print("Top 20 configurations by mean NRMSE:")
    print(
        grouped[
            config_cols
            + [
                "nrmse_mean",
                "nrmse_std",
                "rmse_mean",
                "rmse_over_std_mean",
                "length_scale_mean",
                "num_pts_mean",
                "nrmse_count",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main(sys.argv[1])