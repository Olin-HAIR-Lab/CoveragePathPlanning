from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import os
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
from pyvrp.exceptions import PenaltyBoundWarning
from adaptive_sample import SimulationConfig, run_simulation

# These warnings sometimes indicate a poor choice of parameters, but
# we see that in the poor metrics as well
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=PenaltyBoundWarning)

def run_one(args):
    data_path, ns, nc, wd, seed, min_ls = args

    config = SimulationConfig(
        data_path=data_path,
        make_plots=False,
        seed=seed,
        n_steps=ns,
        n_candidates=nc,
        dist_weight = wd,
        min_length_scale=min_ls
    )
    return run_simulation(config)

def main():
    data_paths = [
        "../scripts/region_previews_farm12/region12_data_filtered.gpkg",
        "../scripts/region_previews_farm12/region16_data_filtered.gpkg",
        "../scripts/region_previews_farm12/region33_data_filtered.gpkg",
        "../scripts/region_previews_farm12/region34_data_filtered.gpkg",

        "../scripts/region_previews_farm03/region20_data_filtered.gpkg",
        "../scripts/region_previews_farm03/region22_data_filtered.gpkg",
        "../scripts/region_previews_farm03/region24_data_filtered.gpkg",
        "../scripts/region_previews_farm03/region15_data_filtered.gpkg",

        "../scripts/region_previews_farm12/region5_data_filtered.gpkg",
        "../scripts/region_previews_farm12/region6_data_filtered.gpkg",
        "../scripts/region_previews_farm12/region7_data_filtered.gpkg",
        "../scripts/region_previews_farm12/region10_data_filtered.gpkg",

        "../scripts/region_previews_farm03/region9_data_filtered.gpkg",
        "../scripts/region_previews_farm03/region10_data_filtered.gpkg",
        "../scripts/region_previews_farm03/region11_data_filtered.gpkg",
        "../scripts/region_previews_farm03/region12_data_filtered.gpkg",
    ]

    n_steps = [3]
    n_candidates = [10]
    n_trials = 10
    dist_weights = [0, 0.0005, 0.001, 0.005, 0.01]
    #dist_weights = [0.001]
    #length_scale_mins = [0.1, 5.0, 10.0, 15.0, 20.0]
    length_scale_mins = [10.0]

    out_path = "adaptive_sample_dist_weights.csv"

    jobs = [
        (data_path, ns, nc, wd, seed, min_ls)
        for data_path, ns, nc, wd, min_ls in product(
            data_paths, n_steps, n_candidates, dist_weights, length_scale_mins
        )
        for seed in range(n_trials)
    ]

    results = []

    max_workers = max(1, os.cpu_count() - 1)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]

        for k, future in enumerate(as_completed(futures), start=1):
            try:
                single_df = future.result()
                results.append(single_df)
                percent_done = 100 * k / len(jobs)
                time_to_now = time.time() - start_time
                time_per_job = time_to_now / k
                est_time_remaining = (len(jobs) - k) * time_per_job

                print(f"{percent_done:.1f}% complete ({k}/{len(jobs)}) in {time_to_now:.1f}; estimating {est_time_remaining // 60:.0f}:{est_time_remaining % 60:.0f} remaining.")

                # Save partial progress each completed run
                pd.concat(results, ignore_index=True).to_csv(out_path, index=False)

            except Exception as e:
                print(f"Run failed: {e}")

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(out_path, index=False)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()