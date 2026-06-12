import numpy as np
import geopandas as gpd
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.affinity import translate
import numpy as np
import pandas as pd

def random_points_in_polygon(poly, n, seed=None):
    rng = np.random.default_rng(seed)

    minx, miny, maxx, maxy = poly.bounds
    points = []

    while len(points) < n:
        p = Point(
            rng.uniform(minx, maxx),
            rng.uniform(miny, maxy)
        )

        if poly.contains(p):
            points.append(p)

    #print(f"random points: {points}")
    return points

def make_rbf_component(name, length_scale, bounds, aniso=False):
    if aniso:
        return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=[length_scale, length_scale],
            length_scale_bounds=[bounds, bounds]
        )
    else:
        return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=length_scale,
            length_scale_bounds=bounds
        )


def extract_kernel_info(gp, noise, aniso, two_RBF):
    params = gp.kernel_.get_params()

    info = {
        "kernel": str(gp.kernel_),
        "length_scales": {},
        "magnitudes": {},
        "noise_level": None,
    }

    if two_RBF:
        # Kernel structure:
        # (long + short) [+ noise]
        if noise:
            long_prefix = "k1__k1"
            short_prefix = "k1__k2"
            noise_key = "k2__noise_level"
        else:
            long_prefix = "k1"
            short_prefix = "k2"
            noise_key = None

        info["magnitudes"]["long"] = params[f"{long_prefix}__k1__constant_value"]
        info["magnitudes"]["short"] = params[f"{short_prefix}__k1__constant_value"]

        info["length_scales"]["long"] = params[f"{long_prefix}__k2__length_scale"]
        info["length_scales"]["short"] = params[f"{short_prefix}__k2__length_scale"]

    else:
        # Kernel structure:
        # single [+ noise]
        if noise:
            rbf_prefix = "k1"
            noise_key = "k2__noise_level"
        else:
            rbf_prefix = ""
            noise_key = None

        if noise:
            info["magnitudes"]["single"] = params[f"{rbf_prefix}__k1__constant_value"]
            info["length_scales"]["single"] = params[f"{rbf_prefix}__k2__length_scale"]
        else:
            info["magnitudes"]["single"] = params["k1__constant_value"]
            info["length_scales"]["single"] = params["k2__length_scale"]

    if noise:
        info["noise_level"] = params[noise_key]

    return info


def fit_gp_flexible(
    coords_input,
    points_input,
    region_input,
    n_synth=0,
    noise=True,
    aniso=False,
    two_RBF=False,
    gui=True
):
    """
    Fits a GP using optional:
      - WhiteKernel noise
      - anisotropic RBF length scales
      - one or two RBF components

    Returns:
      mean_pred, std_pred, kernel_info, rmse, nrmse, nrmse_std
    """

    # Translate everything so that home is 0,0
    home = np.array(coords_input).squeeze()[0]

    points = points_input.copy()
    points["geometry"] = gpd.points_from_xy(
        points.geometry.x - home[0],
        points.geometry.y - home[1],
        crs=points.crs
    )

    coords = [[pos - home for pos in route] for route in coords_input]

    region = translate(
        region_input,
        xoff=-home[0],
        yoff=-home[1]
    )

    # Choose sampling locations
    if n_synth > 0:
        synth_points = random_points_in_polygon(region, n_synth, seed=None)
        synthetic = gpd.GeoDataFrame(
            {"sample_id": range(n_synth)},
            geometry=synth_points,
            crs=points.crs
        )
        waypts = np.column_stack([
            synthetic.geometry.x,
            synthetic.geometry.y
        ])
    else:
        waypts = np.array(coords).squeeze()[1:-1]
        print(f"waypts: {waypts}")

    ground_truth_pos = np.column_stack([
        points.geometry.x,
        points.geometry.y
    ])
    ground_truth_value = points["Moisture"].to_numpy()

    # Snap waypoints to nearest actual data points
    tree = cKDTree(ground_truth_pos)
    _, close_idx = tree.query(waypts, k=1)

    sampled_values = ground_truth_value[close_idx]
    sampled_pos = ground_truth_pos[close_idx]

    # Build kernel
    if two_RBF:
        long_rbf = make_rbf_component(
            "long",
            length_scale=100.0,
            bounds=(40.0, 800.0),
            aniso=aniso
        )

        short_rbf = make_rbf_component(
            "short",
            length_scale=20.0,
            bounds=(5.0, 80.0),
            aniso=aniso
        )

        kernel = long_rbf + short_rbf

    else:
        kernel = make_rbf_component(
            "single",
            length_scale=20.0,
            bounds=(0.1, 800.0),
            aniso=aniso
        )

    if noise:
        kernel = kernel + WhiteKernel(
            noise_level=0.01,
            noise_level_bounds=(1e-5, 1.0)
        )

    gaussian_process = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=20,
        normalize_y=True
    )

    gaussian_process.fit(sampled_pos, sampled_values)

    kernel_info = extract_kernel_info(
        gaussian_process,
        noise=noise,
        aniso=aniso,
        two_RBF=two_RBF
    )

    # Prediction grid
    minx, miny, maxx, maxy = region.bounds
    nx = 100
    ny = 100

    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    xx, yy = np.meshgrid(xs, ys)

    test_points = gpd.GeoSeries(
        gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=points.crs
    )

    inside = test_points.within(region)
    query_xy = np.column_stack([
        test_points.x[inside],
        test_points.y[inside]
    ])

    mean_pred, std_pred = gaussian_process.predict(
        query_xy,
        return_std=True
    )

    rmse, nrmse, nrmse_std = get_err(
        gp=gaussian_process,
        points=points,
        used_indices=close_idx
    )

    if gui:
        fig, ax = plt.subplots(2, 1)

        hi_val = np.quantile(ground_truth_value, 0.98)
        lo_val = np.quantile(ground_truth_value, 0.02)

        ax[0].scatter(
            query_xy[:, 0],
            query_xy[:, 1],
            c=mean_pred,
            vmin=lo_val,
            vmax=hi_val
        )
        ax[0].scatter(sampled_pos[:, 0], sampled_pos[:, 1], c="red", s=10)
        ax[0].set_title("GP mean prediction")

        ax[1].scatter(
            ground_truth_pos[:, 0],
            ground_truth_pos[:, 1],
            c=ground_truth_value,
            vmin=lo_val,
            vmax=hi_val
        )
        ax[1].scatter(sampled_pos[:, 0], sampled_pos[:, 1], c="red", s=10)
        ax[1].set_title("Actual moisture data")

        fig.suptitle(
            f"Predicted mean from samples vs ground truth\n"
            f"NRMSE_std: {nrmse_std:.3f}\n"
            f"{kernel_info['kernel']}"
        )

        plt.tight_layout()
        plt.show()

    return mean_pred, std_pred, kernel_info, rmse, nrmse, nrmse_std

def repeat_gp(coords,points,region,trials=25,voronoi=False,anisotropic=False,noise=False,two_RBF=True):
    
    if voronoi:
        n_points = np.array([0])
    else:
        #n_points = np.array([5, 10, 20, 50, 100, 200])
        n_points = np.array([5, 10, 20, 50, 100])
        #n_points = np.array([100])

    n_trials = trials
    nrmse_avg = np.zeros(n_points.size)
    rmse_avg = np.zeros(n_points.size)
    nrmse_std_avg = np.zeros(n_points.size)

    # Kernel hyperparameters
    mag_long_avg = np.zeros(n_points.size)
    mag_short_avg = np.zeros(n_points.size)
    ls_long_1_avg = np.zeros(n_points.size)
    ls_long_2_avg = np.zeros(n_points.size)
    ls_short_1_avg = np.zeros(n_points.size)
    ls_short_2_avg = np.zeros(n_points.size)
    noise_avg = np.zeros(n_points.size)

    for i in range(n_points.size):
        rmses = np.zeros(n_trials)
        nrmses = np.zeros(n_trials)
        nrmses_std = np.zeros(n_trials)

        mag_long_arr = np.zeros(n_trials)
        mag_short_arr = np.zeros(n_trials)
        ls_long_1_arr = np.zeros(n_trials)
        ls_long_2_arr = np.zeros(n_trials)
        ls_short_1_arr = np.zeros(n_trials)
        ls_short_2_arr = np.zeros(n_trials)
        noise_arr = np.zeros(n_trials)

        for j in range(n_trials):
            _, _, info, rmses[j], nrmses[j], nrmses_std[j] = fit_gp_flexible(
                                                                coords_input=coords,
                                                                points_input=points,
                                                                region_input=region,
                                                                n_synth=n_points[i],
                                                                gui=False,
                                                                noise=noise,aniso=anisotropic,two_RBF=two_RBF)
            
            if not two_RBF:
                mag_long_arr[j] = info["magnitudes"]["single"]
                mag_short_arr[j] = 0.0

                if anisotropic:
                    ls_long_1_arr[j] = info["length_scales"]["single"][0]
                    ls_long_2_arr[j] = info["length_scales"]["single"][1]
                    ls_short_1_arr[j] = 0.0
                    ls_short_2_arr[j] = 0.0
                else:
                    ls_long_1_arr[j] = info["length_scales"]["single"]
                    ls_long_2_arr[j] = info["length_scales"]["single"]
                    ls_short_1_arr[j] = 0.0
                    ls_short_2_arr[j] = 0.0
                
                noise_arr[j] = info["noise_level"] if noise else 0.0
            else:
                mag_long_arr[j] = info["magnitudes"]["long"]
                mag_short_arr[j] = info["magnitudes"]["short"]

                if anisotropic:
                    ls_long_1_arr[j] = info["length_scales"]["long"][0]
                    ls_long_2_arr[j] = info["length_scales"]["long"][1]
                    ls_short_1_arr[j] = info["length_scales"]["short"][0]
                    ls_short_2_arr[j] = info["length_scales"]["short"][1]
                else:
                    ls_long_1_arr[j] = info["length_scales"]["long"]
                    ls_long_2_arr[j] = info["length_scales"]["long"]
                    ls_short_1_arr[j] = info["length_scales"]["short"]
                    ls_short_2_arr[j] = info["length_scales"]["short"]
                
                noise_arr[j] = info["noise_level"] if noise else 0.0

        mag_long_avg[i] = np.mean(mag_long_arr)
        mag_short_avg[i] = np.mean(mag_short_arr)
        ls_long_1_avg[i] = np.mean(ls_long_1_arr)
        ls_long_2_avg[i] = np.mean(ls_long_2_arr)
        ls_short_1_avg[i] = np.mean(ls_short_1_arr)
        ls_short_2_avg[i] = np.mean(ls_short_2_arr)
        noise_avg[i] = np.mean(noise_arr)

        rmse_avg[i] = np.mean(rmses)
        nrmse_avg[i] = np.mean(nrmses)
        nrmse_std_avg[i] = np.mean(nrmses_std)

    if voronoi:
        # All coordinates, divided by 2 (because X,Y is 1 coords) minus 2 (don't sample start and end)
        n_points_true = ((np.array(coords).squeeze().size) / 2 - 2)
    else:
        n_points_true = n_points
    df = pd.DataFrame({
        "voronoi": voronoi,
        "anisotropic": anisotropic,
        "white_kernel": noise,
        "two_RBF": two_RBF,
        "n_points": n_points_true,
        "rmse_avg": rmse_avg,
        "nrmse_avg": nrmse_avg,
        "nrmse_std_avg": nrmse_std_avg,
        "mag_RBF_long_avg": mag_long_avg,
        "mag_RBF_short_avg": mag_short_avg,
        "ls_RBF_long_1_avg": ls_long_1_avg,
        "ls_RBF_long_2_avg": ls_long_2_avg,
        "ls_RBF_short_1_avg": ls_short_1_avg,
        "ls_RBF_short_2_avg": ls_short_2_avg,
        "noise_level_avg": noise_avg,
        "area": region.area,
        "std": rmse_avg / nrmse_std_avg,
        "area per sample (m2 / sample)": region.area / n_points_true, 
    })
    return df

def get_err(gp,points,used_indices):
    # Make sure we don't use the points that were used in fitting the model
    ground_truth_pos = np.column_stack([points.geometry.x, points.geometry.y])
    used_mask = np.zeros(len(ground_truth_pos), dtype=bool)
    used_mask[used_indices] = True
    valid = ~used_mask
    eval_pos = ground_truth_pos[valid]

    predicted_values = gp.predict(eval_pos)
    actual_values = points['Moisture'].to_numpy()
    eval_values = actual_values[valid]

    err = predicted_values - eval_values
    rmse = np.linalg.norm(err) / np.sqrt(err.size)
    nrmse = rmse / np.ptp(eval_values)
    nrmse_std = rmse / np.std(eval_values)
    return rmse,nrmse,nrmse_std
