"""
Function to use a known ground truth moisture map to return samples from queried locations
"""
import numpy as np
from scipy.spatial import cKDTree
import geopandas as gpd

def sample_from_ground_truth(pos,ground_truth_points):
    ground_truth_pos = np.column_stack([
        ground_truth_points.geometry.x,
        ground_truth_points.geometry.y
    ])
    ground_truth_value = ground_truth_points["Moisture"].to_numpy()
    # Snap waypoints to nearest actual data points
    tree = cKDTree(ground_truth_pos)
    _, close_idx = tree.query(pos, k=1)

    sampled_values = ground_truth_value[close_idx]
    sampled_pos = ground_truth_pos[close_idx]

    # Note that we sampled this datapoint
    ground_truth_points[close_idx, "sampled"] = True

    return sampled_pos,sampled_values

def sample_from_ground_truth_knn(
    pos,
    ground_truth_points,
    k=5,
    power=2,
    max_dist=None,
):
    pos = np.atleast_2d(pos)
    ground_truth_pos = np.column_stack([
        ground_truth_points.geometry.x,
        ground_truth_points.geometry.y
    ])

    ground_truth_value = ground_truth_points["Moisture"].to_numpy()
    k = min(k, len(ground_truth_points))

    tree = cKDTree(ground_truth_pos)

    dists, idxs = tree.query(pos, k=k)

    # Make shapes consistent even if pos is a single point
    dists = np.atleast_2d(dists)
    idxs = np.atleast_2d(idxs)

    neighbor_values = ground_truth_value[idxs]

    # Optional: ignore samples too far from ground-truth data
    if max_dist is not None:
        valid = dists <= max_dist
    else:
        valid = np.ones_like(dists, dtype=bool)

    sampled_values = np.full(len(dists), np.nan)

    for i in range(len(dists)):
        valid_i = valid[i]

        if not np.any(valid_i):
            continue

        d = dists[i, valid_i]
        v = neighbor_values[i, valid_i]

        # If exactly on a known point, use it directly
        if np.any(d == 0):
            sampled_values[i] = v[d == 0][0]
        else:
            weights = 1 / (d ** power)
            sampled_values[i] = np.sum(weights * v) / np.sum(weights)

    sampled_pos = pos

    return sampled_pos, sampled_values

def get_err(gp, points):
    # Make sure we don't use the points that were used in fitting the model
    eval_points = points[~points["sampled"]]
    ground_truth_pos = np.column_stack([eval_points.geometry.x, eval_points.geometry.y])    

    predicted_values,prediction_std = gp.predict(ground_truth_pos,return_std=True)
    actual_values = eval_points['Moisture'].to_numpy()

    err = predicted_values - actual_values
    rmse = np.linalg.norm(err) / np.sqrt(err.size)
    nrmse = rmse / np.ptp(actual_values)
    nrmse_std = rmse / np.std(actual_values)

    #print(f"Err: {err}\nRMSE: {rmse}\nNRMSE: {nrmse}")

    # Get the error relative to the std
    eps = 1e-12
    err_over_std = err / np.maximum(prediction_std, eps)
    rmse_over_std = np.linalg.norm(err_over_std) / np.sqrt(err_over_std.size)

    return rmse,nrmse,nrmse_std,rmse_over_std

def make_interpolated_ground_truth_grid(
    points,
    region,
    resolution=2.0,
    k=5,
    power=2,
    max_dist=None,
):
    minx, miny, maxx, maxy = region.bounds

    xs = np.arange(minx, maxx + resolution, resolution)
    ys = np.arange(miny, maxy + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)

    grid_pos = np.column_stack([
        xx.ravel(),
        yy.ravel(),
    ])

    # Keep only grid points inside or on the region boundary
    grid_geometry = gpd.GeoSeries(
        gpd.points_from_xy(grid_pos[:, 0], grid_pos[:, 1]),
        crs=points.crs,
    )

    inside = grid_geometry.covered_by(region)

    grid_pos = grid_pos[inside.to_numpy()]

    _, moisture = sample_from_ground_truth_knn(
        pos=grid_pos,
        ground_truth_points=points,
        k=k,
        power=power,
        max_dist=max_dist,
    )

    # max_dist may cause some interpolated values to be NaN
    valid = np.isfinite(moisture)
    grid_pos = grid_pos[valid]
    moisture = moisture[valid]

    ground_truth_grid = gpd.GeoDataFrame(
        {
            "Moisture": moisture,
            "sampled": False,
        },
        geometry=gpd.points_from_xy(
            grid_pos[:, 0],
            grid_pos[:, 1],
        ),
        crs=points.crs,
    )

    return ground_truth_grid