"""
Function to use a known ground truth moisture map to return samples from queried locations
"""
import numpy as np
from scipy.spatial import cKDTree

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

    return sampled_pos,sampled_values

def get_err(gp, points):
    # Make sure we don't use the points that were used in fitting the model
    ground_truth_pos = np.column_stack([points.geometry.x, points.geometry.y])

    predicted_values,prediction_std = gp.predict(ground_truth_pos,return_std=True)
    actual_values = points['Moisture'].to_numpy()

    err = predicted_values - actual_values
    rmse = np.linalg.norm(err) / np.sqrt(err.size)
    nrmse = rmse / np.ptp(actual_values)
    nrmse_std = rmse / np.std(actual_values)

    # Get the error relative to the std
    eps = 1e-12
    err_over_std = err / np.maximum(prediction_std, eps)
    rmse_over_std = np.linalg.norm(err_over_std) / np.sqrt(err_over_std.size)

    return rmse,nrmse,nrmse_std,rmse_over_std