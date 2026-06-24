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