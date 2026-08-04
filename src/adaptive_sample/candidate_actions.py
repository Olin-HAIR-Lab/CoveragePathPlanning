import numpy as np
from shapely.geometry import Point
from scipy.ndimage import maximum_filter
import numpy as np
from scipy.spatial import cKDTree
from planner import get_gp_length_scale, neighborhood_std_reward

def neighborhood_std_scores(
    grid_points,
    std,
    length_scale,
    grid_resolution,
):
    """
    Compute full-circle-normalized neighborhood standard deviation
    for every valid grid point.

    grid_points: shape (N, 2), containing only points inside the region
    std: shape (N,), GP std at those points
    """
    grid_points = np.asarray(grid_points, dtype=float).reshape(-1, 2)
    std = np.asarray(std, dtype=float).reshape(-1)

    radius = max(float(length_scale), float(grid_resolution))
    tree = cKDTree(grid_points)

    neighbor_indices = tree.query_ball_point(
        grid_points,
        r=radius,
    )

    cell_area = grid_resolution**2
    full_circle_area = np.pi * radius**2

    scores = np.empty(len(grid_points), dtype=float)

    for i, indices in enumerate(neighbor_indices):
        scores[i] = (
            np.sum(std[indices]) * cell_area
            / full_circle_area
        )

    return scores

def grid_points_in_polygon(region, resolution):
    """
    Generate candidate grid points inside a Polygon.

    Args:
        region (shapely Polygon): Region boundary in projected coordinates, e.g. meters.
        resolution (float): Grid spacing in same units as region coordinates.

    Returns
        points: Candidate XY points inside region.
    """
    minx, miny, maxx, maxy = region.bounds

    xs = np.arange(minx, maxx + resolution, resolution)
    ys = np.arange(miny, maxy + resolution, resolution)

    pts = []
    for x in xs:
        for y in ys:
            p = Point(x, y)
            if region.contains(p):
                pts.append([x, y])

    return np.asarray(pts)

def generate_variance_candidates(
    gp,
    region,
    rwd_fun,
    current_pos,
    integrate_std,
    fixed_radius=None,
    resolution=2.0,
    n_candidates=5,
    min_spacing=5,
):
    """
    Generate candidate sample locations by evaluating GP uncertainty
    on a grid and keeping the highest-variance points.
    """
    grid = grid_points_in_polygon(region, resolution)

    if len(grid) == 0:
        raise ValueError("No grid points found inside region.")

    current_pos = np.asarray(current_pos, dtype=float).reshape(2,)

    minx, miny, maxx, maxy = region.bounds
    xs = np.arange(minx, maxx + resolution, resolution)
    ys = np.arange(miny, maxy + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)

    full_grid = np.column_stack([xx.ravel(), yy.ravel()])

    inside = np.array([region.contains(Point(x, y)) for x, y in full_grid])
    grid_inside = full_grid[inside]

    pred_mean, pred_std = gp.predict(grid_inside, return_std=True)
    avg_mean = np.mean(pred_mean.flatten())

    if integrate_std:
        if fixed_radius is None:
            length_scale = get_gp_length_scale(gp)
        else:
            length_scale = fixed_radius

        std_scores = neighborhood_std_scores(
            grid_points=grid_inside,
            std=pred_std,
            length_scale=length_scale,
            grid_resolution=resolution
        )
    else:
        std_scores = pred_std

    mean_grid = np.full(len(full_grid), np.nan)
    std_grid = np.full(len(full_grid), np.nan)

    mean_grid[inside] = pred_mean
    std_grid[inside] = std_scores

    mean_grid = mean_grid.reshape(xx.shape)
    std_grid = std_grid.reshape(xx.shape)

    # np.gradient returns [d/dy, d/dx]
    dmean_dy, dmean_dx = np.gradient(mean_grid, resolution)
    grad_mean_grid = np.sqrt(dmean_dx**2 + dmean_dy**2)

    reward_grid = np.full(xx.shape, np.nan)

    for row in range(xx.shape[0]):
        for col in range(xx.shape[1]):
            if np.isnan(mean_grid[row, col]):
                continue

            target = np.array([xx[row, col], yy[row, col]])

            reward_grid[row, col] = rwd_fun.evaluate(
                float(mean_grid[row, col]),
                avg_mean,
                float(std_grid[row, col]),
                float(grad_mean_grid[row, col]),
                current_pos,
                target,
            )

    # Flatten valid reward locations
    flat_rewards = reward_grid.ravel()
    valid = ~np.isnan(flat_rewards)

    valid_grid = full_grid[valid]
    valid_rewards = flat_rewards[valid]

    order = np.argsort(valid_rewards)[::-1]

    candidates = []
    candidate_scores = []

    for idx in order:
        candidate = valid_grid[idx]
        score = valid_rewards[idx]

        if len(candidates) == 0:
            keep = True
        else:
            existing = np.asarray(candidates)
            dists = np.linalg.norm(existing - candidate, axis=1)
            keep = np.all(dists >= min_spacing)

        if keep:
            candidates.append(candidate)
            candidate_scores.append(score)

        if len(candidates) >= n_candidates:
            break

    return np.asarray(candidates), np.asarray(candidate_scores), avg_mean

def generate_candidate_paths(candidates, n_step, count):
    """
    Given a set of candidate points, generate a number of paths connecting them
    """
    candidates = np.asarray(candidates, dtype=float).reshape(-1, 2)
    candidate_paths = np.zeros((count, n_step, 2), dtype=float)

    for i in range(count):
        choices = np.random.choice(len(candidates), size=n_step, replace=False)
        candidate_paths[i] = candidates[choices]

    return candidate_paths