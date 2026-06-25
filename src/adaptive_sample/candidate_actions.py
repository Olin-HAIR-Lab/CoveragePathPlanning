import numpy as np
from shapely.geometry import Point
from scipy.ndimage import maximum_filter

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

    _, std = gp.predict(grid, return_std=True)

    # Sort by descending uncertainty
    order = np.argsort(std)[::-1]

    candidates = []
    candidate_scores = []

    for idx in order:
        candidate = grid[idx]

        if len(candidates) == 0:
            keep = True
        else:
            existing = np.asarray(candidates)
            dists = np.linalg.norm(existing - candidate, axis=1)
            keep = np.all(dists >= min_spacing)

        if keep:
            candidates.append(candidate)
            candidate_scores.append(std[idx])

        if len(candidates) >= n_candidates:
            break

    return np.asarray(candidates), np.asarray(candidate_scores)

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