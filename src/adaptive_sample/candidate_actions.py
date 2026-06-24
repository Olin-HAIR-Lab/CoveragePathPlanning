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
    n_candidates=30,
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
    
    # Random order for now
    np.random.shuffle(order)

    candidates = grid[order[:n_candidates]]
    candidate_scores = std[order[:n_candidates]]

    return candidates, candidate_scores