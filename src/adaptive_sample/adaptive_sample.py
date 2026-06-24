import numpy as np
import yaml
import time
import os
import sys
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from candidate_actions import generate_variance_candidates
from model import MoistureModel
from sample_ground_truth import sample_from_ground_truth
from rewards import compute_cost
from planner import GreedyVariancePlanner
from visualize import plot_results

GRID_SPACING = 5

# Map loading 
def load_map_data(data_path):
    region = gpd.read_file(
        data_path,
        layer="polygon"
    )
    points = gpd.read_file(
        data_path,
        layer="points"
    )

    points = points.to_crs("32616")
    region = region.to_crs("32616")

    poly = region.geometry.iloc[0]
    # Vertices are stored at lon/lat, so we swap the order here
    #vertices = [[float(lat), float(lon)] for lon, lat in poly.exterior.coords[:-1]]
    vertices = [[float(x), float(y)] for x, y in poly.exterior.coords[:-1]]

    home = np.array(vertices).squeeze() 
    x_home = np.min(home[:,0]) 
    y_home = np.min(home[:,1])

    # shift points
    points["geometry"] = gpd.points_from_xy(
        points.geometry.x - x_home,
        points.geometry.y - y_home,
        crs=points.crs,
    )

    # shift polygon geometry
    region["geometry"] = region.geometry.apply(
        lambda geom: translate(geom, xoff=-x_home, yoff=-y_home)
    )

    poly_shifted = region.geometry.iloc[0]
    vertices = [[float(x), float(y)] for x, y in poly_shifted.exterior.coords[:-1]]

    return points,vertices,x_home,y_home

# ── Main ──────────────────────────────────────────────────────────────────────

def main(path):

    # Load the map
    points, vertices, x_home, y_home = load_map_data(path) 
    region = Polygon(vertices)
    
    start_time = time.time()
    budget_remaining = 1000

    model = MoistureModel()
    planner = GreedyVariancePlanner()

    # Initialize
    centroid = region.centroid
    current_position = np.array([centroid.x, centroid.y])

    ## For now, let's assume we sampled once where we started
    initial_sample_pos, initial_sample_value = sample_from_ground_truth(current_position, points)
    model.add_observation(initial_sample_pos, initial_sample_value, virtual=False)

    visited_pts = current_position.copy()

    while budget_remaining > 0:
        print(f"Budget remaining: {budget_remaining}")
        #gp.fit(real_X, real_y)
        candidates,_ = generate_variance_candidates(gp=model.gp, region=region, resolution=GRID_SPACING)
        candidates = np.asarray(candidates, dtype=float).reshape(-1, 2)

        plan = planner.plan(
            model=model,
            candidates=candidates,
            current_pos=current_position,
            budget_remaining=budget_remaining,
        )
        if len(plan) == 0:
            budget_remaining = -1
            # We don't have anywhere to go under our budget
            continue

        next_location = plan[0]

        sampled_X,sampled_y = sample_from_ground_truth(next_location,points)

        model.add_observation(sampled_X,sampled_y,virtual=False)

        budget_remaining -= compute_cost(current_position, next_location, sample=True)
        current_position = next_location
        visited_pts = np.vstack([visited_pts, current_position])
    
    print(f"Finished! Final trajectory: {visited_pts}")

    plot_results(
        points=points,
        region=region,
        model=model,
        visited_pts=visited_pts,
        resolution=GRID_SPACING,
    )
        


if __name__ == "__main__":
    main(path=sys.argv[1])