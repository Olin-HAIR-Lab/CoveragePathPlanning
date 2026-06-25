import numpy as np
import yaml
import time
import copy
import os
import sys
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from candidate_actions import generate_variance_candidates, generate_candidate_paths
from model import MoistureModel
from sample_ground_truth import sample_from_ground_truth, get_err
from rewards import compute_cost
from planner import GreedyVariancePlanner, VarianceMinusDistancePlanner, score_virtual_path, NStepLookaheadPlanner
from visualize import plot_results, plot_candidate_scores, plot_candidate_paths

from lloydsAlgorithm import Lloyd_algoritm
from vehicleRoutingProblem import solve_vrp_balanced, extract_paths

GRID_SPACING = 5
PRESAMPLE_PTS = 3

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
    #planner = VarianceMinusDistancePlanner()
    planner = NStepLookaheadPlanner(n_steps=3)

    # Initialize
    centroid = region.centroid
    current_position = np.array([centroid.x, centroid.y])

    # Presampling 
    ll_iter = 1
    ll_partition = 300 # density of grid
    ll_seed = None
    history_tessell, history_dots = Lloyd_algoritm(
        ll_iter, PRESAMPLE_PTS, region, ll_partition, ll_seed
    )
    
    final_tessellation = history_tessell[-1]
    final_dots = history_dots[-1]

    print(current_position)
    print(final_dots)
    coords = np.vstack([current_position, final_dots.copy()])
    print(coords)

    travel_duration_matrix = np.array([
        [np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
        for j in range(len(coords))]
        for i in range(len(coords))
    ])

    max_dist = travel_duration_matrix.max()
    travel_duration_matrix = travel_duration_matrix / max_dist * 100

    mission_time = 10000
    time_windows = np.array([
        (0, mission_time - travel_duration_matrix[i][0])
        for i in range(len(coords))
    ])

    solution = solve_vrp_balanced(
        coords, time_windows, travel_duration_matrix, num_vehicles=1)

    _, routes = extract_paths(solution, coords)

    # ── Build per-route coordinate lists ───────────────────────
    routes_coords = []
    i = 0

    for route in routes:
        route_indices = [0] + list(route) + [0]
        coords[0] = current_position

        route_coords = [copy.copy(coords[i]) for i in route_indices]
        routes_coords.append(route_coords)
        i += 1
    
    routes_coords = np.array(routes_coords[0])[:-1] # don't double-sample the start
    print(f"Initial points: {routes_coords}")
    visited_pts = routes_coords.copy()

    ## For now, assume we don't sample at our starting pos, but we do sample at each presample pos
    # We need to count the budget spent as well
    for pt in visited_pts[1:,:]:
        initial_sample_pos, initial_sample_value = sample_from_ground_truth(pt, points)
        model.add_observation(initial_sample_pos, initial_sample_value, virtual=False)

        budget_remaining -= compute_cost(current_position, pt)
    
    current_position = visited_pts[-1,:]

    while budget_remaining > 0:
        print(f"Budget remaining: {budget_remaining}")
        #gp.fit(real_X, real_y)
        candidates,_ = generate_variance_candidates(gp=model.gp, region=region, resolution=GRID_SPACING, n_candidates=10, min_spacing=40)
        candidates = np.asarray(candidates, dtype=float).reshape(-1, 2)
        #candidate_paths = generate_candidate_paths(candidates=candidates,n_step=3,count=300)

        scores = plot_candidate_scores(
            model=model,
            rwd_fun=planner.rwd_fun,
            candidates=candidates,
            current_pos=current_position,
            region=region,
            visited_pts=visited_pts,
            title=f"Candidate rewards, budget={budget_remaining:.1f}",
        )
        print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")

        # scores = np.array([
        #     score_virtual_path(model, planner.rwd_fun, current_position, path)
        #     for path in candidate_paths
        # ])

        # best_idx = np.argmax(scores)
        # best_path = candidate_paths[best_idx]

        # plot_candidate_paths(
        #     region=region,
        #     candidate_paths=candidate_paths,
        #     scores=scores,
        #     best_path=best_path,
        #     current_pos=current_position,
        #     visited_pts=visited_pts,
        #     max_paths_to_plot=200,
        # )

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
        print(f"Plan[0]: {plan[0]}")

        sampled_X,sampled_y = sample_from_ground_truth(next_location,points)

        model.add_observation(sampled_X,sampled_y,virtual=False)

        budget_remaining -= compute_cost(current_position, next_location, sample=True)
        current_position = next_location
        visited_pts = np.vstack([visited_pts, current_position])
    
    print(f"Finished! Final trajectory: {visited_pts}")
    print(f"Retraining hyperparameters")
    model.retrain_hyperparameters()

    plot_results(
        points=points,
        region=region,
        model=model,
        visited_pts=visited_pts,
        resolution=GRID_SPACING,
    )
        


if __name__ == "__main__":
    main(path=sys.argv[1])