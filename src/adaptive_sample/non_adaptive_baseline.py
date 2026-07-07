import numpy as np
import copy
import sys
import pandas as pd
from dataclasses import dataclass
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import translate

from candidate_actions import generate_variance_candidates
from model import MoistureModel
from sample_ground_truth import sample_from_ground_truth, get_err
from rewards import compute_cost, CompositeReward, get_variance_metrics, SAMPLE_COST
from planner import NStepLookaheadPlanner
from visualize import plot_results, plot_candidate_scores, plot_tree_candidate_paths, nrmse_over_dist
from adaptive_sample import SimulationConfig, load_map_data

from lloydsAlgorithm import Lloyd_algoritm
from vehicleRoutingProblem import solve_vrp_balanced, extract_paths

def run_simulation(config):

    # Load the map
    points, vertices, _, _ = load_map_data(config.data_path) 
    region = Polygon(vertices)
    budget_remaining = config.budget

    # Initialize
    centroid = region.centroid
    current_position = np.array([centroid.x, centroid.y])

    # Non-adaptive sampling
    success = False
    num_pts = budget_remaining // SAMPLE_COST
    while not success:
        fig_7_data = pd.DataFrame()
        model = MoistureModel(min_length_scale=config.min_length_scale)
        budget_remaining = config.budget

        # Try Voronoi partitions with N pts until we find one under budget
        print(f"Trying with {num_pts} points")
        ll_iter = config.lloyd_iterations
        ll_partition = config.lloyd_partition # density of grid
        ll_seed = config.seed

        history_tessell, history_dots = Lloyd_algoritm(
            ll_iter, num_pts, region, ll_partition, ll_seed, log=config.make_plots
        )
        
        _ = history_tessell[-1]
        final_dots = history_dots[-1]

        coords = np.vstack([current_position, final_dots.copy()])

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
        if config.make_plots:
            print(f"Initial points: {routes_coords}")
        visited_pts = routes_coords.copy()

        ## For now, assume we don't sample at our starting pos, but we do sample at each presample pos
        # We need to count the budget spent as well
        visited_count = 1
        for pt in visited_pts[1:,:]:
            initial_sample_pos, initial_sample_value = sample_from_ground_truth(pt, points)
            model.add_observation(initial_sample_pos, initial_sample_value, virtual=False)

            budget_remaining -= compute_cost(current_position, pt)

            if config.make_figure_7:
                info = nrmse_over_dist(points=points, model=model, visited_pts=visited_pts[0:visited_count,:])
                info |= {
                    "path": config.data_path,
                    "presample": True
                }
                fig_7_data = pd.concat([fig_7_data,pd.DataFrame([info])],ignore_index=True)
                visited_count += 1

        if budget_remaining < 0:
            print(f"Too expensive: budget at {budget_remaining} for {num_pts} points. Trying again...")
            num_pts -= 1
            continue
        success = True

    print(f"Budget remaining: {budget_remaining}")
    current_position = visited_pts[-1,:]

    if config.make_plots:
        print(f"Finished! Final trajectory: {visited_pts}")
        print("Retraining hyperparameters")
    model.retrain_hyperparameters()

    if config.make_plots:
        plot_results(
            points=points,
            region=region,
            model=model,
            visited_pts=visited_pts,
            resolution=config.grid_spacing,
        )
    
    rmse,nrmse,_,rmse_over_std = get_err(gp=model.gp, points=points)
    data_range = np.ptp(points['Moisture'].to_numpy())
    data_std = np.std(points['Moisture'].to_numpy())
    lengthscale = model.gp.kernel_.get_params()['k1__length_scale']
    num_pts = visited_pts.shape[0]
    results = {
        "rmse": rmse,
        "nrmse": nrmse,
        "rmse_over_std": rmse_over_std,
        "length_scale": lengthscale,
        "num_pts": num_pts,
        "presample_pts": num_pts,
        "n_steps": 0.0,
        "budget": config.budget,
        "n_candidates": 0.0,
        "min_spacing": 0.0,
        "min_length_scale": config.min_length_scale,
        "w_mean": 0.0,
        "w_std": 0.0,
        "w_dist": 0.0,
        "w_grad_mean": 0.0,
        "w_far_from_mean": 0.0,
        "data_range": data_range,
        "data_std": data_std,
        "region_area": region.area,
        "path": config.data_path
    }
    results |= get_variance_metrics(model=model,region=region,resolution=5.0)
    if config.make_plots:
        print(results)
    if config.make_figure_7:
        print(fig_7_data)
    result_df = pd.DataFrame([results])
    return result_df
        
if __name__ == "__main__":
    config_in = SimulationConfig(data_path=sys.argv[1])
    config_in.make_plots = True
    config_in.make_figure_y = True 
    run_simulation(config=config_in)