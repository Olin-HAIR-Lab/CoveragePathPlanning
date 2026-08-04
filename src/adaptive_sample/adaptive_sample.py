import numpy as np
import copy
from datetime import datetime
import sys
import pandas as pd
from dataclasses import dataclass
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import translate

from candidate_actions import generate_variance_candidates
from model import MoistureModel
from sample_ground_truth import sample_from_ground_truth, get_err, sample_from_ground_truth_knn, make_interpolated_ground_truth_grid
from rewards import compute_cost, CompositeReward, get_variance_metrics
from planner import NStepLookaheadPlanner
from visualize import plot_results, plot_candidate_scores, plot_tree_candidate_paths, nrmse_over_dist

from lloydsAlgorithm import lloyd_algorithm
from vehicleRoutingProblem import solve_vrp_balanced, extract_paths

@dataclass
class SimulationConfig:
    data_path: str
    n_steps: int = 3
    budget: float = 1000
    grid_spacing: float = 2
    n_candidates: int = 10
    min_candidate_spacing: float = 20
    presample_pts: int = 3
    lloyd_iterations: int = 10
    seed: int | None = None
    make_plots: bool = True
    mean_weight: float = 0
    std_weight: float = 1
    grad_mean_weight: float = 0.0
    dist_weight: float = 0.005
    far_from_mean_weight: float = 0
    min_length_scale: float = 1.0
    integrate_std: bool = False
    integrate_std_planner: bool = False
    fixed_radius: float = None
    make_figure_7: bool = False

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

def run_simulation(config):

    # Load the map
    points, vertices, home_x, home_y = load_map_data(config.data_path) 
    region = Polygon(vertices)
    budget_remaining = config.budget

    fig_7_data = pd.DataFrame()

    rwd_fun = CompositeReward(
        w_mean=config.mean_weight,
        w_std=config.std_weight,
        w_mean_grad=config.grad_mean_weight,
        w_dist=config.dist_weight,
        w_far_from_mean=config.far_from_mean_weight
    )

    model = MoistureModel(min_length_scale=config.min_length_scale)
    planner = NStepLookaheadPlanner(
        n_steps=config.n_steps,
        rwd_fun=rwd_fun,
        region=region,
        integrate_std=config.integrate_std_planner,
        fixed_length_scale=config.fixed_radius
    )

    # ground truth grid is a GDF with a 'Moisture' field
    ground_truth_grid = make_interpolated_ground_truth_grid(
        points=points,
        region=region,
        resolution=config.grid_spacing,
        k=4,
        power=2,
        max_dist=None
    )

    # Initialize
    centroid = region.centroid
    current_position = np.array([centroid.x, centroid.y])

    # Presampling 
    if config.presample_pts > 0:
        ll_iter = config.lloyd_iterations
        ll_seed = config.seed
        _, history_dots = lloyd_algorithm(
            ll_iter, config.presample_pts, region, ll_seed, log=config.make_plots
        )
        
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
            initial_sample_pos, initial_sample_value = sample_from_ground_truth(pt, ground_truth_grid)
            model.add_observation(initial_sample_pos, initial_sample_value, virtual=False)

            budget_remaining -= compute_cost(current_position, pt)

            if config.make_figure_7:
                info = nrmse_over_dist(points=ground_truth_grid, model=model, visited_pts=visited_pts[0:visited_count,:])
                info |= {
                    "path": config.data_path,
                    "presample": True,
                    "x": pt[0] + home_x,
                    "y": pt[1] + home_y
                }
                fig_7_data = pd.concat([fig_7_data,pd.DataFrame([info])],ignore_index=True)
                visited_count += 1
        
        current_position = visited_pts[-1,:]
    
    else:
        # No presample pts
        visited_pts = current_position.copy().reshape((1,2))

    while budget_remaining > 0:
        if config.make_plots:
            print(f"Budget remaining: {budget_remaining}")
        #gp.fit(real_X, real_y)
        candidates,_,avg_mean = generate_variance_candidates(
            gp=model.gp, 
            region=region, 
            rwd_fun=rwd_fun,
            current_pos=current_position,
            resolution=config.grid_spacing,
            n_candidates=config.n_candidates, 
            min_spacing=config.min_candidate_spacing,
            fixed_radius=config.fixed_radius,
            integrate_std=config.integrate_std
        )
        candidates = np.asarray(candidates, dtype=float).reshape(-1, 2)

        if config.make_plots:
            scores = plot_candidate_scores(
                model=model,
                rwd_fun=planner.rwd_fun,
                candidates=candidates,
                current_pos=current_position,
                region=region,
                visited_pts=visited_pts,
                field_mean=avg_mean,
                title=f"Candidate rewards, budget={budget_remaining:.1f}",
            )
            print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")

        plan, leaf_nodes, best_node = planner.plan(
            model=model,
            candidates=candidates,
            current_pos=current_position,
            budget_remaining=budget_remaining,
            field_mean=avg_mean,
            log=config.make_plots
        )
        if len(plan) == 0:
            budget_remaining = -1
            # We don't have anywhere to go under our budget
            continue

        next_location = plan[0]
        # if config.make_plots:
        #     print(f"Plan[0]: {plan[0]}")
        #     plot_tree_candidate_paths(
        #         region=region,
        #         leaf_nodes=leaf_nodes,
        #         best_node=best_node,
        #         current_pos=current_position,
        #         visited_pts=visited_pts,
        #         max_paths_to_plot=300,
        #     )

        sampled_X,sampled_y = sample_from_ground_truth(next_location,ground_truth_grid)

        model.add_observation(sampled_X,sampled_y,virtual=False)

        if config.make_figure_7:
            info = nrmse_over_dist(points=ground_truth_grid, model=model, visited_pts=visited_pts)
            info |= {
                "path": config.data_path,
                "presample": False,
                "x": sampled_X[0] + home_x,
                "y": sampled_X[1] + home_y
            }
            fig_7_data = pd.concat([fig_7_data,pd.DataFrame([info])],ignore_index=True)

        budget_remaining -= compute_cost(current_position, next_location, sample=True)
        current_position = next_location
        visited_pts = np.vstack([visited_pts, current_position])
    
    if config.make_plots:
        print(f"Finished! Final trajectory: {visited_pts}")
        print("Retraining hyperparameters")
    model.retrain_hyperparameters()

    if config.make_plots:
        plot_results(
            points=ground_truth_grid,
            points_original=points,
            region=region,
            model=model,
            visited_pts=visited_pts,
            resolution=config.grid_spacing,
        )
    
    rmse,nrmse,_,rmse_over_std = get_err(gp=model.gp, points=ground_truth_grid)
    data_range = np.ptp(ground_truth_grid['Moisture'].to_numpy())
    data_std = np.std(ground_truth_grid['Moisture'].to_numpy())
    lengthscale = model.gp.kernel_.get_params()['k1__length_scale']
    num_pts = visited_pts.shape[0]
    results = {
        "rmse": rmse,
        "nrmse": nrmse,
        "rmse_over_std": rmse_over_std,
        "length_scale": lengthscale,
        "num_pts": num_pts,
        "presample_pts": config.presample_pts,
        "n_steps": config.n_steps,
        "budget": config.budget,
        "n_candidates": config.n_candidates,
        "min_spacing": config.min_candidate_spacing,
        "min_length_scale": config.min_length_scale,
        "w_mean": config.mean_weight,
        "w_std": config.std_weight,
        "w_dist": config.dist_weight,
        "w_grad_mean": config.grad_mean_weight,
        "w_far_from_mean": config.far_from_mean_weight,
        "data_range": data_range,
        "data_std": data_std,
        "region_area": region.area,
        "path": config.data_path,
        "integrate_std": config.integrate_std,
        "integrate_std_planner": config.integrate_std_planner,
        "fixed_radius": config.fixed_radius
    }
    results |= get_variance_metrics(model=model,region=region,resolution=5.0)
    if config.make_plots:
        print(results)
    if config.make_figure_7:
        print(fig_7_data)
        fig_7_data.to_csv(f"fig7/data_{datetime.now().strftime('%Y%m%d %H:%M:%S')}_presample={config.presample_pts}.csv")
    result_df = pd.DataFrame([results])

    return result_df
        
if __name__ == "__main__":
    config_in = SimulationConfig(data_path=sys.argv[1])
    config_in.make_plots = True
    config_in.min_length_scale = 20.0
    config_in.dist_weight = 1e-6
    config_in.grad_mean_weight = 0.15
    config_in.far_from_mean_weight = 0.0
    config_in.presample_pts = 3
    config_in.budget = 180
    config_in.std_weight = 1.0
    config_in.make_figure_7 = False
    config_in.seed = None
    config_in.integrate_std = True
    config_in.integrate_std_planner = True
    config_in.fixed_radius = None
    run_simulation(config=config_in)