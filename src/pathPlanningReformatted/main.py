import math
import copy
import os
import yaml
import pickle
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer

from lloydsAlgorithm import Lloyd_algoritm
from vehicleRoutingProblem import (
    solve_vrp_unlimited,
    solve_vrp_balanced,
    extract_paths,
)
from collisionAvoidance import add_delays_to_avoid_collisions, animate_trajectories
from plotting import plot_results
from jsonTesting import makeJSONMission
from datetime import datetime


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def polygon_area_m2(poly_latlon):
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32619", always_xy=True)

    def _swap_xy(x, y, z=None):
        return transformer.transform(y, x)

    poly_utm = transform(_swap_xy, poly_latlon)
    return poly_utm.area


def compute_sample_count(map, sample_time, speed, mission_time, num_agents=1):
    area = polygon_area_m2(map)
    max_possible = int((mission_time * num_agents) / sample_time)
    best_N = 1
    for N in range(1, max_possible + 1):
        if N % num_agents != 0:
            continue
        N_per_agent = N / num_agents
        total_time = N_per_agent * sample_time + \
            math.sqrt(area * N_per_agent) / speed
        if total_time <= mission_time:
            best_N = N
        else:
            break
    return best_N


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    m = cfg["mission"]
    ll = cfg["lloyd"]
    v = cfg["vrp"]
    d = cfg["depot"]
    out = cfg["output"]
    ani = cfg["animation"]

    num_agents = m["num_agents"]
    mission_time = m["mission_time"]
    sample_time = m["sample_time"]
    speed = m["speed"]
    d_safe = m["d_safe"]

    # ── Polygon ───────────────────────────────────────────────────────────────
    poly = Polygon([tuple(pt) for pt in cfg["polygon"]["vertices"]])

    # ── Sample count ──────────────────────────────────────────────────────────
    if ll["n_dots_override"] is not None:
        N_dots = ll["n_dots_override"]
        print(f"[config] n_dots_override = {N_dots}")
    else:
        N_dots = compute_sample_count(
            poly,
            sample_time=sample_time,
            speed=speed,
            mission_time=mission_time,
            num_agents=num_agents,
        )
        print(f"[config] N_dots computed = {N_dots}")

    # ── Lloyd's Algorithm ─────────────────────────────────────────────────────
    CACHE_FILE = f"lloyd_cache_N{1}_iter{1}_part{1}_seed{1}.pkl"

    if os.path.exists(CACHE_FILE):
        print(f"Loading Lloyd cache from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            history_tessell, history_dots = pickle.load(f)
    else:
        history_tessell, history_dots = Lloyd_algoritm(
            ll["iterations"], N_dots, poly, ll["partition"], ll["seed"]
        )
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((history_tessell, history_dots), f)
        print(f"Lloyd result cached to {CACHE_FILE}")
    
    final_tessellation = history_tessell[-1]
    final_dots = history_dots[-1]

    # ── Depot Coordinate ──────────────────────────────────────────────────────
    minx, miny, maxx, maxy = poly.bounds
    map_width = maxx - minx
    map_height = maxy - miny

    if d["mode"] == "offset":
        depot_coord = np.array([
            (miny + maxy) / 2 + d["offset_y"] * map_height,
        ])
    elif d["mode"] == "coordinate":
        depot_coord = np.array([d["depot1_x"],d["depot1_y"]])

    # ── VRP ───────────────────────────────────────────────────────────────────
    coords = np.vstack([depot_coord, final_dots.copy()])

    travel_duration_matrix = np.array([
        [np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
         for j in range(len(coords))]
        for i in range(len(coords))
    ])

    max_dist = travel_duration_matrix.max()
    travel_duration_matrix = travel_duration_matrix / max_dist * 100

    time_windows = np.array([
        (0, mission_time - travel_duration_matrix[i][0])
        for i in range(len(coords))
    ])

    if v["mode"] == "balanced":
        solution = solve_vrp_balanced(
            coords, time_windows, travel_duration_matrix, num_vehicles=num_agents
        )
    elif v["mode"] == "unlimited":
        solution = solve_vrp_unlimited(
            coords, time_windows, travel_duration_matrix)
    else:
        raise ValueError(
            f"Unknown VRP mode: '{v['mode']}'. Use 'balanced' or 'unlimited'.")

    paths, routes = extract_paths(solution, coords)

    # ── Build per-route coordinate lists (2-D and 3-D) ───────────────────────
    routes_coords = []
    routes_coords_3d = []
    i = 1

    for route in solution.routes():
        route_indices = [0] + list(route) + [0]
        coords[0] = np.array([d[f"depot{i}_x"],d[f"depot{i}_y"]]) 

        route_coords = [copy.copy(coords[i]) for i in route_indices]
        route_coords_3d = [np.append(copy.copy(coords[i]), out["drone_altitude"])
                           for i in route_indices]

        routes_coords.append(route_coords)
        routes_coords_3d.append(route_coords_3d)

        i += 1

    # ── Collision avoidance ───────────────────────────────────────────────────
    trajectories, delays = add_delays_to_avoid_collisions(
        routes_coords, speed=speed, d_safe=d_safe
    )

    # ── JSON output ───────────────────────────────────────────────────────────
    if out["json_file"]:
        makeJSONMission(out["json_file"], *routes_coords_3d[:num_agents])

    # ── Static plot (optional) ────────────────────────────────────────────────
    if ani["show_static_plot"]:
        plot_results(poly, final_tessellation, coords, solution,
                     coord_order="latlon")

    # ── Animation ─────────────────────────────────────────────────────────────
    os.makedirs("animation_output", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"animation_output/trajectories_{timestamp}.gif"

    animate_trajectories(
        trajectories,
        routes_coords,
        speed_multiplier=ani["speed_multiplier"],
        poly=poly,
        tessellation=final_tessellation,
        map_coords=coords,
        solution=solution,
        coord_order="latlon",
        save_path=save_path
    )
    


if __name__ == "__main__":
    main()
