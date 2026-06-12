import math
import copy
import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer
import geopandas as gpd

from lloydsAlgorithm import Lloyd_algoritm
from vehicleRoutingProblem import (
    solve_vrp_unlimited,
    solve_vrp_balanced,
    extract_paths,
)
from fit_gp import repeat_gp

from sampleCount import compute_sample_count
from animation import animate_trajectories
from collisionAvoidance import add_delays_to_avoid_collisions
from plotting import plot_results
from jsonTesting import makeJSONMission
from datetime import datetime


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

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

    # Convert to CRS with units of meters
    # We store them in 4326 (units of latlon) but we want meters
    # points = points.to_crs("32616")
    # region = region.to_crs("32616")

    poly = region.geometry.iloc[0]
    # Vertices are stored at lon/lat, so we swap the order here
    vertices = [[float(lat), float(lon)] for lon, lat in poly.exterior.coords[:-1]]
    #vertices = [[float(x), float(y)] for x, y in poly.exterior.coords[:-1]]
    #print(f"Vertices: {vertices}")

    return points,vertices



# ── Main ──────────────────────────────────────────────────────────────────────

def main(yaml_path):
    with open(yaml_path,'r') as f:
        yaml_data = yaml.safe_load(f)
    
    full_df = pd.DataFrame()
    for data_path in yaml_data["data_paths"]:
        print(f"\n\nStarting path {data_path}...\n\n")
        cfg = load_config()

        # Override config with our vertex values from the map
        points, vertices = load_map_data(data_path=data_path)
        crs = points.crs
        utm_crs = points.estimate_utm_crs()
        print(f"Coordinates are in {crs} / UTM {utm_crs}")

        m = cfg["mission"]
        ll = cfg["lloyd"]
        v = cfg["vrp"]
        d = cfg["depot"]
        d["depots"] = [vertices[0]]
        print(f"vert 0 = {vertices[0]}")
        out = cfg["output"]
        ani = cfg["animation"]

        num_agents = m["num_agents"]
        mission_time = m["mission_time"]
        sample_time = m["sample_time"]
        speed = m["speed"]
        d_safe = m["d_safe"]

        # Override --- we want random seeds
        ll["iterations"] = 9
        ll["seed"] = None

        # ── Polygon ───────────────────────────────────────────────────────────────
        poly = Polygon(vertices)

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
                ll_crs=crs,
                utm_crs=utm_crs
            )
            print(f"[config] N_dots computed = {N_dots}")

        history_tessell, history_dots = Lloyd_algoritm(
            ll["iterations"], N_dots, poly, ll["partition"], ll["seed"]
        )
        
        final_tessellation = history_tessell[-1]
        final_dots = history_dots[-1]

        # ── Depot Coordinate ──────────────────────────────────────────────────────
        minx, miny, maxx, maxy = poly.bounds
        map_width = maxx - minx
        map_height = maxy - miny

        if d["mode"] == "offset":
            '''Intended for use when testing without known depot coords'''
            depot = [
                (minx + maxx) / 2 + d["offset_x"] * map_width,
                (miny + maxy) / 2 + d["offset_y"] * map_height,
            ]
            depot_coords = []
            for i in range(num_agents):
                depot_coords.append(depot)
        elif d["mode"] == "coordinate":
            '''Set each individual depot specifically'''
            depot_coords = []
            for i, depot in enumerate(d["depots"]):
                depot_coords.append(depot)

        # ── VRP ───────────────────────────────────────────────────────────────────
        coords = np.concat([depot_coords, final_dots.copy()])

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
        i = 0

        for route in routes:
            route_indices = [0] + list(route) + [0]
            coords[0] = np.array([d[f"depots"][i][0],d[f"depots"][i][1]]) 

            route_coords = [copy.copy(coords[i]) for i in route_indices]
            route_coords_3d = [np.append(copy.copy(coords[i]), out["drone_altitude"])
                            for i in route_indices]

            routes_coords.append(route_coords)
            routes_coords_3d.append(route_coords_3d)

            i += 1

        # Fit GP to sampled data
        # We want to work only on units of meters, so convert the CRS
        latlon_to_xy_tf = Transformer.from_crs(
            crs, utm_crs, always_xy=True
        )
        def tf(x, y):
            return latlon_to_xy_tf.transform(y, x) # because values here are latlon
        
        points_utm = points.to_crs(utm_crs)
        coords_utm = [[tf(coord[0],coord[1]) for coord in route] for route in routes_coords]
        poly_utm = transform(tf, poly) # applies CRS transform

        for aniso in [False]:
            for noise in [True]:
                for two_RBF in [False]:
                    df = repeat_gp(coords=coords_utm,points=points_utm,region=poly_utm,trials=1,voronoi=True,anisotropic=aniso,noise=noise,two_RBF=two_RBF)
                    df['region_name'] = data_path
                    full_df = pd.concat([full_df, df], ignore_index=True)

    full_df.to_csv("gp_metrics_Voronoi_farm12.csv",index=False)
    print(full_df)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        yaml_path = sys.argv[1]
    else:
        raise ValueError("Unexpected number of additional arguments --- expected 1 to specify YAML path")
    main(yaml_path=yaml_path)
