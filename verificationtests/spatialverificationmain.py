import math
import copy
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import transform, voronoi_diagram
from shapely.geometry import MultiPoint, GeometryCollection
from pyproj import Transformer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'pathPlanningReformatted'))

from lloydsAlgorithm import Lloyd_algoritm
from vehicleRoutingProblem import solve_vrp_balanced, extract_paths
from collisionAvoidance import add_delays_to_avoid_collisions


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', 'src', 'config.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def polygon_area_m2(poly_latlon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)
    def _swap_xy(x, y, z=None):
        return transformer.transform(y, x)
    return transform(_swap_xy, poly_latlon).area


# ── Spatial distribution (Voronoi zone per depot) ────────────────────────────

def compute_depot_zones(poly, depot_coords):
    seeds = MultiPoint(depot_coords)
    regions = voronoi_diagram(seeds, envelope=poly)
    zones = []
    for region in regions.geoms:
        clipped = region.intersection(poly)
        if not clipped.is_empty:
            zones.append(clipped)

    # match each zone to its nearest depot
    ordered = []
    used = set()
    for depot in depot_coords:
        from shapely.geometry import Point
        depot_pt = Point(depot)
        available = [i for i in range(len(zones)) if i not in used]
        
        if not available:
            break  # fewer zones than depots, skip remaining
            
        best_idx = min(available, key=lambda i: zones[i].distance(depot_pt))
        ordered.append(zones[best_idx])
        used.add(best_idx)

    return ordered

def spatial_mission_time(poly, depot_coords, N_dots, speed, sample_time,
                          lloyd_iterations=50, seed=42):
    """
    Approach A: each drone gets its own Voronoi zone and samples only within it.
    Returns the bottleneck (slowest drone) mission time in seconds.
    """
    zones = compute_depot_zones(poly, depot_coords)
    num_agents = len(depot_coords)
    times = []

    for i, (zone, depot) in enumerate(zip(zones, depot_coords)):
        zone_fraction = zone.area / poly.area
        n_zone = max(1, round(N_dots * zone_fraction))

        try:
            _, history_dots = Lloyd_algoritm(lloyd_iterations, n_zone, zone,
                                            num_agents, seed)
            pts = history_dots[-1]
        except Exception as e:
            print(f"  Skipping zone {i}: {e}")
            continue

        _, history_dots = Lloyd_algoritm(lloyd_iterations, n_zone, zone,
                                          num_agents, seed)
        pts = history_dots[-1]

        depot = np.array(depot)
        pts   = np.array(pts)

        # greedy nearest-neighbor route from depot
        unvisited = list(range(len(pts)))
        current   = depot
        total_dist = 0
        while unvisited:
            dists  = [np.linalg.norm(pts[j] - current) for j in unvisited]
            nearest = unvisited[np.argmin(dists)]
            total_dist += np.min(dists)
            current = pts[nearest]
            unvisited.remove(nearest)
        # return to depot
        total_dist += np.linalg.norm(current - depot)

        travel_time = total_dist / speed
        probe_time  = n_zone * sample_time
        times.append(travel_time + probe_time)
    
    return max(times)   # bottleneck drone


# ── Temporal distribution (your existing VRP pipeline) ───────────────────────

def temporal_mission_time(poly, depot_coords, N_dots, speed, sample_time,
                           mission_time_budget, lloyd_iterations=50, seed=42):
    """
    Approach B: VRP distributes all points optimally across all drones.
    Returns the bottleneck (slowest drone) mission time in seconds after delays.
    """
    num_agents = len(depot_coords)

    _, history_dots = Lloyd_algoritm(lloyd_iterations, N_dots, poly,
                                      num_agents, seed)
    final_dots = history_dots[-1]

    depots = [np.array(d) for d in depot_coords]
    coords = np.vstack([*depots, final_dots])

    travel_duration_matrix = np.array([
        [np.linalg.norm(coords[i] - coords[j]) for j in range(len(coords))]
        for i in range(len(coords))
    ])

    max_dist = travel_duration_matrix.max()
    travel_duration_matrix = travel_duration_matrix / max_dist * 100

    time_windows = np.array([
        (0, mission_time_budget - travel_duration_matrix[i][0])
        for i in range(len(coords))
    ])

    solution = solve_vrp_balanced(
        coords, time_windows, travel_duration_matrix, num_vehicles=num_agents
    )

    routes_coords = []
    for drone_idx, route in enumerate(solution.routes()):
        route_indices = [drone_idx] + list(route) + [drone_idx]
        route_coords  = [copy.copy(coords[i]) for i in route_indices]
        routes_coords.append(route_coords)

    trajectories, _ = add_delays_to_avoid_collisions(
        routes_coords, speed=speed, d_safe=0.0001
    )

    # mission time = length of longest trajectory
    times = []
    for traj in trajectories:
        t_end = float(traj[-1][3]) if traj else 0.0
        times.append(t_end)
    return max(times)
# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(cfg):
    m    = cfg["mission"]
    ll   = cfg["lloyd"]
    d    = cfg["depot"]

    speed       = m["speed"]
    sample_time = m["sample_time"]
    mission_budget = m["mission_time"]
    num_agents  = m["num_agents"]

    poly = Polygon([tuple(pt) for pt in cfg["polygon"]["vertices"]])

    depot_coords = [
        [d["depot1_x"], d["depot1_y"]],
        [d["depot2_x"], d["depot2_y"]],
        [d["depot3_x"], d["depot3_y"]],
    ]

    # ── Sweep 1: vary sample points ───────────────────────────────────────────
    point_range = range(num_agents, 10 * num_agents + 1, num_agents)
    spatial_by_points  = []
    temporal_by_points = []

    print("Sweeping sample points...")
    for N in point_range:
        print(f"  N={N}")
        spatial_by_points.append(
            spatial_mission_time(poly, depot_coords, N, speed, sample_time,
                                  ll["iterations"], ll["seed"]) / 60
        )
        temporal_by_points.append(
            temporal_mission_time(poly, depot_coords, N, speed, sample_time,
                                   mission_budget, ll["iterations"], ll["seed"]) / 60
        )

    # ── Sweep 2: vary plot area (scale polygon) ───────────────────────────────
    scale_factors = np.linspace(0.5, 3.0, 10)
    cx = sum(p[0] for p in cfg["polygon"]["vertices"]) / len(cfg["polygon"]["vertices"])
    cy = sum(p[1] for p in cfg["polygon"]["vertices"]) / len(cfg["polygon"]["vertices"])

    N_fixed = num_agents * 3   # fixed point count for area sweep
    spatial_by_area  = []
    temporal_by_area = []
    areas_m2         = []

    print("Sweeping plot area...")
    for s in scale_factors:
        scaled_verts = [
            [(p[0] - cx) * s + cx, (p[1] - cy) * s + cy]
            for p in cfg["polygon"]["vertices"]
        ]
        scaled_poly = Polygon([tuple(p) for p in scaled_verts])
        areas_m2.append(polygon_area_m2(scaled_poly))
        print(f"  scale={s:.2f}, area={areas_m2[-1]:.0f} m²")

        spatial_by_area.append(
            spatial_mission_time(scaled_poly, depot_coords, N_fixed, speed,
                                  sample_time, ll["iterations"], ll["seed"]) / 60
        )
        temporal_by_area.append(
            temporal_mission_time(scaled_poly, depot_coords, N_fixed, speed,
                                   sample_time, mission_budget, ll["iterations"],
                                   ll["seed"]) / 60
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -- subplot 1: vary sample points
    ax1.plot(list(point_range), spatial_by_points,
             color='steelblue', linewidth=2, label='Spatial distribution')
    ax1.plot(list(point_range), temporal_by_points,
             color='coral', linewidth=2, label='Temporal distribution')
    ax1.fill_between(list(point_range), temporal_by_points, spatial_by_points,
                     alpha=0.15, color='green', label='efficiency gain')
    ax1.set_xlabel('Number of sample points')
    ax1.set_ylabel('Mission time — slowest drone (min)')
    ax1.set_title('Varying sample points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -- subplot 2: vary plot area
    areas_ha = [a / 10000 for a in areas_m2]
    ax2.plot(areas_ha, spatial_by_area,
             color='steelblue', linewidth=2, label='Spatial distribution')
    ax2.plot(areas_ha, temporal_by_area,
             color='coral', linewidth=2, label='Temporal distribution')
    ax2.fill_between(areas_ha, temporal_by_area, spatial_by_area,
                     alpha=0.15, color='green', label='efficiency gain')
    ax2.set_xlabel('Plot area (hectares)')
    ax2.set_ylabel('Mission time — slowest drone (min)')
    ax2.set_title('Varying plot area')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Spatial vs temporal distribution — bottleneck drone mission time',
                 fontsize=13)
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    run_sweep(cfg)
