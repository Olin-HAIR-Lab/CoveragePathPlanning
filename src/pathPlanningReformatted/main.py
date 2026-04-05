import math
import numpy as np
import copy
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer

from lloydsAlgorithm import Lloyd_algoritm
from vehicleRoutingProblem import (
    solve_vrp_unlimited,
    solve_vrp_balanced,
    extract_paths
)
from collisionAvoidance import (
    add_delays_to_avoid_collisions,
    animate_trajectories
)
from plotting import plot_results
from jsonTesting import makeJSONMission


def polygon_area_m2(poly_latlon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)
    poly_utm = transform(transformer.transform, poly_latlon)
    return poly_utm.area


def compute_sample_count(map, sample_time, speed, mission_time, num_agents=1):
    area = polygon_area_m2(map)
    max_possible = int((mission_time * num_agents) / sample_time)
    for N in range(1, max_possible + 1):

        if N % num_agents != 0:
            continue  
        N_per_agent = N / num_agents

        sampling_time = N_per_agent * sample_time
        travel_time = math.sqrt(area * N_per_agent) / speed

        total_time = sampling_time + travel_time

        if total_time <= mission_time:
            best_N = N
        else:
            break  

    return best_N


# ================= SETTINGS =================

poly = Polygon([
    (-71.2633943948142, 42.29151053590485),
    (-71.2628929214584, 42.290784490885386),
    (-71.26149845009414, 42.291353821977765),
    (-71.26200796847752, 42.29205010492521),
    (-71.2633943948142, 42.29151053590485)
])

N_dots = compute_sample_count(
    poly,
    sample_time=180,
    speed=5.0,
    mission_time=900,
    num_agents=3
)

N_dots = 15
# N_dots = 30

iterations = 10
partition = 300
seed = 6

# ================= LLOYD =================

history_tessell, history_dots = Lloyd_algoritm(
    iterations, N_dots, poly, partition, seed
)

final_tessellation = history_tessell[-1]
final_dots = history_dots[-1]




# ================= VRP =================

minx, miny, maxx, maxy = poly.bounds
map_width = maxx - minx
map_height = maxy - miny
depot_coord = np.array([(minx+maxx)/2 - map_width*0.25, (miny + maxy)/2 - map_height])

# depot_coord1 = np.array([42.29146,-71.26296])
# depot_coord2 = np.array([42.29146,-71.26292])
# depot_coord3 = np.array([42.29146,-71.26288])

coords = np.vstack([depot_coord, final_dots.copy()])

travel_duration_matrix = np.empty((len(coords), len(coords)))

for i in range(len(coords)):
    for j in range(len(coords)):
        travel_duration_matrix[i][j] = np.hypot(
            coords[i][0] - coords[j][0],
            coords[i][1] - coords[j][1]
        )

max_time = 900
max_dist = travel_duration_matrix.max()
travel_duration_matrix = travel_duration_matrix / max_dist * 100

time_windows = np.array([
    (0, max_time - travel_duration_matrix[i][0])
    for i in range(len(coords))
])



# ===== CHOOSE MODE =====

# Recharge
# solution = solve_vrp_unlimited(coords, time_windows, travel_duration_matrix)

# No Recharge
solution = solve_vrp_balanced(coords, time_windows, travel_duration_matrix, num_vehicles=3)


paths, routes = extract_paths(solution, coords)

# coords: list of (x, y) or (lon, lat)
routes_coords = []
routes_coords_3d = []  # NEW: same structure but with z=3
for r_idx, route in enumerate(solution.routes()):
    route_indices = [0] + list(route) + [0]
    coords[0][0] += map_height*0.3
    # coords[0][0] += 0.00004
    route_coords = [copy.copy(coords[i]) for i in route_indices]
    routes_coords.append(route_coords)
    
    # Build the 3D version by appending z=3 to each point
    route_coords_3d = [np.append(copy.copy(coords[i]), 3) for i in route_indices]  # NEW
    routes_coords_3d.append(route_coords_3d)

# Collision Detection
trajectories, delays = add_delays_to_avoid_collisions(routes_coords, speed=5, d_safe=20)

# JSON output
makeJSONMission("output.json", *routes_coords_3d[:3])

# Plot
# plot_results(poly, final_tessellation, coords, solution)

# Animation
animate_trajectories(
    trajectories,
    routes_coords,
    speed_multiplier=8,
    poly=poly,
    tessellation=final_tessellation,
    map_coords=coords,
    solution=solution,
)