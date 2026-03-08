# ================== IMPORTS ==================
import matplotlib
matplotlib.use('TkAgg')  # interactive backend (must be before pyplot)

from outline import image_to_polygon_points
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from shapely.prepared import prep
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
import pyvrp
import pyvrp.plotting
import pyvrp.stop


# ================== LLOYD CORE ==================

def varinoci(dots, domain_poly, partition):
    tessellation = {}
    minx, miny, maxx, maxy = domain_poly.bounds
    X = np.linspace(minx, maxx, partition)
    Y = np.linspace(miny, maxy, partition)
    prepared = prep(domain_poly)

    for xi in X:
        for yj in Y:
            p = np.array([xi, yj])
            if not prepared.contains(Point(p)):
                continue

            minimo = float("inf")
            k_star = None

            for k, dot in enumerate(dots):
                value = LA.norm(p - dot)
                if value < minimo:
                    minimo = value
                    k_star = k

            if k_star is not None:
                tessellation.setdefault(k_star, []).append(p)

    return tessellation


def new_centroids(tessellation, old_dots):
    new_dots = []

    for i in range(len(old_dots)):
        if i in tessellation and len(tessellation[i]) > 0:
            xi = np.array(tessellation[i])
            new_dots.append(np.mean(xi, axis=0))
        else:
            # preserve centroid if cell empty
            new_dots.append(old_dots[i])

    return np.array(new_dots)


def Lloyd_algoritm(Iterations, N, domain_poly, partition, seed=None):
    if seed is not None:
        np.random.seed(seed)

    minx, miny, maxx, maxy = domain_poly.bounds
    new_dots = []

    while len(new_dots) < N:
        cand = np.random.uniform([minx, miny], [maxx, maxy])
        if domain_poly.contains(Point(cand)):
            new_dots.append(cand)

    new_dots = np.array(new_dots)

    history_tessell = []
    history_dots = [new_dots]

    for itera in range(Iterations):
        tessellation = varinoci(new_dots, domain_poly, partition)
        new_dots = new_centroids(tessellation, new_dots)

        history_tessell.append(tessellation)
        history_dots.append(new_dots)

        print(f"Iteration {itera} complete.")

    return history_tessell, history_dots

def make_lloyd_gif(history_tessell, history_dots, poly, N_dots, filename="lloyd.gif"):
    import imageio.v2 as imageio
    import os

    tmpdir = "/tmp/lloyd_frames"
    os.makedirs(tmpdir, exist_ok=True)

    frame_paths = []

    for itera, (tessellation, dots) in enumerate(zip(history_tessell, history_dots[1:])):
        fig, ax = plt.subplots(figsize=(7, 7))

        x_poly, y_poly = poly.exterior.xy
        ax.plot(x_poly, y_poly, 'k-', linewidth=2)

        for k in range(N_dots):
            if k not in tessellation:
                continue
            pts = np.array(tessellation[k])
            if len(pts) < 3:
                continue
            hull = ConvexHull(pts)
            cell_poly = Polygon(pts[hull.vertices]).intersection(poly)
            if cell_poly.is_empty:
                continue
            regions = [cell_poly] if isinstance(cell_poly, Polygon) else list(cell_poly.geoms)
            for region in regions:
                x_reg, y_reg = region.exterior.xy
                ax.fill(x_reg, y_reg, alpha=0.3)

        ax.scatter(dots[:, 0], dots[:, 1], c='black', s=40)
        ax.set_title(f"Lloyd Iteration {itera + 1}")
        ax.axis('equal')

        path = os.path.join(tmpdir, f"frame_{itera:03d}.png")
        fig.savefig(path)
        plt.close(fig)
        frame_paths.append(path)

    with imageio.get_writer(filename, mode='I', duration=0.4) as writer:
        for path in frame_paths:
            writer.append_data(imageio.imread(path))

    print(f"GIF saved to {filename}")



# ================== SETTINGS ==================

image_path = "usa.png"
poly, _ = image_to_polygon_points(
    image_path,
    num_points=200,
    scale_to_range=(-16, 8)
)

# poly = Polygon([
#     (-71.2633943948142, 42.29151053590485),
#     (-71.2628929214584, 42.290784490885386),
#     (-71.26149845009414, 42.291353821977765),
#     (-71.26200796847752, 42.29205010492521),
#     (-71.2633943948142, 42.29151053590485)])

N_dots = 9
iterations = 15
partition = 600
seed = 6


# ================== RUN LLOYD ==================

history_tessell, history_dots = Lloyd_algoritm(
    iterations, N_dots, poly, partition, seed
)

final_tessellation = history_tessell[-1]
final_dots = history_dots[-1]  # THIS is the single source of truth

make_lloyd_gif(history_tessell, history_dots, poly, N_dots)


# ================== BUILD VRP DIRECTLY FROM final_dots ==================

minx, miny, maxx, maxy = poly.bounds
map_width = maxx - minx
depot_coord = np.array([minx - map_width * 0.05, (miny + maxy) / 2])

# Prepend depot to coords array
coords = np.vstack([depot_coord, final_dots.copy()])
# coords = final_dots.copy()
travel_duration_matrix = np.empty((len(coords), len(coords)))
time_windows = np.empty((len(coords), 2))
for i in range(len(coords)):
    for j in range(len(coords)):
        travel_duration_matrix[i][j] = float(np.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1]))

max_time = 2000
max_dist = travel_duration_matrix.max()
travel_duration_matrix = travel_duration_matrix / max_dist * 100
#travel_duration_matrix = travel_duration_matrix * 50000
print(travel_duration_matrix)

for i in range(len(coords)):
    time_windows[i] = (0,max_time-travel_duration_matrix[i][0])

# m = pyvrp.Model()
# m.add_depot(
#     x=coords[0][0],
#     y=coords[0][1],
#     name="Depot",
# )
# m.add_vehicle_type(3, capacity=3)
# for i in range(1, len(coords)):
#     m.add_client(
#         x=coords[i][0],
#         y=coords[i][1],
#         delivery=1,
#         name=f"Client {i}",
#     )
# for i, frm in enumerate(m.locations):
#     for j, to in enumerate(m.locations):
#         dx = coords[i][0] - coords[j][0]
#         dy = coords[i][1] - coords[j][1]
#         dist = int(np.hypot(dx, dy) * 100000)
#         m.add_edge(frm, to, distance=dist)
# res = m.solve(stop=pyvrp.stop.MaxRuntime(5))
# solution = res.best

m = pyvrp.Model()
m.add_vehicle_type(3, unit_distance_cost=0, unit_duration_cost=1)
m.add_depot(
    x=coords[0][0],
    y=coords[0][1],
    tw_early=time_windows[0][0],
    tw_late=time_windows[0][1],
    name="Depot",
)
for idx in range(1, len(coords)):
    m.add_client(
        x=coords[idx][0],
        y=coords[idx][1],
        tw_early=time_windows[idx][0],
        tw_late=time_windows[idx][1],
        service_duration=180,
        name=f"Client {idx}",
    )
for frm_idx, frm in enumerate(m.locations):
    for to_idx, to in enumerate(m.locations):
        duration = travel_duration_matrix[frm_idx][to_idx]
        m.add_edge(frm, to, distance=duration, duration=duration)
res = m.solve(stop=pyvrp.stop.MaxRuntime(1))
solution = res.best









# ================== PLOT FINAL LLOYD + VRP ==================

plt.figure(figsize=(7, 7))

# Polygon border
x_poly, y_poly = poly.exterior.xy
plt.plot(x_poly, y_poly, 'k-', linewidth=2)

# Tessellation cells
for k in range(N_dots):
    if k not in final_tessellation:
        continue

    pts = np.array(final_tessellation[k])
    if len(pts) < 3:
        continue

    hull = ConvexHull(pts)
    cell_poly = Polygon(pts[hull.vertices]).intersection(poly)

    if cell_poly.is_empty:
        continue

    regions = [cell_poly] if isinstance(cell_poly, Polygon) else list(cell_poly.geoms)

    for region in regions:
        x_reg, y_reg = region.exterior.xy
        plt.fill(x_reg, y_reg, alpha=0.3)

# Plot centroids
plt.scatter(coords[:, 0], coords[:, 1], c='black', s=40)

# Highlight depot
plt.scatter(
    coords[0][0],
    coords[0][1],
    c='red',
    s=250,
    marker='*',
    zorder=5
)

# Overlay VRP routes
colors = ['blue', 'green', 'purple']

for r_idx, route in enumerate(solution.routes()):
    route_indices = [0]  # start at depot

    for visit in route:
        route_indices.append(visit)

    route_indices.append(0)  # return to depot

    xs = [coords[i][0] for i in route_indices]
    ys = [coords[i][1] for i in route_indices]

    plt.plot(xs, ys, color=colors[r_idx % len(colors)], linewidth=2)

plt.title("Final Lloyd Iteration + VRP Routes")
plt.axis('equal')
plt.show()