import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull


def plot_results(poly, tessellation, coords, solution):
    plt.figure(figsize=(7, 7))

    # Polygon
    x_poly, y_poly = poly.exterior.xy
    plt.plot(x_poly, y_poly, 'k-', linewidth=2)

    # Tessellation
    for k in tessellation:
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
            plt.fill(x_reg, y_reg, alpha=0.3)

    # Points
    plt.scatter(coords[:, 0], coords[:, 1], c='black', s=40)

    # Depot
    plt.scatter(coords[0][0], coords[0][1], c='red', s=250, marker='*')

    # Routes
    colors = ['blue', 'green', 'purple', 'orange']

    for r_idx, route in enumerate(solution.routes()):
        route_indices = [0] + list(route) + [0]

        xs = [coords[i][0] for i in route_indices]
        ys = [coords[i][1] for i in route_indices]

        plt.plot(xs, ys, color=colors[r_idx % len(colors)], linewidth=2)

    plt.title("Final Lloyd + VRP Routes")
    plt.axis('equal')
    plt.show()