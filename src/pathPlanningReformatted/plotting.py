import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull


def _plot_coords(point, coord_order="lonlat"):
    if coord_order == "lonlat":
        return point[0], point[1]
    if coord_order == "latlon":
        return point[0], point[1]
    if coord_order == "xy":
        return point[0], point[1]
    raise ValueError(f"Unknown coord_order: {coord_order}")


def plot_results(poly, tessellation, coords, solution, coord_order="lonlat", datapoints=None, means=None, stds=None):
    plt.figure(figsize=(7, 7))

    # Polygon
    poly_coords = np.array(poly.exterior.coords)
    if coord_order in ("latlon", "lonlat"):
        poly_plot = np.array([
            _plot_coords(pt, coord_order=coord_order)
            for pt in poly_coords
        ])
        plt.plot(poly_plot[:, 0], poly_plot[:, 1], 'k-', linewidth=2)
    else:
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

        regions = [cell_poly] if isinstance(
            cell_poly, Polygon) else list(cell_poly.geoms)

        for region in regions:
            region_coords = np.array(region.exterior.coords)
            if coord_order in ("latlon", "lonlat"):
                region_coords = np.array([
                    _plot_coords(pt, coord_order=coord_order)
                    for pt in region_coords
                ])
            x_reg, y_reg = region_coords[:, 0], region_coords[:, 1]
            plt.plot(x_reg, y_reg, linewidth=2, alpha=0.8, color="black")

    # Points
    if coord_order in ("latlon", "lonlat"):
        coords_plot = np.array([
            _plot_coords(pt, coord_order=coord_order)
            for pt in coords
        ])
    else:
        coords_plot = coords

    plt.scatter(coords_plot[:, 0], coords_plot[:, 1], c='black', s=40)

    # Depot
    plt.scatter(coords_plot[0][0], coords_plot[0]
                [1], c='red', s=250, marker='*')

    # Routes
    colors = ['blue', 'green', 'purple', 'orange']

    for r_idx, route in enumerate(solution.routes()):
        route_indices = [0] + list(route) + [0]

        xs = [coords_plot[i][0] for i in route_indices]
        ys = [coords_plot[i][1] for i in route_indices]

        plt.plot(xs, ys, color=colors[r_idx % len(colors)], linewidth=2)
    
    # # Data
    # if datapoints is not None:
    #     # Plot contained points (have to convert them back to lonlat)
    #     x = [point.y for point in datapoints.geometry]
    #     y = [point.x for point in datapoints.geometry]
    #     plt.scatter(x,y,c=datapoints['Moisture'].to_numpy(),s=20,marker='o',edgecolors='k',linewidths=0,cmap="RdBu",vmin=12,vmax=20)

    # Prediction
    

    plt.title("Drone Trajectories (Top View)", fontsize=18)
    plt.xlabel("Latitude", fontsize=14)
    plt.ylabel("Longitude", fontsize=14)
    plt.axis('equal')
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
    plt.show()
