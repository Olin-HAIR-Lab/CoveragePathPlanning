import numpy as np
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import voronoi_diagram
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from grid_based_path_grant_version import planning_animation, do_animation

def sample_points_in_polygon(polygon, n=20000):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < n:
        p = Point(
            np.random.uniform(minx, maxx),
            np.random.uniform(miny, maxy)
        )
        if polygon.contains(p):
            points.append(p)
    return points

def split_polygon_equal_area(polygon, k=5, samples=2000):
    points = sample_points_in_polygon(polygon, samples)
    coords = np.array([(p.x, p.y) for p in points])

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(coords)
    centroids = kmeans.cluster_centers_

    centroid_points = MultiPoint(centroids)
    vor = voronoi_diagram(
        centroid_points,
        envelope=polygon.envelope,
        edges=False
    )

    parts = []
    for cell in vor.geoms:
        clipped = cell.intersection(polygon)
        if not clipped.is_empty:
            parts.append(clipped)

    return parts, centroids
    
def plot_polygon_split(original_polygon, parts, centroids=None):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot split parts
    for poly in parts:
        if poly.geom_type == "Polygon":
            x, y = poly.exterior.xy
            ax.fill(
                x, y,
                alpha=0.6,
                edgecolor="black",
                linewidth=1
            )

        elif poly.geom_type == "MultiPolygon":
            for p in poly.geoms:
                x, y = p.exterior.xy
                ax.fill(
                    x, y,
                    alpha=0.6,
                    edgecolor="black",
                    linewidth=1
                )

    # Plot original polygon outline
    ox, oy = original_polygon.exterior.xy
    ax.plot(ox, oy, color="black", linewidth=2)

    # Plot centroids (optional)
    if centroids is not None:
        cx, cy = zip(*centroids)
        ax.scatter(cx, cy, c="red", s=40, zorder=5)

    ax.set_aspect("equal")
    ax.set_title("Polygon Split via Voronoi + KMeans")
    ax.axis("off")
    plt.tight_layout()
    plt.show()



