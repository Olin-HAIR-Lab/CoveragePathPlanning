import numpy as np
import matplotlib.pyplot as plt
from shapely import Point
import matplotlib.colors as colors

def plot_results(points, region, model, visited_pts, resolution=2.0):
    minx, miny, maxx, maxy = region.bounds

    xs = np.arange(minx, maxx + resolution, resolution)
    ys = np.arange(miny, maxy + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)

    grid = np.column_stack([xx.ravel(), yy.ravel()])

    inside = np.array([region.contains(Point(x, y)) for x, y in grid])
    grid_inside = grid[inside]

    pred = np.full(len(grid), np.nan)

    if len(model.y) > 0:
        pred_inside = model.gp.predict(grid_inside)
        pred[inside] = pred_inside

    pred = pred.reshape(xx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    vmin = points["Moisture"].quantile(0.02)
    vmax = points["Moisture"].quantile(0.98)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Original data
    points.plot(
        ax=axes[0],
        column="Moisture",
        markersize=3,
        cmap="RdYlGn",
        norm=norm,
    )
    axes[0].set_title("Original moisture data")

    # GP prediction
    im = axes[1].imshow(
        pred,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        cmap="RdYlGn",
        norm=norm,
    )
    axes[1].set_title("Final GP prediction")

    # Trajectory
    points.plot(
        ax=axes[2],
        column="Moisture",
        markersize=2,
        cmap="RdYlGn",
        norm=norm,
        alpha=0.4,
    )

    visited_pts = np.asarray(visited_pts)

    axes[2].plot(
        visited_pts[:, 0],
        visited_pts[:, 1],
        marker="o",
        linewidth=2,
    )
    axes[2].scatter(
        visited_pts[0, 0],
        visited_pts[0, 1],
        marker="s",
        s=80,
        label="start",
    )
    axes[2].scatter(
        visited_pts[-1, 0],
        visited_pts[-1, 1],
        marker="*",
        s=120,
        label="end",
    )
    axes[2].legend()
    axes[2].set_title("Sample trajectory")

    # Region boundary on all axes
    boundary_x, boundary_y = region.exterior.xy
    for ax in axes:
        ax.plot(boundary_x, boundary_y, linewidth=2)
        ax.set_aspect("equal")
        ax.set_xlabel("x offset (m)")
        ax.set_ylabel("y offset (m)")

    fig.colorbar(im, ax=axes, label="Moisture")
    plt.show()