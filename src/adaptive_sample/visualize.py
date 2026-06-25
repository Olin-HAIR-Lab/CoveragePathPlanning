import numpy as np
import matplotlib.pyplot as plt
from shapely import Point
import matplotlib.colors as colors

from sample_ground_truth import get_err

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

    # Get metrics
    rmse,nrmse,nrmse_std,rmse_over_std = get_err(gp=model.gp, points=points)

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
    axes[1].set_title(f"Final GP prediction\n{nrmse:.3f} NRMSE")

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
def plot_candidate_scores(
    model,
    rwd_fun,
    candidates,
    current_pos,
    region=None,
    visited_pts=None,
    title="Candidate reward scores",
    resolution=2.0,
):

    candidates = np.asarray(candidates, dtype=float).reshape(-1, 2)
    current_pos = np.asarray(current_pos, dtype=float).reshape(2,)

    cand_means, cand_stds = model.gp.predict(candidates, return_std=True)

    scores = np.array([
        rwd_fun.evaluate(float(mean), float(std), current_pos, candidate)
        for mean, std, candidate in zip(cand_means, cand_stds, candidates)
    ])

    minx, miny, maxx, maxy = region.bounds
    xs = np.arange(minx, maxx + resolution, resolution)
    ys = np.arange(miny, maxy + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    inside = np.array([region.contains(Point(x, y)) for x, y in grid])
    grid_inside = grid[inside]

    mean_grid = np.full(len(grid), np.nan)
    std_grid = np.full(len(grid), np.nan)

    pred_mean, pred_std = model.gp.predict(grid_inside, return_std=True)
    mean_grid[inside] = pred_mean
    std_grid[inside] = pred_std

    mean_grid = mean_grid.reshape(xx.shape)
    std_grid = std_grid.reshape(xx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    im0 = axes[0].imshow(
        mean_grid,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
    )
    axes[0].set_title("GP mean prediction")
    fig.colorbar(im0, ax=axes[0], label="Predicted moisture")

    im1 = axes[1].imshow(
        std_grid,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
    )
    axes[1].set_title("GP predictive std")
    fig.colorbar(im1, ax=axes[1], label="Predicted std")

    sc = axes[2].scatter(
        candidates[:, 0],
        candidates[:, 1],
        c=scores,
        s=45,
    )
    axes[2].set_title(title)
    fig.colorbar(sc, ax=axes[2], label="Reward score")

    if region is not None:
        bx, by = region.exterior.xy
        for ax in axes:
            ax.plot(bx, by, linewidth=2)

    if visited_pts is not None:
        visited_pts = np.asarray(visited_pts, dtype=float).reshape(-1, 2)

        for ax in axes:
            ax.plot(
                visited_pts[:, 0],
                visited_pts[:, 1],
                marker="o",
                linewidth=2,
                label="trajectory",
            )
            ax.scatter(
                current_pos[0],
                current_pos[1],
                marker="x",
                s=100,
                label="current position",
            )

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x offset (m)")
        ax.set_ylabel("y offset (m)")

    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return scores

def plot_candidate_paths(
    region,
    candidate_paths,
    scores=None,
    best_path=None,
    current_pos=None,
    visited_pts=None,
    max_paths_to_plot=200,
    title="Candidate paths",
):
    import numpy as np
    import matplotlib.pyplot as plt

    candidate_paths = np.asarray(candidate_paths, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Region boundary
    bx, by = region.exterior.xy
    ax.plot(bx, by, linewidth=2)

    # Current trajectory
    if visited_pts is not None:
        visited_pts = np.asarray(visited_pts, dtype=float).reshape(-1, 2)
        ax.plot(visited_pts[:, 0], visited_pts[:, 1], marker="o", linewidth=2, label="visited")

    # Downsample paths for readability
    n_paths = len(candidate_paths)
    idxs = np.arange(n_paths)

    if n_paths > max_paths_to_plot:
        idxs = np.random.choice(idxs, size=max_paths_to_plot, replace=False)

    # Optional score coloring
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        norm = plt.Normalize(scores.min(), scores.max())
        cmap = plt.cm.get_cmap('viridis')

    for idx in idxs:
        path = candidate_paths[idx]

        if current_pos is not None:
            start = np.asarray(current_pos, dtype=float).reshape(1, 2)
            path_plot = np.vstack([start, path])
        else:
            path_plot = path

        if scores is not None:
            ax.plot(
                path_plot[:, 0],
                path_plot[:, 1],
                alpha=0.25,
                linewidth=1,
                color=cmap(norm(scores[idx])),
            )
        else:
            ax.plot(
                path_plot[:, 0],
                path_plot[:, 1],
                alpha=0.15,
                linewidth=1,
            )

    # Highlight best path
    if best_path is not None:
        best_path = np.asarray(best_path, dtype=float).reshape(-1, 2)

        if current_pos is not None:
            start = np.asarray(current_pos, dtype=float).reshape(1, 2)
            best_path = np.vstack([start, best_path])

        ax.plot(
            best_path[:, 0],
            best_path[:, 1],
            marker="o",
            linewidth=3,
            label="best path",
        )

    if current_pos is not None:
        current_pos = np.asarray(current_pos, dtype=float).reshape(2,)
        ax.scatter(current_pos[0], current_pos[1], marker="x", s=100, label="current")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x offset (m)")
    ax.set_ylabel("y offset (m)")
    ax.legend()

    if scores is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, label="path score")

    plt.tight_layout()
    plt.show()