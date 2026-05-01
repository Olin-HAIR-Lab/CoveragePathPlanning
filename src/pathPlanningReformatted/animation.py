from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import ConvexHull
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import math
import matplotlib
matplotlib.use("TkAgg")

# ===============================
# Geographic helpers
# ===============================

EARTH_R = 6_371_000  # metres


def dist(p1, p2):
    """
    Haversine distance in metres between two (lat, lon) points.
    """
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _lat_aspect(lats):
    """
    Returns the aspect ratio 1/cos(mean_lat) to correct lon/lat distortion.
    """
    mean_lat = math.radians(sum(lats) / len(lats))
    return 1.0 / math.cos(mean_lat)


def _plot_coords(point, coord_order="lonlat"):
    """Convert input point to plotting coordinates.

    coord_order is the ordering of the input point:
    - "lonlat": (longitude, latitude)
    - "latlon": (latitude, longitude)
    - "xy": plain XY coordinates
    """
    if coord_order == "lonlat":
        return point[0], point[1]
    if coord_order == "latlon":
        return point[1], point[0]
    if coord_order == "xy":
        return point[0], point[1]
    raise ValueError(f"Unknown coord_order: {coord_order}")


def interpolate(p1, p2, t_ratio):
    return (
        p1[0] + (p2[0] - p1[0]) * t_ratio,
        p1[1] + (p2[1] - p1[1]) * t_ratio,
    )

# ===============================
# Sample position and state at time t
# ===============================

def state_at(traj, t):
    for seg in traj:
        p1, p2, t1, t2, kind = seg
        if t1 <= t <= t2:
            ratio = (t - t1) / (t2 - t1 + 1e-9)
            return interpolate(p1, p2, ratio), kind
    return traj[-1][1], "done"


def position_at(traj, t):
    return state_at(traj, t)[0]

def animate_trajectories(trajectories, routes, dt=0.5, trail_length=30,
                         speed_multiplier=8,
                         poly=None, tessellation=None,
                         map_coords=None, solution=None,
                         coord_order="latlon", d=None,
                         save_path=None):

    t_start = 0.0
    t_end = max(seg[3] for traj in trajectories for seg in traj)
    total_duration = t_end - t_start

    interval_ms = max(16, int(1000 * dt / speed_multiplier))
    times = np.arange(t_start, t_end + dt, dt)

    all_states = [
        [state_at(traj, t) for t in times]
        for traj in trajectories
    ]

    # ── Figure Setup ──────────────────────────────────────────────────────────
    raw_points = [pt for route in routes for pt in route]
    if map_coords is not None:
        raw_points.extend(map_coords.tolist())
    if poly is not None:
        raw_points.extend(poly.exterior.coords)

    all_x = []
    all_y = []
    for pt in raw_points:
        x_plot, y_plot = _plot_coords(pt, coord_order=coord_order)
        all_x.append(x_plot)
        all_y.append(y_plot)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    range_x = (max_x - min_x) if max_x != min_x else 0.001
    range_y = (max_y - min_y) if max_y != min_y else 0.001
    pad = max(range_x, range_y) * 0.12
    pad_x = pad
    pad_y = pad

    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 8),
        gridspec_kw={"width_ratios": [20, 1]}
    )
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

    ax.set_title("Drone Trajectories (Top View)", fontsize=18)
    if coord_order in ("lonlat", "latlon"):
        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)
        ax.set_aspect(_lat_aspect(all_y), adjustable="box")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.5f}°"))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.5f}°"))
    else:
        ax.set_xlabel("X", fontsize=14)
        ax.set_ylabel("Y", fontsize=14)
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    if coord_order in ("lonlat", "latlon"):
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # ── Static map background ──────────────────────────────────────────────
    if poly is not None:
        if coord_order in ("latlon", "lonlat"):
            poly_plot = np.array([
                _plot_coords(pt, coord_order=coord_order)
                for pt in poly.exterior.coords
            ])
            ax.plot(poly_plot[:, 0], poly_plot[:, 1],
                    'k-', linewidth=2, zorder=1)
        else:
            x_poly, y_poly = poly.exterior.xy
            ax.plot(x_poly, y_poly, 'k-', linewidth=2, zorder=1)

    if tessellation is not None and poly is not None:
        for k in tessellation:
            pts = np.array(tessellation[k])
            if len(pts) < 3:
                continue
            hull = ConvexHull(pts)
            cell_poly = ShapelyPolygon(pts[hull.vertices]).intersection(poly)
            if cell_poly.is_empty:
                continue
            regions = ([cell_poly] if isinstance(cell_poly, ShapelyPolygon)
                       else list(cell_poly.geoms))
            for region in regions:
                region_coords = np.array(region.exterior.coords)
                if coord_order in ("latlon", "lonlat"):
                    region_coords = np.array([
                        _plot_coords(pt, coord_order=coord_order)
                        for pt in region_coords
                    ])
                x_reg, y_reg = region_coords[:, 0], region_coords[:, 1]
                ax.fill(x_reg, y_reg, alpha=0.3, zorder=1)

    if map_coords is not None:
        map_plot = np.array([_plot_coords(pt, coord_order=coord_order)
                             for pt in map_coords])
        ax.scatter(map_plot[:, 0], map_plot[:, 1],
                   c='black', s=40, zorder=2)


    if solution is not None and map_coords is not None:
        route_colors = ['blue', 'orange', 'green', 'orange']

        for r_idx, route in enumerate(solution.routes()):
            route_indices = [0] + list(route) + [0]
            xs = [map_plot[i][0] for i in route_indices]
            ys = [map_plot[i][1] for i in route_indices]

            xs[0] = d[f"depots"][r_idx][1]
            xs[-1] = d[f"depots"][r_idx][1]
            ys[0] = d[f"depots"][r_idx][0]
            ys[-1] = d[f"depots"][r_idx][0]

            ax.plot(xs, ys, color=route_colors[r_idx % len(route_colors)],
                    linewidth=1.5, alpha=0.4, zorder=2)
    
    if d is not None:
        for i in range(3):
            ax.scatter(d[f"depots"][i][1], d[f"depots"][i][0],
                c='red', s=250, marker='*', zorder=3)
        
    # ── End static map background ──────────────────────────────────────────

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    points, trails, rings = [], [], []
    for k in range(len(trajectories)):
        c = colors[k % len(colors)]
        pt, = ax.plot([], [], "o", color=c, ms=8, zorder=5)
        tr, = ax.plot([], [], "-", color=c, lw=1.5, alpha=0.6)
        rg, = ax.plot([], [], "o", color=c, ms=18, zorder=4,
                      markerfacecolor="none", markeredgewidth=1.5, alpha=0.5)
        points.append(pt)
        trails.append(tr)
        rings.append(rg)

    # ── Progress bar ──────────────────────────────────────────────────────────
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_title("Time", fontsize=9)
    ax_bar.add_patch(mpatches.Rectangle(
        (0.2, 0), 0.6, 1.0, fc="#e0e0e0", ec="#aaaaaa"))
    bar_fill = mpatches.Rectangle((0.2, 0), 0.6, 0.0, fc="#2196F3")
    ax_bar.add_patch(bar_fill)
    pct_text = ax_bar.text(0.5, -0.05, "0%", ha="center", va="top", fontsize=9)

    def update(frame):
        for i, states in enumerate(all_states):
            pos, kind = states[frame]
            x, y = _plot_coords(pos, coord_order=coord_order)
            points[i].set_data([x], [y])
            s = max(0, frame - trail_length)
            trails[i].set_data([
                _plot_coords(states[k][0], coord_order=coord_order)[0]
                for k in range(s, frame)
            ], [
                _plot_coords(states[k][0], coord_order=coord_order)[1]
                for k in range(s, frame)
            ])
            if kind == "hold":
                rings[i].set_data([x], [y])
            else:
                rings[i].set_data([], [])

        progress = (times[frame] - t_start) / \
            total_duration if total_duration > 0 else 1
        bar_fill.set_height(min(progress, 1.0))
        pct_text.set_text(f"{progress * 100:.0f}%")

    ani = FuncAnimation(fig, update, frames=len(times),
                        interval=interval_ms, blit=False, repeat=False)
    if save_path:
        ani.save(save_path, writer="pillow", fps=20)
    plt.show()
