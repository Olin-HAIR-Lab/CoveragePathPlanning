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
# Build time-parameterized trajectory
# ===============================

def build_timed_trajectory(route, speed, sample_time=5, extra_waits=None):
    if extra_waits is None:
        extra_waits = [0.0] * len(route)

    traj = []
    t = 0.0

    for i in range(len(route) - 1):
        p1, p2 = route[i], route[i + 1]

        wait = extra_waits[i]
        if wait > 0:
            traj.append((p1, p1, t, t + wait, "hold"))
            t += wait

        leg_dt = dist(p1, p2) / speed
        traj.append((p1, p2, t, t + leg_dt, "travel"))
        t += leg_dt

        if i < len(route) - 2:
            traj.append((p2, p2, t, t + sample_time, "sample"))
            t += sample_time

    return traj


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


# ===============================
# Collision detection
# ===============================

def first_collision_time(traj1, traj2, d_safe=2.0, dt=0.5):
    t_start = min(traj1[0][2], traj2[0][2])
    t_end = max(traj1[-1][3], traj2[-1][3])
    t = t_start
    while t <= t_end:
        if dist(position_at(traj1, t), position_at(traj2, t)) < d_safe:
            return t
        t += dt
    return None


# ===============================
# Find which waypoint to hold at
# ===============================

def waypoint_before_collision(route, speed, extra_waits, t_col, sample_time=5):
    t = 0.0
    for i in range(len(route) - 1):
        wait = extra_waits[i]
        if wait > 0:
            if t_col <= t + wait:
                return i
            t += wait

        leg_dt = dist(route[i], route[i + 1]) / speed
        if t_col <= t + leg_dt:
            return i
        t += leg_dt

        if i < len(route) - 2:
            if t_col <= t + sample_time:
                return i + 1
            t += sample_time

    return len(route) - 2


# ===============================
# Resolve collisions
# ===============================

def add_delays_to_avoid_collisions(routes, speed, d_safe=2.0,
                                   step=2.0, max_iter=200):
    trajectories = []
    all_waits = []

    for i, route_i in enumerate(routes):
        extra_waits = [0.0] * len(route_i)

        for _ in range(max_iter):
            traj_i = build_timed_trajectory(route_i, speed,
                                            extra_waits=extra_waits)
            earliest_t = None
            for j in range(i):
                ct = first_collision_time(traj_i, trajectories[j], d_safe)
                if ct is not None and (earliest_t is None or ct < earliest_t):
                    earliest_t = ct

            if earliest_t is None:
                break

            wp = waypoint_before_collision(route_i, speed, extra_waits,
                                           earliest_t)
            extra_waits[wp] += step
        else:
            print(
                f"Warning: route {i} still collides after {max_iter} iterations")

        traj_i = build_timed_trajectory(
            route_i, speed, extra_waits=extra_waits)
        trajectories.append(traj_i)
        all_waits.append(extra_waits)
        print(f"Route {i}: waits = {[round(w, 1) for w in extra_waits]}")

    return trajectories, all_waits


# ===============================
# Animation
# ===============================

def animate_trajectories(trajectories, routes, dt=0.5, trail_length=30,
                         speed_multiplier=8,
                         poly=None, tessellation=None,
                         map_coords=None, solution=None,
                         coord_order="latlon"):

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

    ax.set_title("Drone Trajectories (Top View)")
    if coord_order in ("lonlat", "latlon"):
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect(_lat_aspect(all_y), adjustable="box")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.5f}°"))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.5f}°"))
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    if coord_order in ("lonlat", "latlon"):
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)

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
        ax.scatter(map_plot[0][0], map_plot[0][1],
                   c='red', s=250, marker='*', zorder=3)

    if solution is not None and map_coords is not None:
        route_colors = ['blue', 'green', 'purple', 'orange']
        for r_idx, route in enumerate(solution.routes()):
            route_indices = [0] + list(route) + [0]
            xs = [map_plot[i][0] for i in route_indices]
            ys = [map_plot[i][1] for i in route_indices]
            ax.plot(xs, ys, color=route_colors[r_idx % len(route_colors)],
                    linewidth=1.5, alpha=0.4, zorder=2)
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
    plt.show()


if __name__ == "__main__":
    routes = [
        [(42.2915, -71.2633), (42.2908, -71.2620), (42.2920, -71.2615)],
        [(42.2910, -71.2628), (42.2918, -71.2625), (42.2912, -71.2618)],
        [(42.2922, -71.2630), (42.2913, -71.2617), (42.2905, -71.2622)],
    ]

    speed = 5.0
    trajectories, all_waits = add_delays_to_avoid_collisions(
        routes, speed, d_safe=3.0)
    animate_trajectories(trajectories, routes, speed_multiplier=8)
