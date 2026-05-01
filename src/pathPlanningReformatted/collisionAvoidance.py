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