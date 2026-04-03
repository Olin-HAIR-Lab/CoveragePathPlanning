import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation


# ===============================
# Helpers
# ===============================

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def interpolate(p1, p2, t_ratio):
    return (
        p1[0] + (p2[0] - p1[0]) * t_ratio,
        p1[1] + (p2[1] - p1[1]) * t_ratio,
    )


# ===============================
# Build time-parameterized trajectory
# ===============================

def build_timed_trajectory(route, speed, sample_time=5, extra_waits=None):
    """
    extra_waits[k] = extra seconds to hold at route[k] BEFORE departing.
    Segment order per leg i:
      1. Hold at route[i]            (extra_waits[i], omitted if 0)
      2. Travel route[i]→route[i+1]  (dist/speed)
      3. Sample pause at route[i+1]  (sample_time, omitted on last leg)
    """
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
    """Returns (position, segment_type) at time t."""
    for seg in traj:
        p1, p2, t1, t2, kind = seg
        if t1 <= t <= t2:
            ratio = (t - t1) / (t2 - t1 + 1e-9)
            pos = interpolate(p1, p2, ratio)
            return pos, kind
    return traj[-1][1], "done"

def position_at(traj, t):
    return state_at(traj, t)[0]


# ===============================
# Collision detection
# ===============================

def first_collision_time(traj1, traj2, d_safe=2.0, dt=0.5):
    t_start = min(traj1[0][2], traj2[0][2])
    t_end   = max(traj1[-1][3], traj2[-1][3])
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
# Resolve collisions with local waypoint delays
# ===============================

def add_delays_to_avoid_collisions(routes, speed, d_safe=2.0,
                                   step=2.0, max_iter=200):
    trajectories = []
    all_waits    = []

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
            print(f"Warning: route {i} still collides after {max_iter} iterations")

        traj_i = build_timed_trajectory(route_i, speed, extra_waits=extra_waits)
        trajectories.append(traj_i)
        all_waits.append(extra_waits)
        print(f"Route {i}: waits = {[round(w, 1) for w in extra_waits]}")

    return trajectories, all_waits


# ===============================
# Animation
# ===============================

def animate_trajectories(trajectories, dt=0.5, trail_length=30,
                          speed_multiplier=8):
    # End when the last drone completes its final TRAVEL move, not any trailing holds
    t_start = 0.0
    t_end   = max(
        seg[3] for traj in trajectories
        for seg in traj if seg[4] == "travel"
    )
    total_duration = t_end - t_start

    interval_ms = max(16, int(1000 * dt / speed_multiplier))
    times = np.arange(t_start, t_end + dt, dt)

    print(f"Sim duration: {total_duration:.1f}s | Frames: {len(times)} | "
          f"Interval: {interval_ms}ms | {speed_multiplier}x speed")

    all_states = [
        [state_at(traj, t) for t in times]
        for traj in trajectories
    ]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 6),
        gridspec_kw={"width_ratios": [14, 1]}
    )
    fig.subplots_adjust(left=0.07, right=0.97, wspace=0.12)

    ax.set_title("Drone Trajectories (Top View)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    all_x = [s[0][0] for states in all_states for s in states]
    all_y = [s[0][1] for states in all_states for s in states]
    ax.set_xlim(min(all_x) - 5, max(all_x) + 5)
    ax.set_ylim(min(all_y) - 5, max(all_y) + 5)
    ax.set_aspect("equal")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    points, trails, rings = [], [], []
    for k in range(len(trajectories)):
        c = colors[k % len(colors)]
        pt, = ax.plot([], [], "o", color=c, ms=8, zorder=5)
        tr, = ax.plot([], [], "-", color=c, lw=1.5, alpha=0.6)
        # Hollow ring shown while drone is holding
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
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    ax_bar.add_patch(mpatches.Rectangle((0.2, 0), 0.6, 1.0,
                                         fc="#e0e0e0", ec="#aaaaaa", lw=0.8))
    bar_fill = mpatches.Rectangle((0.2, 0), 0.6, 0.0, fc="#2196F3", ec="none")
    ax_bar.add_patch(bar_fill)

    pct_text = ax_bar.text(0.5, -0.04, "0%", ha="center", va="top", fontsize=9)
    ax_bar.text(0.5,  1.03, f"{t_end:.0f}s",   ha="center", va="bottom",
                fontsize=7, color="#666")
    ax_bar.text(0.5, -0.12, f"{t_start:.0f}s", ha="center", va="bottom",
                fontsize=7, color="#666")

    def update(frame):
        for i, states in enumerate(all_states):
            pos, kind = states[frame]
            x, y = pos
            points[i].set_data([x], [y])

            s = max(0, frame - trail_length)
            trails[i].set_data([states[k][0][0] for k in range(s, frame)],
                               [states[k][0][1] for k in range(s, frame)])

            # Show ring when holding, hide otherwise
            if kind == "hold":
                rings[i].set_data([x], [y])
            else:
                rings[i].set_data([], [])

        progress = (times[frame] - t_start) / total_duration
        bar_fill.set_height(min(progress, 1.0))
        pct_text.set_text(f"{progress * 100:.0f}%")

    ani = FuncAnimation(fig, update, frames=len(times),
                        interval=interval_ms, blit=False, repeat=False)
    plt.show()


# ===============================
# Example usage
# ===============================

if __name__ == "__main__":
    # NOTE: routes 2 and 3 previously shared waypoints (30,10) and (70,10)
    # which forced drone 3 to accumulate hundreds of seconds of hold time.
    # Staggered here so no two routes share an exact waypoint.
    routes = [
        [(0,  0),  (20, 20), (40,  0), (60, 20)],
        [(0,  20), (20,  0), (40, 20), (60,  0)],
        [(10, -5), (30, 15), (50, -5), (70, 15)],
        [(10, 25), (30,  5), (50, 25), (70,  5)],
    ]

    speed = 2.0

    trajectories, all_waits = add_delays_to_avoid_collisions(routes, speed)
    animate_trajectories(trajectories, speed_multiplier=8)