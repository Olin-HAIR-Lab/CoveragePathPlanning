import math
import copy

# ===============================
# Helpers
# ===============================

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def interpolate(p1, p2, t_ratio):
    """Linear interpolation between two points"""
    return (
        p1[0] + (p2[0] - p1[0]) * t_ratio,
        p1[1] + (p2[1] - p1[1]) * t_ratio
    )


# ===============================
# Build time-parameterized trajectory
# ===============================

def build_timed_trajectory(route, speed, start_time=0.0):
    """
    Converts a route into a list of segments with time.
    
    Returns list of:
    [(p_start, p_end, t_start, t_end), ...]
    """
    traj = []
    t = start_time

    for i in range(len(route) - 1):
        p1, p2 = route[i], route[i + 1]
        d = dist(p1, p2)
        dt = d / speed

        traj.append((p1, p2, t, t + dt))
        t += dt

    return traj


# ===============================
# Sample trajectory position at time t
# ===============================

def position_at(traj, t):
    for (p1, p2, t1, t2) in traj:
        if t1 <= t <= t2:
            ratio = (t - t1) / (t2 - t1 + 1e-9)
            return interpolate(p1, p2, ratio)
    return traj[-1][1]  # final position


# ===============================
# Collision check
# ===============================

def trajectories_collide(traj1, traj2, d_safe=2.0, dt=0.5):
    """
    Check if two trajectories collide.
    """
    t_start = min(traj1[0][2], traj2[0][2])
    t_end = max(traj1[-1][3], traj2[-1][3])

    t = t_start
    while t <= t_end:
        p1 = position_at(traj1, t)
        p2 = position_at(traj2, t)

        if dist(p1, p2) < d_safe:
            return True

        t += dt

    return False


# ===============================
# Resolve collisions with delays
# ===============================

def add_delays_to_avoid_collisions(routes, speed, d_safe=2.0):
    """
    routes: list of routes (each is list of (x,y))
    
    Returns:
        delayed trajectories (list of trajs)
    """
    trajectories = []
    delays = [0.0 for _ in routes]

    for i, route_i in enumerate(routes):

        delay = 0.0

        while True:
            traj_i = build_timed_trajectory(route_i, speed, start_time=delay)

            collision_found = False

            for j in range(i):
                if trajectories_collide(traj_i, trajectories[j], d_safe):
                    collision_found = True
                    break

            if collision_found:
                delay += 2.0  # ⬅️ tune this (seconds)
            else:
                break

        delays[i] = delay
        trajectories.append(traj_i)

    return trajectories, delays


# ===============================
# Example usage
# ===============================

if __name__ == "__main__":

    # Example: 2 crossing routes (collision scenario)
    routes = [
        [(0, 0), (10, 10)],
        [(0, 10), (10, 0)]
    ]

    speed = 2.0  # m/s

    trajectories, delays = add_delays_to_avoid_collisions(routes, speed)

    print("Delays applied:", delays)
