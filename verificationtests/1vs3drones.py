import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────
drone_speed   = 5.0   # m/s
drone_dwell   = 30    # seconds per point
area          = 6000  # m² — Olin soccer field
mission_time  = 10    # minutes per charge cycle (flight endurance)
recharge_time = 45    # minutes (fixed)
max_points    = 150   # total sample points to plot across

# ── Mission time model ────────────────────────────────────────────────────────
def points_per_cycle(num_agents, speed, dwell, area, flight_endurance_min):
    """How many total points can the fleet cover in one charge cycle."""
    flight_endurance_s = flight_endurance_min * 60
    # binary search for max N per agent that fits in one cycle
    best = 0
    for n in range(1, 500):
        travel = np.sqrt(area * n) / speed
        probe  = n * dwell
        if travel + probe <= flight_endurance_s:
            best = n
        else:
            break
    return best * num_agents

def total_time_steps(total_points, num_agents, speed, dwell, area,
                     flight_endurance_min, recharge_min):
    """
    Returns (time_axis, points_completed) as step function arrays.
    Each cycle: fleet covers a batch of points, then all recharge.
    """
    capacity   = points_per_cycle(num_agents, speed, dwell, area, flight_endurance_min)
    times      = [0]
    completed  = [0]
    elapsed    = 0
    done       = 0

    while done < total_points:
        # fly a cycle
        batch    = min(capacity, total_points - done)
        elapsed += flight_endurance_min
        done    += batch
        times.append(elapsed)
        completed.append(done)

        if done < total_points:
            # recharge
            elapsed += recharge_min
            times.append(elapsed)
            completed.append(done)  # no new points during recharge

    return np.array(times), np.array(completed)

# ── Build step functions ───────────────────────────────────────────────────────
t1, p1 = total_time_steps(max_points, 1, drone_speed, drone_dwell, area,
                           mission_time, recharge_time)
t3, p3 = total_time_steps(max_points, 3, drone_speed, drone_dwell, area,
                           mission_time, recharge_time)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.step(t1, p1, where='post', color='steelblue', linewidth=2, label='1 drone')
ax.step(t3, p3, where='post', color='coral',     linewidth=2, label='3 drones')

# shade recharge periods for 1 drone
capacity1 = points_per_cycle(1, drone_speed, drone_dwell, area, mission_time)
elapsed = 0
first   = True
while elapsed < t1[-1]:
    elapsed += mission_time
    ax.axvspan(elapsed, elapsed + recharge_time,
               alpha=0.08, color='steelblue',
               label='recharge period (1 drone)' if first else None)
    elapsed += recharge_time
    first = False

ax.axhline(max_points, color='gray', linestyle='--', linewidth=1, label='target points')
ax.set_xlabel('Elapsed time (min)')
ax.set_ylabel('Sample points completed')
ax.set_title('Scalability: 1 drone vs 3 drones with recharge cycles')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
