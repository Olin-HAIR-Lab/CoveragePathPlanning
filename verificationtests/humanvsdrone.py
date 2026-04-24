import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Parameters ────────────────────────────────────────────────────────────────
human_speed  = 1.4   # m/s
drone_speed  = 5.0   # m/s
human_dwell  = 8     # seconds per point (quick manual probe)
drone_dwell  = 30    # seconds per point (mechanical sampler)

# ── Axes ──────────────────────────────────────────────────────────────────────
areas  = np.linspace(100, 40000, 40)   # m²
points = np.linspace(5, 100, 40)       # number of sample points
A, N   = np.meshgrid(areas, points)

# ── Mission time model ────────────────────────────────────────────────────────
def mission_time(area, N, speed, dwell):
    spacing = np.sqrt(area / N)
    travel  = (N * spacing) / speed
    probe   = N * dwell
    return (travel + probe) / 60   # convert to minutes

human_time = mission_time(A, N, human_speed, human_dwell)
drone_time = mission_time(A, N, drone_speed, drone_dwell)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(A, N, human_time, alpha=0.6, color='steelblue', label='Human')
ax.plot_surface(A, N, drone_time, alpha=0.6, color='coral',     label='Drone')

ax.set_xlabel('Plot area (m²)')
ax.set_ylabel('Sample points')
ax.set_zlabel('Mission time (min)')
ax.set_title('Human vs drone sampling efficiency')

human_patch = plt.matplotlib.patches.Patch(color='steelblue', label='Human')
drone_patch = plt.matplotlib.patches.Patch(color='coral',     label='Drone')
ax.legend(handles=[human_patch, drone_patch])

plt.tight_layout()

breakeven = human_time - drone_time  # positive = drone wins
ax.contour(A, N, breakeven, levels=[0], colors='black', linewidths=2)
plt.show()
