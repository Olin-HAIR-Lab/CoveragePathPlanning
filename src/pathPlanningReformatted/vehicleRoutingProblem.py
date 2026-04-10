import numpy as np
import pyvrp
import pyvrp.stop
from shapely.geometry import Point


def build_vrp_model(coords, time_windows, travel_duration_matrix, 
                    max_vehicles, mode, service_time=180, capacity=None, 
                    fixed_cost=0):
    
    m = pyvrp.Model()

    m.add_depot(
        x=coords[0][0],
        y=coords[0][1],
        name="Depot1",
    )
    m.add_depot(
        x=coords[1][0],
        y=coords[1][1],
        name="Depot2",
    )
    m.add_depot(
        x=coords[2][0],
        y=coords[2][1],
        name="Depot3",
    )

    num_depots = len(m._depots)

    if mode == "unlimited":

        m.add_vehicle_type(
            max_vehicles,
            unit_distance_cost=0,
            unit_duration_cost=1,
        )
        for idx in range(num_depots, len(coords)):
            m.add_client(
                x=coords[idx][0],
                y=coords[idx][1],
                tw_early=time_windows[idx][0],
                tw_late=time_windows[idx][1],
                service_duration=service_time,
                name=f"Client {idx}",
            )

    elif mode == "balanced":
        for idx in range(max_vehicles):
            m.add_vehicle_type(
                1,
                capacity=capacity,
                unit_distance_cost=0,
                unit_duration_cost=1,
                fixed_cost=fixed_cost,
                start_depot=m._depots[idx],
                end_depot=m._depots[idx]
            )
        for idx in range(num_depots, len(coords)):
            m.add_client(
                x=coords[idx][0],
                y=coords[idx][1],
                service_duration=service_time,
                name=f"Client {idx}",
                delivery=1
            )

    for i, frm in enumerate(m.locations):
        for j, to in enumerate(m.locations):
            duration = travel_duration_matrix[i][j]
            m.add_edge(frm, to, distance=duration, duration=duration)

    return m


def solve_vrp_unlimited(coords, time_windows, travel_duration_matrix):
    """
    Unlimited vehicles, minimize total time (your current behavior)
    """
    m = build_vrp_model(
        coords,
        time_windows,
        travel_duration_matrix,
        mode="unlimited",
        max_vehicles=100
    )

    res = m.solve(stop=pyvrp.stop.MaxRuntime(1))
    return res.best


def solve_vrp_balanced(coords, time_windows, travel_duration_matrix, num_vehicles):
    """
    Evenly distribute points across fixed number of drones
    """
    num_clients = len(coords) - 3
    capacity = int(np.ceil(num_clients / num_vehicles))

    m = build_vrp_model(
        coords,
        time_windows,
        travel_duration_matrix,
        mode="balanced",
        max_vehicles=num_vehicles,
        capacity=capacity,
        fixed_cost=500
    )

    res = m.solve(stop=pyvrp.stop.MaxRuntime(1))
    return res.best


def extract_paths(solution, coords):
    routes = solution.routes()
    paths = [None] * len(routes)

    for i, route in enumerate(routes):
        paths[i] = [
            Point(coords[client][0], coords[client][1], 3.0)
            for client in route
        ]

    return paths, routes