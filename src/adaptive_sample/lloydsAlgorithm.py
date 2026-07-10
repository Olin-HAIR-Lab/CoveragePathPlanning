import numpy as np
import matplotlib.pyplot as plt
import time
from shapely.geometry import LineString, Point
from shapely.ops import split, unary_union

def keep_component_closer_to_site(geometry, pi, pj, domain_scale):
    """
    Clip geometry to the half-plane containing points closer to pi
    than to pj.
    """
    if geometry.is_empty:
        return geometry

    pi = np.asarray(pi, dtype=float)
    pj = np.asarray(pj, dtype=float)

    delta = pj - pi
    distance = np.linalg.norm(delta)

    midpoint = 0.5 * (pi + pj)

    # Unit vector along the perpendicular bisector.
    tangent = np.array([-delta[1], delta[0]]) / distance

    # Long enough to cross the complete domain.
    half_length = 4.0 * domain_scale
    splitter = LineString([
        midpoint - half_length * tangent,
        midpoint + half_length * tangent,
    ])

    result = split(geometry, splitter)

    # If the line does not cross the current cell, the complete cell is entirely on one side
    # Figure out which side by testing a point inside the cell
    if len(result.geoms) == 1:
        representative = geometry.representative_point()
        x = np.asarray(representative.coords[0])

        if np.linalg.norm(x - pi) <= np.linalg.norm(x - pj):
            return geometry

        return geometry.difference(geometry)

    # Keep every resulting component whose interior lies on pi's side.
    retained = []

    for part in result.geoms:
        representative = np.asarray(
            part.representative_point().coords[0]
        )

        if np.linalg.norm(representative - pi) <= \
           np.linalg.norm(representative - pj):
            retained.append(part)

    if not retained:
        return geometry.difference(geometry)

    return unary_union(retained)

def bounded_voronoi_cells(points, domain_poly):
    points = np.asarray(points, dtype=float)

    minx, miny, maxx, maxy = domain_poly.bounds
    domain_scale = np.hypot(maxx - minx, maxy - miny)

    cells = []

    for i, pi in enumerate(points):
        cell = domain_poly

        for j, pj in enumerate(points):
            if i == j:
                continue

            cell = keep_component_closer_to_site(
                geometry=cell,
                pi=pi,
                pj=pj,
                domain_scale=domain_scale,
            )

            if cell.is_empty:
                break

        cells.append(cell)
    


    return cells

def lloyd_algorithm(iterations, n_pts, domain_poly, seed=None, log=False, plot=False):
    start_time = time.perf_counter()
    if seed is not None:
        np.random.seed(seed)
    
    minx, miny, maxx, maxy = domain_poly.bounds
    random_pts = []

    while len(random_pts) < n_pts:
        cand = np.random.uniform([minx, miny], [maxx, maxy])
        if domain_poly.contains(Point(cand)):
            random_pts.append(cand)

    new_points = np.array(random_pts)

    history_tessell = []
    history_dots = [new_points]
    cells = []

    for iteration in range(iterations):
        old_points = new_points.copy()

        cells = bounded_voronoi_cells(
            points=old_points,
            domain_poly=domain_poly,
        )

        if plot:
            _, ax = plt.subplots()

            x, y = domain_poly.exterior.xy
            ax.plot(x, y, 'k', linewidth=2)

            for cell in cells:
                if cell.is_empty:
                    continue

                if cell.geom_type == "Polygon":
                    x, y = cell.exterior.xy
                    ax.fill(x, y, alpha=0.3)

                elif cell.geom_type == "MultiPolygon":
                    for part in cell.geoms:
                        x, y = part.exterior.xy
                        ax.fill(x, y, alpha=0.3)

            ax.plot(old_points[:,0], old_points[:,1], 'ko')

            ax.set_aspect('equal')
            plt.show()

        updated_points = []

        for site_index, cell in enumerate(cells):
            if cell.is_empty:
                print(
                    f"Empty cell for site {site_index}; "
                    "preserving previous generator"
                )
                updated_points.append(old_points[site_index])
                continue

            centroid = cell.centroid

            if cell.covers(centroid):
                updated_points.append(centroid.coords[0])
            else:
                # This can happen for disconnected or highly concave cells.
                updated_points.append(
                    cell.representative_point().coords[0]
                )

        new_points = np.asarray(updated_points, dtype=float)
        history_dots.append(new_points.copy())

    if log:
        print(f"Finished {iterations} Lloyd iterations in {time.perf_counter() - start_time:.3f}s")
    return history_tessell, history_dots
