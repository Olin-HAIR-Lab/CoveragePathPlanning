import numpy as np
from numpy import linalg as LA
from shapely.prepared import prep
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull


def varinoci(dots, domain_poly, partition):
    tessellation = {}
    minx, miny, maxx, maxy = domain_poly.bounds
    X = np.linspace(minx, maxx, partition)
    Y = np.linspace(miny, maxy, partition)
    prepared = prep(domain_poly)

    for xi in X:
        for yj in Y:
            p = np.array([xi, yj])
            if not prepared.contains(Point(p)):
                continue

            minimo = float("inf")
            k_star = None

            for k, dot in enumerate(dots):
                value = LA.norm(p - dot)
                if value < minimo:
                    minimo = value
                    k_star = k

            if k_star is not None:
                tessellation.setdefault(k_star, []).append(p)

    return tessellation


def new_centroids(tessellation, old_dots):
    new_dots = []

    for i in range(len(old_dots)):
        if i in tessellation and len(tessellation[i]) > 0:
            xi = np.array(tessellation[i])
            new_dots.append(np.mean(xi, axis=0))
        else:
            new_dots.append(old_dots[i])

    return np.array(new_dots)


def Lloyd_algoritm(Iterations, N, domain_poly, partition, seed=None):
    if seed is not None:
        np.random.seed(seed)

    minx, miny, maxx, maxy = domain_poly.bounds
    new_dots = []

    while len(new_dots) < N:
        cand = np.random.uniform([minx, miny], [maxx, maxy])
        if domain_poly.contains(Point(cand)):
            new_dots.append(cand)

    new_dots = np.array(new_dots)

    history_tessell = []
    history_dots = [new_dots]

    for itera in range(Iterations):
        tessellation = varinoci(new_dots, domain_poly, partition)
        new_dots = new_centroids(tessellation, new_dots)

        history_tessell.append(tessellation)
        history_dots.append(new_dots)

        print(f"Iteration {itera} complete.")

    return history_tessell, history_dots