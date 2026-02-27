

import numpy as np
from numpy import linalg as LA  # For the varinici algorithm
import matplotlib.pyplot as plt  # For the gifs
import imageio  # For the gifs
import os
# To plot the boundary of each cell
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from shapely.geometry import Point, Polygon
from shapely.prepared import prep


def varinoci(dots, domain_poly, partition):
    """Compute a very simple voronoi partition over the polygon.

    Grid points are generated over the bounding box of ``domain_poly`` and
    those falling outside the polygon are discarded.  Each remaining cell is
    assigned to the closest site in ``dots`` using Euclidean distance.
    """
    tessellation = {}
    minx, miny, maxx, maxy = domain_poly.bounds
    X = np.linspace(minx, maxx, partition)
    Y = np.linspace(miny, maxy, partition)
    prepared = prep(domain_poly)
    for xi in X:
        for yj in Y:
            p = np.array([xi, yj])
            if not prepared.contains(Point(p)):
                # skip grid points outside the polygon
                continue
            minimo = float("inf")
            actual_dot = p
            for k, dot in enumerate(dots):
                value = LA.norm(actual_dot - dot)
                if value < minimo:
                    minimo = value
                    k_star = k
            tessellation.setdefault(k_star, []).append(actual_dot)
    return tessellation


def new_centroids(tessellation):
    """Return the arithmetic mean of the points in each cell.

    ``tessellation`` is a dict mapping site index to a list of grid points
    assigned to that site.  The new centroid is just the average of those
    points; since all grid points lie within the domain the average will also
    lie inside the polygon.  The partition and domain size are irrelevant.
    """
    new_dots = []
    for i in range(len(tessellation)):
        xi = np.array(tessellation[i])
        CI = xi.shape[0]
        new_dots.append(np.sum(xi, axis=0) / CI)
    return np.array(new_dots)


def plot_tessell(N, tessellation, new_dots, itera, imagen_names, domain_poly=None):
    ################
    # To plot the tessellation with differents colors.
    ################
    cwd = os.getcwd()

    # We create a new path to save the images
    newpath = os.path.join(cwd, "imagen")
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    plt.figure(itera)
    # draw polygon border if provided
    if domain_poly is not None:
        x_poly, y_poly = domain_poly.exterior.xy
        plt.plot(x_poly, y_poly, 'k-')

    for k in range(N):
        points = np.array(tessellation[k])
        hull = ConvexHull(points)
        a = 0.5
        x = points[hull.vertices, 0]
        y = points[hull.vertices, 1]
        plt.fill(x, y, c='C' + str(k), alpha=a)

        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1],
                     'k-', linewidth=3.1)

    plt.scatter(new_dots[:, 0], new_dots[:, 1], c='k', s=20, linewidth=2)
    plt.axis('off')

    save_path = os.path.join(
        newpath, 'Lloyd_algorithm_'+str(itera)+'.png')  # We save the images

    plt.savefig(save_path)
    plt.close()
    imagen_names.append(save_path)
    return imagen_names


def Lloyd_algoritm(Iterations, N, domain_poly, partition, seed=None,
                   plot=False, history=False):
    """Run Lloyd's algorithm on ``N`` sites inside ``domain_poly``.

    ``domain_poly`` must be a closed ``shapely.geometry.Polygon``; the grid
    resolution is controlled by ``partition``.  The optional ``seed`` fixes the
    pseudo‑random initialisation.  ``plot``/``history`` behave as before.  The
    function returns the same three-tuples as the original version.
    """
    if seed is not None:
        np.random.seed(seed)

    # initialise sites randomly within polygon
    minx, miny, maxx, maxy = domain_poly.bounds
    new_dots = []
    while len(new_dots) < N:
        cand = np.random.uniform([minx, miny], [maxx, maxy])
        if domain_poly.contains(Point(cand)):
            new_dots.append(cand)
    new_dots = np.array(new_dots)

    history_tessell = []
    history_dots = [new_dots]
    imagen_names = []

    for itera in range(Iterations):
        tessellation = varinoci(new_dots, domain_poly, partition)
        new_dots = new_centroids(tessellation)
        if plot:
            imagen_names = plot_tessell(N, tessellation, new_dots, itera,
                                        imagen_names, domain_poly=domain_poly)
        if history:
            history_tessell.append(tessellation)
            history_dots.append(new_dots)
        if itera % 10 == 0:
            print('iteration number ' + str(itera) + ' executed.')
    return imagen_names, history_tessell, history_dots


############### SETTINGS #################

# example domain: any closed polygon (first and last point may coincide)
poly = Polygon([
    (-71.2633943948142, 42.29151053590485),
    (-71.2628929214584, 42.290784490885386),
    (-71.26149845009414, 42.291353821977765),
    (-71.26200796847752, 42.29205010492521),
    (-71.2633943948142, 42.29151053590485)])

N_dots = 15        # number of generating points/sites
iterations = 15    # Lloyd iterations to perform
plot = True       # produce image files
history = False   # return history of tessellations
partition = 350   # grid resolution for approximate voronoi

########### MAIN ALGORITHM ##################

imagen_names, history_tessell, history_dots = \
    Lloyd_algoritm(iterations, N_dots, poly, partition,
                   plot=plot, history=history)


if plot is True:
    new_imagen_names = []
    Frames = 1  # If you want an animation more slow, you can increce this parameter
    for itera in range(iterations):
        for frame in range(Frames):
            new_imagen_names.append(imagen_names[itera])

    gif_name = ('Lloyd_algorithm_P'+str(partition)+'_S'+str(seed) +
                '_I'+str(iterations)+'_N'+str(N_dots))

    cwd = os.getcwd()

    newpath_gif = cwd + '\\gifs'
    if not os.path.exists(newpath_gif):
        os.makedirs(newpath_gif)

    with imageio.get_writer(newpath_gif+'\\'+gif_name+'.gif', mode='I') as writer:
        for filename in new_imagen_names:
            image = imageio.imread(filename)
            writer.append_data(image)
