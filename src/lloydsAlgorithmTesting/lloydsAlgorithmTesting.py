

from shapely.prepared import prep
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from outline import image_to_polygon_points
import os
import imageio.v2 as imageio    # ← note the “.v2” here
import matplotlib.pyplot as plt  # For the gifs
import numpy as np
from numpy import linalg as LA  # For the varinici algorithm
import matplotlib
matplotlib.use('Agg')


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
        # build polygon from hull vertices and clip to domain
        cell_poly = Polygon(points[hull.vertices])
        if domain_poly is not None:
            cell_poly = cell_poly.intersection(domain_poly)
        if cell_poly.is_empty:
            continue
        if isinstance(cell_poly, Polygon):
            regions = [cell_poly]
        else:
            # MultiPolygon -> iterate over component polygons
            regions = list(cell_poly.geoms)
        for region in regions:
            x_reg, y_reg = region.exterior.xy
            plt.fill(x_reg, y_reg, c='C' + str(k), alpha=a)
            # draw edges explicitly from exterior coords
            coords = list(region.exterior.coords)
            for i in range(len(coords)-1):
                plt.plot([coords[i][0], coords[i+1][0]],
                         [coords[i][1], coords[i+1][1]],
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
                   plot=False, history=False, num_images=5):
    """Run Lloyd's algorithm on ``N`` sites inside ``domain_poly``.

    ``domain_poly`` must be a closed ``shapely.geometry.Polygon``; the grid
    resolution is controlled by ``partition``.  The optional ``seed`` fixes the
    pseudo‑random initialisation.  ``plot``/``history`` behave as before.  The
    function returns the same three-tuples as the original version.
    """
    if seed is not None:
        np.random.seed(seed)

    # remove old images if plotting so we start clean
    if plot:
        imgdir = os.path.join(os.getcwd(), "imagen")
        if os.path.isdir(imgdir):
            for fname in os.listdir(imgdir):
                path = os.path.join(imgdir, fname)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

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

    # determine which iterations we will save (evenly spaced across 0..Iterations-1)
    if plot and Iterations > 0:
        indices = np.linspace(
            0, Iterations-1, num=min(num_images, Iterations), dtype=int)
        plot_iters = set(indices.tolist())
    else:
        plot_iters = set()

    for itera in range(Iterations):
        tessellation = varinoci(new_dots, domain_poly, partition)
        new_dots = new_centroids(tessellation)
        if plot and itera in plot_iters:
            imagen_names = plot_tessell(N, tessellation, new_dots, itera,
                                        imagen_names, domain_poly=domain_poly)
        if history:
            history_tessell.append(tessellation)
            history_dots.append(new_dots)
        if itera % 10 == 0:
            print('iteration number ' + str(itera) + ' executed.')
    return imagen_names, history_tessell, history_dots


def makeGif(imagen_names):
    new_imagen_names = []
    Frames = 1  # duplicate each frame if you want a slower animation
    for name in imagen_names:
        for frame in range(Frames):
            new_imagen_names.append(name)

    gif_name = ('Lloyd_algorithm_P'+str(partition)+'_S'+str(seed) +
                '_I'+str(iterations)+'_N'+str(N_dots))

    cwd = os.getcwd()

    newpath_gif = os.path.join(cwd, 'gifs')
    if not os.path.exists(newpath_gif):
        os.makedirs(newpath_gif)

    gif_path = os.path.join(newpath_gif, gif_name + '.gif')
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in new_imagen_names:
            image = imageio.imread(filename)
            writer.append_data(image)


############### SETTINGS #################


# image_path = "usa.png"
# poly, points = image_to_polygon_points(
#     image_path,
#     num_points=200,         # Adjust for more/fewer points
#     scale_to_range=(-16, 8)  # Adjust coordinate range
# )

poly = Polygon([
    (-71.2633943948142, 42.29151053590485),
    (-71.2628929214584, 42.290784490885386),
    (-71.26149845009414, 42.291353821977765),
    (-71.26200796847752, 42.29205010492521),
    (-71.2633943948142, 42.29151053590485)])


N_dots = 5        # number of generating points/sites
iterations = 5    # Lloyd iterations to perform
plot = True       # produce image files
history = False   # return history of tessellations
seed = 6           # random seed (set to None for non-deterministic behaviour)
partition = 350   # grid resolution for approximate voronoi


########### MAIN ALGORITHM ##################
imagen_names, history_tessell, history_dots = \
    Lloyd_algoritm(iterations, N_dots, poly, partition,
                   seed=seed, plot=plot, history=history, num_images=iterations)

if plot is True:
    makeGif(imagen_names)
