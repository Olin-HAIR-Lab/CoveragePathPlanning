import numpy as np
import geopandas as gpd
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from shapely.geometry import Point
import numpy as np

def random_points_in_polygon(poly, n, seed=None):
    rng = np.random.default_rng(seed)

    minx, miny, maxx, maxy = poly.bounds
    points = []

    while len(points) < n:
        p = Point(
            rng.uniform(minx, maxx),
            rng.uniform(miny, maxy)
        )

        if poly.contains(p):
            points.append(p)

    return points

def get_nmrse(gp,points):
    ground_truth_pos = np.column_stack([points.geometry.x, points.geometry.y])
    predicted_values = gp.predict(ground_truth_pos)
    actual_values = points['Moisture'].to_numpy()
    err = predicted_values - actual_values
    rmse = np.linalg.norm(err) / np.sqrt(err.size)
    nrmse = rmse / np.ptp(actual_values)
    print(f"NRMSE: {nrmse}")
    return nrmse

def fit_gp(coords,points,region,gui=True):
    """
    Everything is in units of meters, X (east) vs Y (north) --- we don't need to change CRS
    """
    n_synth = 10

    if n_synth > 0:
        # Generate points to sample at random
        synth_points = random_points_in_polygon(region, n_synth, seed=42)
        synthetic = gpd.GeoDataFrame(
            {"sample_id": range(n_synth)},
            geometry=synth_points,
            crs=points.crs
        )
        waypts = np.column_stack([
            synthetic.geometry.x,
            synthetic.geometry.y
        ])
    else:
        # Use our actual chosen points
        waypts = np.array(coords).squeeze()[1:-1] # ignore start and end (these are the depot position)
        print(f"waypts: {waypts}")

    ground_truth_pos = np.column_stack([points.geometry.x, points.geometry.y])
    ground_truth_value = points['Moisture'].to_numpy()

    tree = cKDTree(ground_truth_pos)
    dist, close_idx = tree.query(waypts, k=1)
    for i in range(len(waypts)):
        print(f"Closest to {waypts[i,:]} is {ground_truth_pos[close_idx[i],:]} with dist {dist[i]}")
    sampled_values = ground_truth_value[close_idx]

    kernel = 1 * RBF(length_scale=10.0, length_scale_bounds=(1.0, 200.0))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True)
    gaussian_process.fit(waypts, sampled_values)
    print(f"Kernel: {gaussian_process.kernel_}")
    length_scale = gaussian_process.kernel_.get_params()['k2__length_scale']
    print(f"Length scale: {length_scale} meters")

    minx, miny, maxx, maxy = region.bounds
    nx = 100
    ny = 100
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    xx, yy = np.meshgrid(xs, ys)
    test_points = gpd.GeoSeries(
        gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=points.crs,
    )

    inside = test_points.within(region)  
    query_xy = np.column_stack([test_points.x[inside], test_points.y[inside]])
    mean_pred, std_pred = gaussian_process.predict(query_xy, return_std=True)

    fig, ax = plt.subplots(2,1)
    hi_val = np.quantile(ground_truth_value,0.98)
    lo_val = np.quantile(ground_truth_value,0.02)

    ax[0].scatter(query_xy[:,0],query_xy[:,1],c=mean_pred,vmin=lo_val,vmax=hi_val)
    ax[0].scatter(waypts[:,0],waypts[:,1],c='red',s=10)
    ax[0].set_title("GP mean prediction")
    ax[1].scatter(ground_truth_pos[:,0],ground_truth_pos[:,1],c=ground_truth_value,vmin=lo_val,vmax=hi_val)
    ax[1].scatter(waypts[:,0],waypts[:,1],c='red',s=10)
    ax[1].set_title("Actual moisture data")

    nrmse = get_nmrse(gp=gaussian_process,points=points)
    fig.suptitle(f"Predicted mean from samples vs ground truth\nNRMSE: {nrmse:.3f}")
    plt.tight_layout()
    plt.show()

    sys.exit()
    return mean_pred, std_pred