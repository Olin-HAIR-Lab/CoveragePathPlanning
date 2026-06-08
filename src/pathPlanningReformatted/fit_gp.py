import numpy as np
import geopandas as gpd
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.affinity import translate
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

    #print(f"random points: {points}")
    return points

def repeat_gp(coords,points,region):
    #n_points = np.array([5, 10, 20, 50, 100])
    n_points = np.array([100])
    n_trials = 50
    length_scale_avg = np.zeros((n_points.size,2))
    nrmse_avg = np.zeros(n_points.size)

    for i in range(n_points.size):
        length_scales = np.zeros((n_trials,2))
        nrmses = np.zeros(n_trials)
        for j in range(n_trials):
            _, _, ls, err = fit_gp(coords,points,region,n_synth=n_points[i],gui=False)
            length_scales[j,:] = ls
            nrmses[j] = err
        length_scale_avg[i,:] = np.mean(length_scales,axis=0)
        nrmse_avg[i] = np.mean(nrmses)
    
    print(f"Length scales: {length_scale_avg}")
    print(f"NRMSE: {nrmse_avg}")
    sys.exit()
            
def get_nmrse(gp,points):
    ground_truth_pos = np.column_stack([points.geometry.x, points.geometry.y])
    predicted_values = gp.predict(ground_truth_pos)
    actual_values = points['Moisture'].to_numpy()
    err = predicted_values - actual_values
    rmse = np.linalg.norm(err) / np.sqrt(err.size)
    nrmse = rmse / np.ptp(actual_values)
    #print(f"NRMSE: {nrmse}")
    return nrmse

def fit_gp(coords_input,points_input,region_input,n_synth=0,gui=True):
    """
    Everything is in units of meters, X (east) vs Y (north) --- we don't need to change CRS
    """
    n_synth = 100
    # Translate everything so that home is 0,0
    home = np.array(coords_input).squeeze()[0]
    # print(f"Home pos: {home}")
    points = points_input.copy()
    points["geometry"] = gpd.points_from_xy(
        points.geometry.x - home[0],
        points.geometry.y - home[1],
        crs=points.crs
    )
    coords = [[pos - home for pos in route] for route in coords_input]
    region = translate(
        region_input,
        xoff=-home[0],
        yoff=-home[1]
    )

    if n_synth > 0:
        # Generate points to sample at random
        synth_points = random_points_in_polygon(region, n_synth, seed=None)
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
    _, close_idx = tree.query(waypts, k=1)
    # for i in range(len(waypts)):
    #     print(f"Closest to {waypts[i,:]} is {ground_truth_pos[close_idx[i],:]} with dist {dist[i]}")
    sampled_values = ground_truth_value[close_idx]

    kernel = 1 * RBF(length_scale=[10.0,10.0], length_scale_bounds=[(0.1, 300.0), (0.1, 300.0)]) + 1 * WhiteKernel()
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True)
    gaussian_process.fit(waypts, sampled_values)
    # print(f"Kernel: {gaussian_process.kernel_}")
    print(gaussian_process.kernel_.get_params())
    length_scale = gaussian_process.kernel_.get_params()['k1__k2__length_scale']
    #print(f"Length scale: {length_scale} meters")

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
    nrmse = get_nmrse(gp=gaussian_process,points=points)

    if gui:
        fig, ax = plt.subplots(2,1)
        hi_val = np.quantile(ground_truth_value,0.98)
        lo_val = np.quantile(ground_truth_value,0.02)

        ax[0].scatter(query_xy[:,0],query_xy[:,1],c=mean_pred,vmin=lo_val,vmax=hi_val)
        ax[0].scatter(waypts[:,0],waypts[:,1],c='red',s=10)
        ax[0].set_title("GP mean prediction")
        ax[1].scatter(ground_truth_pos[:,0],ground_truth_pos[:,1],c=ground_truth_value,vmin=lo_val,vmax=hi_val)
        ax[1].scatter(waypts[:,0],waypts[:,1],c='red',s=10)
        ax[1].set_title("Actual moisture data")

        fig.suptitle(f"Predicted mean from samples vs ground truth\nNRMSE: {nrmse:.3f}")
        plt.tight_layout()
        plt.show()

    return mean_pred, std_pred, length_scale, nrmse