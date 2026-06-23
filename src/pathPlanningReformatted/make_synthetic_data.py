import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import geopandas as gpd
from shapely.geometry import Point
from shapely import concave_hull

SIDE_LENGTHS = [20, 40, 60, 80, 100]
NUM_GAUSSIANS = 5
SPACE_BETWEEN_POINTS = 1

def sum_of_gaussians(side_length,symmetric=False):
    sum_arr = np.zeros((int(side_length//SPACE_BETWEEN_POINTS), int(side_length//SPACE_BETWEEN_POINTS)))
    x = np.linspace(0, side_length, sum_arr.shape[0])
    y = np.linspace(0, side_length, sum_arr.shape[0])
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    means = np.random.uniform(low=0.0,high=side_length,size=(NUM_GAUSSIANS,2))
    sigma_x = np.random.uniform(
        low=0.025 * side_length,
        high=0.2 * side_length,
        size=NUM_GAUSSIANS
    )

    sigma_y = np.random.uniform(
        low=0.01 * side_length,
        high=0.1 * side_length,
        size=NUM_GAUSSIANS
    )

    if symmetric:
        covs = [np.array([[sigma_x[j]**2, 0.0],[0.0, sigma_x[j]**2]]) for j in range(NUM_GAUSSIANS)]
    else:
        rots = np.random.uniform(low=0,high=2*np.pi,size=NUM_GAUSSIANS)
        covs_initial = [np.array([[sigma_x[j]**2, 0.0],[0.0, sigma_y[j]**2]]) for j in range(NUM_GAUSSIANS)]
        covs = [np.array([[np.cos(rot), -np.sin(rot)],[np.sin(rot), np.cos(rot)]]) @ \
            cov @ np.array([[np.cos(rot), np.sin(rot)],[-np.sin(rot), np.cos(rot)]]) \
            for rot,cov in zip(rots,covs_initial)
        ]
        
    for i in range(NUM_GAUSSIANS):
        #print(f"Means: {means}")
        #print(f"Covs: {covs}")
        gauss = multivariate_normal(means[i,:], covs[i])
        Z = gauss.pdf(pos)
        #print(f"Sum arr shape: {sum_arr.shape}, Z shape {Z.shape}")
        sum_arr += Z * np.sqrt(np.linalg.det(covs[i]))
    
    return sum_arr

def make_gradient(side_length):
    sum_arr = np.zeros((int(side_length//SPACE_BETWEEN_POINTS), int(side_length//SPACE_BETWEEN_POINTS)))
    x = np.linspace(0, side_length, sum_arr.shape[0])
    y = np.linspace(0, side_length, sum_arr.shape[0])
    X, Y = np.meshgrid(x, y)

    return 0.3 * X / side_length



def make_map(side_length,num_gaussians,gradient,symmetric):
    arr = np.zeros((int(side_length//SPACE_BETWEEN_POINTS), int(side_length//SPACE_BETWEEN_POINTS)))

    if num_gaussians > 0:
        arr += sum_of_gaussians(side_length=side_length,symmetric=symmetric)

    if gradient:
        arr += make_gradient(side_length=side_length)

    return arr

def main():
    for ii in range(60):
        print(f"Starting iteration {ii}")
        arr = make_map(side_length=40,num_gaussians=NUM_GAUSSIANS,gradient=True,symmetric=True)

        fig, ax = plt.subplots()
        ax.imshow(arr)
        #plt.show()
        plt.savefig(f"../scripts/synth_gradient_3gauss/data{ii}.png")
        plt.close(fig)

        rows, cols = np.indices(arr.shape)
        points_gdf = gpd.GeoDataFrame(
            {
                "Moisture": arr.ravel()
            },
            geometry=[
                Point(x/1e4, y/1e4)
                for x, y in zip(cols.ravel(), rows.ravel())
            ],
            crs="EPSG:4326"  # replace with your CRS
        )
        points_union = points_gdf.geometry.union_all()
        boundary = concave_hull(points_union, ratio=0.2)
        print(f"geom type {boundary.geom_type}")
        region = gpd.GeoDataFrame(
            geometry=[
                boundary
            ],
            crs="EPSG:4326"
        )

        # save to file
        out_path = f"../scripts/synth_gradient_3gauss/data{ii}.gpkg"
        region.to_file(out_path, layer="polygon", driver="GPKG")
        points_gdf.to_file(out_path, layer="points", driver="GPKG")

if __name__ == "__main__":
    main()

    