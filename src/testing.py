from shapely.geometry import Polygon, Point, MultiPoint
import matplotlib.pyplot as plt
import numpy as np
import csv
from grid_based_path_grant_version import planning_animation, do_animation
from polygon_splitting import split_polygon_equal_area, plot_polygon_split

replace_path = False

poly = Polygon([
    (-71.2633943948142, 42.29151053590485),
    (-71.2628929214584, 42.290784490885386),
    (-71.26149845009414, 42.291353821977765),
    (-71.26200796847752, 42.29205010492521),
    (-71.2633943948142, 42.29151053590485)])

parts, centroids = split_polygon_equal_area(poly, k=3)

plot_polygon_split(poly, parts, centroids)

i = 0
paths = []

for part in parts:
    x, y = part.exterior.xy

    resolution = 0.000075
    px, py = planning_animation(x, y, resolution)
    paths.append((px, py))


if (replace_path):
    open('paths.csv', 'w').close()
    max_length = max(len(px) for px, py in paths)
    with open('paths.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(max_length):
            row = []
            for px, py in paths:
                if i < len(px):
                    row.extend([px[i], py[i]])
                else:
                    row.extend(['', ''])
            writer.writerow(row)

if do_animation:
    plt.show()
print("done!!")
