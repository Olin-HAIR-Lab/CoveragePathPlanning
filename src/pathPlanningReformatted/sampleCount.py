import math
import copy
import os
import yaml
import pickle
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer

def polygon_area_m2(poly_latlon):
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32619", always_xy=True)

    def _swap_xy(x, y, z=None):
        return transformer.transform(y, x)

    poly_utm = transform(_swap_xy, poly_latlon)
    return poly_utm.area


def compute_sample_count(map, sample_time, speed, mission_time, num_agents=1):
    area = polygon_area_m2(map)
    max_possible = int((mission_time * num_agents) / sample_time)
    best_N = 1
    for N in range(1, max_possible + 1):
        if N % num_agents != 0:
            continue
        N_per_agent = N / num_agents
        total_time = N_per_agent * sample_time + \
            math.sqrt(area * N_per_agent) / speed
        if total_time <= mission_time:
            best_N = N
        else:
            break
    return best_N