import math
import copy
import os
import yaml
import pickle
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer

SQ_METERS_PER_ACRE = 4046.86

def polygon_area_m2(poly_latlon, ll_crs, utm_crs):
    # transformer = Transformer.from_crs(
    #     "EPSG:4326", "EPSG:32619", always_xy=True)
    transformer = Transformer.from_crs(
        ll_crs, utm_crs, always_xy=True
    )

    def _swap_xy(x, y, z=None):
        return transformer.transform(y, x)

    poly_utm = transform(_swap_xy, poly_latlon)
    return poly_utm.area


def compute_sample_count(map, sample_time, speed, mission_time, num_agents=1, ll_crs="4326", utm_crs="32619"):
    area = polygon_area_m2(map,ll_crs,utm_crs)
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
    print(f"Resolution: {(area/SQ_METERS_PER_ACRE)/best_N:.2f} acres per sample")
    return best_N