import math
from shapely.ops import transform
from pyproj import Transformer
# To plot the boundary of each cell

from shapely.geometry import Point, Polygon

def polygon_area_m2(poly_latlon):
    """
    Convert WGS84 polygon to UTM and return area in m².
    """

    # UTM zone 19N (correct for Massachusetts)
    transformer = Transformer.from_crs(
        "EPSG:4326",      # WGS84
        "EPSG:32619",     # UTM zone 19N
        always_xy=True
    )

    poly_utm = transform(transformer.transform, poly_latlon)

    return poly_utm.area
def compute_sample_count(poly,
                         sample_time,
                         speed,
                         mission_time,
                         num_agents=1):
    """
    Returns TOTAL sampling points all agents can complete.

    poly: shapely Polygon in lat/lon
    sample_time: seconds per sample
    speed: m/s
    mission_time: per-agent time limit (s)
    num_agents: number of UAVs
    """

    area = polygon_area_m2(poly)

    # Upper bound assuming zero travel time
    max_possible = int((mission_time * num_agents) / sample_time)

    best_N = 0

    for N in range(1, max_possible + 1):

        N_per_agent = N / num_agents

        sampling_time = N_per_agent * sample_time
        travel_time = math.sqrt(area * N_per_agent) / speed

        total_time = sampling_time + travel_time

        if total_time <= mission_time:
            best_N = N
        else:
            break  # further N will only increase time

    return best_N
############### SETTINGS #################

# example domain: any closed polygon (first and last point may coincide)
poly = Polygon([
    (-71.2633943948142, 42.29151053590485),
    (-71.2628929214584, 42.290784490885386),
    (-71.26149845009414, 42.291353821977765),
    (-71.26200796847752, 42.29205010492521),
    (-71.2633943948142, 42.29151053590485)])

N_dots = compute_sample_count(
    poly,
    sample_time=180,        # 3 minute per soil sampling time
    speed=5.0,            # 5 m/s cruise
    mission_time=480,      # 15 minute mission
    num_agents=3
)



######PRINT######
print(N_dots)
