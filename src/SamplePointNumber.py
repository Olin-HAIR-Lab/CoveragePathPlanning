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


# def compute_sample_count(map,
#                          home,
#                          sample_time,
#                          speed,
#                          mission_time,
#                          num_agents=1):
#     """
#     Returns TOTAL sampling points all agents can complete.

#     map: shapely Polygon in lat/lon
#     home: (lat, lon) home/base point
#     sample_time: seconds per sample
#     speed: m/s
#     mission_time: per-agent time limit (s)
#     num_agents: number of UAVs
#     """

#     area = polygon_area_m2_2(map)
#     centroid = map.centroid
#     home_point = Point(home[1], home[0])  # shapely uses (lon, lat)

#     # Convert home->centroid distance from degrees to metres
#     home_to_area_m = haversine_m(home[0], home[1],
#                                  centroid.y, centroid.x)

#     # Round-trip home overhead per agent (fixed cost regardless of N)
#     home_travel_time = (2 * home_to_area_m) / speed

#     # Upper bound assuming zero travel time
#     max_possible = int((mission_time * num_agents) / sample_time)

#     best_N = 0

#     for N in range(1, max_possible + 1):

#         N_per_agent = N / num_agents

#         sampling_time = N_per_agent * sample_time

#         # Internal traversal among sample points
#         internal_travel_time = math.sqrt(area * N_per_agent) / speed

#         # Total = fixed home round-trip + sampling + internal travel
#         total_time = home_travel_time + sampling_time + internal_travel_time

#         if total_time <= mission_time:
#             best_N = N
#         else:
#             break

#     return best_N
class SamplePoints:
    def __init__(self, map, home, sample_time, speed, mission_time, num_agents):
        self.map = map
        self.home = home
        self.sample_time = sample_time
        self.speed = speed
        self.mission_time = mission_time
        self.num_agents = num_agents
    def compute_home(home, speed, mission_time):
        pass
    def compute_sample_count(map,
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

        area = polygon_area_m2(map)

        # Upper bound assuming zero travel time
        max_possible = int((mission_time * num_agents) / sample_time)
        print(max_possible)
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
soccer_field = Polygon([
    (-71.2633943948142, 42.29151053590485),
    (-71.2628929214584, 42.290784490885386),
    (-71.26149845009414, 42.291353821977765),
    (-71.26200796847752, 42.29205010492521),
    (-71.2633943948142, 42.29151053590485)])
soccer_home = (-71.2633943948142, 42.29151053590485)
soccer_sample = SamplePoints
N_dots = soccer_sample.compute_sample_count(
    soccer_field,
    soccer_home,
    sample_time=180,        # 3 minute per soil sampling time
    speed=5.0,            # 5 m/s cruise
    mission_time=480,      # 15 minute mission
    num_agents=3
)



######PRINT######
print(N_dots)
