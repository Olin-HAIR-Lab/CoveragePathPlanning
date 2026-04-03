import math
from shapely.ops import transform
from shapely.geometry import Point, Polygon
from pyproj import Transformer


def polygon_area_m2(poly_latlon):
    """
    Convert WGS84 polygon to UTM and return area in m².
    """
    transformer = Transformer.from_crs(
        "EPSG:4326",   # WGS84
        "EPSG:32619",  # UTM zone 19N (Massachusetts)
        always_xy=True
    )

    poly_utm = transform(transformer.transform, poly_latlon)
    return poly_utm.area


class SamplePoints:
    def __init__(self, map_polygon, home, sample_time, speed, mission_time, num_agents):
        """
        map_polygon: shapely Polygon (lat/lon)
        home: (lat, lon)
        sample_time: seconds per sample
        speed: m/s
        mission_time: seconds (per agent)
        num_agents: number of UAVs
        """
        self.map = map_polygon
        self.home = home
        self.sample_time = sample_time
        self.speed = speed
        self.mission_time = mission_time
        self.num_agents = num_agents

    def compute_sample_count(self):
        """
        Returns TOTAL sampling points all agents can complete.
        Ensures result is divisible by num_agents.
        """
        area = polygon_area_m2(self.map)

        max_possible = int(
            (self.mission_time * self.num_agents) / self.sample_time
        )

        best_N = 0

        # iterate ONLY over divisible values (cleaner + faster)
        for N in range(self.num_agents, max_possible + 1, self.num_agents):

            N_per_agent = N / self.num_agents

            sampling_time = N_per_agent * self.sample_time
            travel_time = math.sqrt(area * N_per_agent) / self.speed

            total_time = sampling_time + travel_time

            if total_time <= self.mission_time:
                best_N = N
            else:
                break  # safe: time increases monotonically

        return best_N


# ================== TEST / EXAMPLE ==================

soccer_field = Polygon([
    (-71.2633943948142, 42.29151053590485),
    (-71.2628929214584, 42.290784490885386),
    (-71.26149845009414, 42.291353821977765),
    (-71.26200796847752, 42.29205010492521),
    (-71.2633943948142, 42.29151053590485)
])

soccer_home = (-71.2633943948142, 42.29151053590485)

sampler = SamplePoints(
    soccer_field,
    soccer_home,
    sample_time=180,   # 3 min/sample
    speed=5.0,         # m/s
    mission_time=480,  # 8 min
    num_agents=3
)

N_dots = sampler.compute_sample_count()

print("Total sample points:", N_dots)
