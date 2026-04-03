import json
import numpy as np
from shapely.geometry import Point, Polygon

def makeJSONMission(file_path, path1=None, path2=None, path3=None):
    data = {
        "": {"waypoints": []},
        "px4_1/": {"waypoints": []},
        "px4_2/": {"waypoints": []},
    }

    drone_keys = ["", "px4_1/", "px4_2/"]
    paths = [path1, path2, path3]

    for key, path in zip(drone_keys, paths):
        if path:
            data[key]["waypoints"] = [
                {"lat": point.x, "lon": point.y, "alt": point.z}
                for point in path
            ]

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    path1 = [Point(42.292506, -71.262602, 3.0), Point(42.292432, -71.262597, 3.0)]
    path2 = [Point(42.292699, -71.262827, 3.0), Point(42.292588, -71.262906, 3.0)]
    path3 = [Point(42.292100, -71.262500, 3.0), Point(42.292200, -71.262400, 3.0)]

    # Test with 3 drones
    makeJSONMission("test_3drone.json", path1, path2, path3)
    print("\n3 drone output:")
    print(json.dumps(json.load(open("test_3drone.json")), indent=4))