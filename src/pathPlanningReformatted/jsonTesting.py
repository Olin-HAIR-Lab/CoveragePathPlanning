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
                {"lat": point[0], "lon": point[1], "alt": point[2], "type": "sample"}
                for point in path
            ]
    
    for key in drone_keys: 
        data[key]["waypoints"][0]["type"] = "depot"
        data[key]["waypoints"][-1]["type"] = "depot"

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)