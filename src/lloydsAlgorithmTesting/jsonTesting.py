import json
import numpy as np
from shapely.geometry import Point, Polygon

def makeJSONMission(path, file_path):

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Clear previous data
    data['mission']['items'] = []

    # Add takeoff
    data['mission']['items'].append({'type': 'takeoff'})

    # Add waypoints
    for point in path:
        data['mission']['items'].append({
            'type': 'navigation',
            'navigationType': 'waypoint',
            'x': str(point.x),
            'y': str(point.y),
            'z': str(point.z),
            'frame': 'global'
        })
        data['mission']['items'].append({'type': 'sample'})

    # Add return to launch
    data['mission']['items'].append({'type': 'rtl'})


    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)


file_path = 'pathJSONTemplate.json'
testList = np.array(
    [Point(1,2,3),Point(1,2,3),Point(1,2,3),Point(1,2,3)]
)
makeJSONMission(testList, file_path)