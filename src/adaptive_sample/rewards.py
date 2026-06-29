import numpy as np


class CompositeReward:
    def __init__(self,w_mean,w_std,w_mean_grad,w_dist,w_far_from_mean):
        self.w_mean = w_mean
        self.w_std = w_std
        self.w_mean_grad = w_mean_grad
        self.w_dist = w_dist
        self.w_far_from_mean = w_far_from_mean
    
    def evaluate(self, mean, mean_field, std, mean_grad, current_pos, target_pos):
        reward = 0
        reward += self.w_mean * mean
        reward += self.w_std * std
        reward -= self.w_dist * np.linalg.norm(current_pos - target_pos)
        reward += self.w_mean_grad * mean_grad
        reward += self.w_far_from_mean * abs(mean - mean_field)
        return reward


class MaximumVarianceReward:
    def __init__(self):
        pass

    def evaluate(self, mean, std, mean_grad, current_pos, target_pos):
        return std

class UCBReward:
    def __init__(self,std_weight):
        self.beta = std_weight

    def evaluate(self, mean, std, mean_grad, current_pos, target_pos):
        return mean + self.beta * std

class VarianceMinusDistanceReward:
    def __init__(self,distance_weight):
        self.distance_weight = distance_weight
    
    def evaluate(self, mean, std, mean_grad, current_pos, target_pos):
        return std - self.distance_weight * np.linalg.norm(current_pos - target_pos)

# Very rough estimate --- let's assume cost + budget is in seconds
SAMPLE_COST = 60 # 60 seconds to sample
COST_PER_METER = 0.5 # 2 m/s

def compute_cost(current,target,sample=True):
    # Cost is distance plus sampling time
    # arbitrary values right now
    dist = np.linalg.norm(current - target)
    if sample:
        return SAMPLE_COST + dist * COST_PER_METER
    return dist * COST_PER_METER

import numpy as np
from shapely.geometry import Point

def get_variance_metrics(model, region, resolution=5.0):
    minx, miny, maxx, maxy = region.bounds

    xs = np.arange(minx, maxx + resolution, resolution)
    ys = np.arange(miny, maxy + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)

    grid = np.column_stack([xx.ravel(), yy.ravel()])
    inside = np.array([region.contains(Point(x, y)) for x, y in grid])

    grid_inside = grid[inside]

    if len(grid_inside) == 0:
        return {
            "avg_variance": np.nan,
            "max_variance": np.nan
        }

    _, std = model.gp.predict(grid_inside, return_std=True)
    var = std**2

    return {
        "avg_variance": float(np.mean(var)),
        "max_variance": float(np.max(var))
    }
