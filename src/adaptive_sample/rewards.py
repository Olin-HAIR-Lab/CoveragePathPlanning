import numpy as np

class MaximumVarianceReward:
    def __init__(self):
        pass

    def evaluate(self, mean, std, current_pos, target_pos):
        return std

class UCBReward:
    def __init__(self,std_weight):
        self.beta = std_weight

    def evaluate(self, mean, std, current_pos, target_pos):
        return mean + self.beta * std

class VarianceMinusDistanceReward:
    def __init__(self,distance_weight):
        self.distance_weight = distance_weight
    
    def evaluate(self, mean, std, current_pos, target_pos):
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
