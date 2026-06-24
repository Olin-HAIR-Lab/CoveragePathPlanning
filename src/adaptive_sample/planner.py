import numpy as np
import copy
from rewards import compute_cost, MaximumVarianceReward, VarianceMinusDistanceReward

def score_virtual_path(model, rwd_fun, start_pos, path):
    virtual_model = copy.deepcopy(model)
    current_pos = start_pos
    total_score = 0.0

    for x in path:
        #print(f"Waypoint candidate: {x}")
        x = np.asarray(x, dtype=float).reshape(1, 2)
        # Evaluate model at position 
        mean,std = virtual_model.gp.predict(x,return_std=True)
        mean = np.asarray(mean).item()
        std = np.asarray(std).item()
        #print(f"Mean {mean:.2f}, std {std:.2f}")

        # Evaluate value of this next point under the current virtual model
        step_score = rwd_fun.evaluate(mean, std, current_pos=current_pos, target_pos=x)

        # Add this point virtually
        virtual_model.add_observation(x,mean,virtual=True)

        # Accumulate value
        total_score += step_score
        current_pos = x.reshape(2,)

    return total_score

class VarianceMinusDistancePlanner:
    def __init__(self):
        dist_weight = 0.001
        self.rwd_fun = VarianceMinusDistanceReward(dist_weight)
    
    def plan(self, model, candidates, current_pos, budget_remaining):
        best_score = -np.inf
        best_point = None

        for x in candidates:
            #print(f"Candidate: {x}")
            cost = compute_cost(current=current_pos, target=x, sample=True)

            if cost > budget_remaining:
                continue

            score = score_virtual_path(
                model=model,
                rwd_fun=self.rwd_fun,
                path=[x],
                start_pos=current_pos
            )

            # Information per unit cost
            print(f"Score {score:.4f}, cost {cost:.2f}")
            utility = score

            if utility > best_score:
                best_score = utility
                best_point = x

        if best_point is None:
            #print("Couldn't find any plans under budget!")
            return []

        print(f"Best point to sample is {best_point}")
        return [best_point]

class GreedyVariancePlanner:
    def __init__(self):
        self.rwd_fun = MaximumVarianceReward()

    def plan(self, model, candidates, current_pos, budget_remaining):
        best_score = -np.inf
        best_point = None

        for x in candidates:
            #print(f"Candidate: {x}")
            cost = compute_cost(current=current_pos, target=x, sample=True)

            if cost > budget_remaining:
                continue

            score = score_virtual_path(
                model=model,
                rwd_fun=self.rwd_fun,
                path=[x],
                start_pos=current_pos
            )

            # Information per unit cost
            print(f"Score {score:.4f}, cost {cost:.2f}")
            utility = score - 0.0001 * cost
            #print(f"Plan has utility {utility:.2f} from score {score:.2f}")

            if utility > best_score:
                best_score = utility
                best_point = x

        if best_point is None:
            #print("Couldn't find any plans under budget!")
            return []

        print(f"Best point to sample is {best_point}")
        return [best_point]