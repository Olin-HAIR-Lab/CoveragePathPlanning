from __future__ import annotations
import numpy as np
import copy
from dataclasses import dataclass
from rewards import compute_cost, MaximumVarianceReward, VarianceMinusDistanceReward
from model import MoistureModel

@dataclass
class TreeNode:
    parent: TreeNode | None
    path: list | None
    score: float
    budget_remaining: float
    current_pos: np.ndarray
    model: MoistureModel
    used_indices: list

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

class NStepLookaheadPlanner:
    def __init__(self,rwd_fun,n_steps=3):
        self.n_steps = n_steps
        self.rwd_fun = rwd_fun
    def plan(self, model, candidates, current_pos, budget_remaining, log=False):
        # Start with the root node (zero reward, hasn't traveled at all)
        root = TreeNode(
            parent=None,
            path=[],
            score=0,
            model=model,
            budget_remaining=budget_remaining,
            current_pos=current_pos,
            used_indices=set()
        )
        frontier = [root] # all the alive leaf nodes
        best_score = -np.inf
        best_node = None

        for depth in range(self.n_steps):
            new_frontier = []

            for node in frontier:
                for candidate_idx, candidate in enumerate(candidates):
                    if candidate_idx in node.used_indices:
                        # This would involve repeating a measurement
                        continue

                    cost = compute_cost(node.current_pos, candidate, sample=True)
                    if cost > node.budget_remaining:
                        # This would go over budget
                        continue

                    child_model = copy.deepcopy(node.model)

                    #print(f"Trying to predict {candidate}")
                    mean, std = child_model.gp.predict(candidate.reshape(-1,2),return_std=True)

                    # If we want the gradient of the mean, we need a grid around the point we want
                    h = 5.0
                    xs = candidate[0] + np.array([-h, 0.0, h])
                    ys = candidate[1] + np.array([-h, 0.0, h])

                    xx, yy = np.meshgrid(xs, ys)
                    eval_pts = np.column_stack([xx.ravel(), yy.ravel()])
                    mean_grid = child_model.gp.predict(eval_pts).reshape(3, 3)

                    # np.gradient returns [d/dy, d/dx]
                    dmean_dy, dmean_dx = np.gradient(mean_grid, h)

                    # Center point gradient
                    grad_mean = np.linalg.norm(np.array([
                        dmean_dx[1, 1],
                        dmean_dy[1, 1],
                    ]))

                    reward = self.rwd_fun.evaluate(mean, std, grad_mean, node.current_pos, candidate)

                    child_model.add_observation(candidate, mean, virtual=True)

                    child = TreeNode(
                        parent=node,
                        path=node.path + [candidate],
                        model=child_model,
                        score=node.score + reward,
                        current_pos=candidate,
                        budget_remaining=node.budget_remaining - cost,
                        used_indices=node.used_indices | {candidate_idx},
                    )

                    new_frontier.append(child)

                    if child.score > best_score:
                        best_score = child.score
                        best_node = child

            frontier = new_frontier
            if log:
                print(f"Finished depth {depth} with {len(frontier)} leaf nodes")
            
        
        if best_node is None:
            best_node = root
            best_score = root.score
        
        if log:
            print(f"Best score is {best_node.score} for path {best_node.path}")
        if best_node is None or len(best_node.path) == 0:
            return [], frontier, best_node
        if log:
            print(f"Choosing waypoint {best_node.path[0]}")
        return [best_node.path[0]], frontier, best_node


class VarianceMinusDistancePlanner:
    def __init__(self):
        dist_weight = 0.001
        self.rwd_fun = VarianceMinusDistanceReward(dist_weight)
    
    def plan(self, model, candidates, current_pos, budget_remaining):
        best_score = -np.inf
        best_candidate = None
        print(f"Candidates: {candidates}")

        for x in candidates:
            print(f"Candidate: {x}")
            cost = compute_cost(current=current_pos, target=x, sample=True)

            if cost > budget_remaining:
                continue

            score = score_virtual_path(
                model=model,
                rwd_fun=self.rwd_fun,
                path=x,
                start_pos=current_pos
            )

            # Information per unit cost
            print(f"Score {score:.4f}, cost {cost:.2f}")
            utility = score

            if utility > best_score:
                best_score = utility
                best_candidate = x

        if best_candidate is None:
            #print("Couldn't find any plans under budget!")
            return []

        print(f"Best point to sample is {best_candidate[0]} from path {best_candidate} with score {best_score}")
        return [best_candidate[0]]

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