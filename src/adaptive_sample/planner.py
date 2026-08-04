from __future__ import annotations
import numpy as np
import copy
from dataclasses import dataclass
from rewards import compute_cost, MaximumVarianceReward, VarianceMinusDistanceReward
from model import MoistureModel
from shapely.geometry import Point
from shapely.prepared import prep

def get_gp_length_scale(gp) -> float:
    params = gp.kernel.get_params()

    length_scale_keys = [key for key in params if key.endswith("length_scale")]

    if not length_scale_keys:
        raise ValueError(
            f"Could not find an RBF length scale in kernel {gp.kernel_}"
        )

    length_scale = np.asarray(
        params[length_scale_keys[0]],
        dtype=float,
    )

    return float(length_scale.item())

def neighborhood_std_reward(
    gp,
    candidate,
    region,
    fixed_radius=None,
    resolution=2.0,
    radius_scale=1.0,
):
    candidate = np.asarray(candidate, dtype=float).reshape(2)

    length_scale = get_gp_length_scale(gp)

    if fixed_radius is None:
        radius = max(radius_scale * length_scale, resolution)
    else:
        radius = fixed_radius

    # Grid covering the candidate-centered disk.
    offsets = np.arange(-radius, radius + resolution, resolution)
    dx, dy = np.meshgrid(offsets, offsets)

    disk_mask = dx**2 + dy**2 <= radius**2

    eval_pts = np.column_stack([
        candidate[0] + dx[disk_mask],
        candidate[1] + dy[disk_mask],
    ])

    # Remove points outside the field.
    prepared_region = prep(region)
    inside_mask = np.fromiter(
        (
            prepared_region.covers(Point(float(x), float(y)))
            for x, y in eval_pts
        ),
        dtype=bool,
        count=len(eval_pts),
    )

    eval_pts = eval_pts[inside_mask]

    if len(eval_pts) == 0:
        return 0.0

    _, std = gp.predict(eval_pts, return_std=True)

    # Each grid point represents approximately resolution**2 square metres.
    cell_area = resolution**2
    integrated_std = np.sum(std) * cell_area

    full_circle_area = np.pi * radius**2

    return float(integrated_std / full_circle_area)

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
    def __init__(
        self,
        rwd_fun,
        region,
        integrate_std,
        fixed_length_scale = None,
        n_steps=3,
        std_resolution=2.0,
        std_radius_scale=1.0
    ):
        self.n_steps = n_steps
        self.rwd_fun = rwd_fun
        self.region = region
        self.integrate_std = integrate_std
        self.fixed_length_scale = fixed_length_scale
        self.std_resolution = std_resolution
        self.std_radius_scale = std_radius_scale
    
    def plan(self, model, candidates, current_pos, budget_remaining, field_mean, log=False):
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

                    if self.integrate_std:
                        mean = child_model.gp.predict(candidate.reshape(-1,2)).item()
                        std = neighborhood_std_reward(
                            gp=child_model.gp,
                            candidate=candidate,
                            region=self.region,
                            fixed_radius=self.fixed_length_scale,
                            resolution=self.std_resolution,
                            radius_scale=self.std_radius_scale,
                        )
                    
                    else:
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

                    reward = self.rwd_fun.evaluate(mean, field_mean, std, grad_mean, node.current_pos, candidate)

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
