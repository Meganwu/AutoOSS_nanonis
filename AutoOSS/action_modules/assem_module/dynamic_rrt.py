#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Nian Wu
# Date: 2025-11-26
# Description: Module for assembling task with Nanonis controller

"""
Dynamic RRT* assignment for moving multiple molecules to target positions

- Molecules (all_mols_nm) move to target positions (design_mols_nm)
- Static obstacles (obstacles_nm)
- Each moved molecule becomes a new obstacle for the next moves
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# =========================
#  RRT* CORE IMPLEMENTATION
# =========================

class Node:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.parent = None
        self.cost = 0.0


def collision_segment(p1, p2, obstacles, r=0.4):
    """
    Check if the line segment p1 -> p2 intersects any obstacle (treated as a circle of radius r).
    obstacles: array of shape (N, 2)
    """
    if obstacles is None or len(obstacles) == 0:
        return False

    v = p2 - p1
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return False

    for o in obstacles:
        u = o - p1
        t = np.clip(np.dot(u, v) / (v_norm ** 2), 0.0, 1.0)
        proj = p1 + t * v
        if np.linalg.norm(proj - o) < r:
            return True

    return False


def extract_path(node):
    """Trace back through parents and return path as (N, 2) array."""
    path = []
    while node is not None:
        path.append(node.pos)
        node = node.parent
    return np.array(path[::-1])  # reverse order


def rrt_star(start,
             goal,
             obstacles,
             step_size=0.3,
             max_iter=3000,
             goal_radius=0.5,
             rewiring_radius=1.0,
             bounds=(0, 10, 0, 10),
             goal_bias=0.1,
             verbose=False):
    """
    Compute a path from start -> goal with RRT* in 2D.

    Parameters
    ----------
    start, goal : array-like, shape (2,)
    obstacles   : np.ndarray, shape (N, 2)
    bounds      : (xmin, xmax, ymin, ymax)

    Returns
    -------
    path : (M, 2) np.ndarray
    """
    xmin, xmax, ymin, ymax = bounds

    start_node = Node(start)
    nodes = [start_node]

    for it in range(max_iter):

        # 1) Sample (with goal bias)
        if np.random.rand() < goal_bias:
            sample = np.array(goal)
        else:
            sample = np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax)
            ])

        # 2) Nearest node
        dists = [np.linalg.norm(n.pos - sample) for n in nodes]
        nearest = nodes[int(np.argmin(dists))]

        direction = sample - nearest.pos
        dist = np.linalg.norm(direction)
        if dist == 0:
            continue
        direction /= dist

        new_pos = nearest.pos + step_size * direction

        # 3) Collision check
        if collision_segment(nearest.pos, new_pos, obstacles):
            continue

        # 4) Create new node
        new_node = Node(new_pos)
        new_node.parent = nearest
        new_node.cost = nearest.cost + np.linalg.norm(new_pos - nearest.pos)

        # 5) Choose best parent among neighbors
        for n in nodes:
            if np.linalg.norm(n.pos - new_pos) < rewiring_radius:
                if not collision_segment(n.pos, new_pos, obstacles):
                    c = n.cost + np.linalg.norm(n.pos - new_pos)
                    if c < new_node.cost:
                        new_node.parent = n
                        new_node.cost = c

        nodes.append(new_node)

        # 6) Rewire neighbors to new_node if shorter
        for n in nodes:
            if n is new_node:
                continue
            if np.linalg.norm(n.pos - new_pos) < rewiring_radius:
                if not collision_segment(n.pos, new_pos, obstacles):
                    c = new_node.cost + np.linalg.norm(n.pos - new_pos)
                    if c < n.cost:
                        n.parent = new_node
                        n.cost = c

        # 7) Goal reached?
        if np.linalg.norm(new_pos - goal) < goal_radius:
            if verbose:
                print(f"RRT*: goal reached at iteration {it}")
            return extract_path(new_node)

    # Fallback: nearest node to goal
    best = min(nodes, key=lambda n: np.linalg.norm(n.pos - goal))
    if verbose:
        print("RRT*: max_iter reached, returning best approximate path.")
    return extract_path(best)


def path_length(path):
    if path is None or len(path) < 2:
        return np.inf
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


def rrt_star_cost(start,
                  goal,
                  obstacles,
                  bounds=(0, 10, 0, 10),
                  step_size=0.4,
                  max_iter=800,
                  goal_radius=0.5,
                  rewiring_radius=1.0,
                  goal_bias=0.2):
    """
    Convenience wrapper: run a faster RRT* and return only the path length.
    Used for building the cost matrix in assignment.
    """
    path = rrt_star(start=start,
                    goal=goal,
                    obstacles=obstacles,
                    step_size=step_size,
                    max_iter=max_iter,
                    goal_radius=goal_radius,
                    rewiring_radius=rewiring_radius,
                    bounds=bounds,
                    goal_bias=goal_bias,
                    verbose=False)
    return path_length(path)


# =========================
#  DYNAMIC ASSIGNMENT LOGIC
# =========================

def dynamic_rrt_assignment(all_mols_nm,
                           design_mols_nm,
                           obstacles_nm,
                           bounds=(0, 10, 0, 10),
                           display_progress=True):
    """
    Dynamically assign molecules to targets, planning each move with RRT*.
    After each move, the molecule's final position is added as a new obstacle.

    Returns
    -------
    assign_sequence : list of (mol_idx, target_idx)
    paths           : list of np.ndarray (each (M, 2) path)
    """

    all_mols_nm = np.asarray(all_mols_nm, dtype=float)
    design_mols_nm = np.asarray(design_mols_nm, dtype=float)
    dynamic_obstacles = np.asarray(obstacles_nm, dtype=float)

    remaining_mols = list(range(len(all_mols_nm)))
    remaining_targets = list(range(len(design_mols_nm)))

    assign_sequence = []
    paths = []

    step_count = 0

    while remaining_mols and remaining_targets:
        step_count += 1
        if display_progress:
            print(f"\n=== Step {step_count} ===")
            print(f"Remaining mols: {remaining_mols}")
            print(f"Remaining targets: {remaining_targets}")

        # Build cost matrix for current obstacles
        cost_matrix = np.zeros((len(remaining_mols), len(remaining_targets)))

        for i, mol_idx in enumerate(remaining_mols):
            start = all_mols_nm[mol_idx]
            for j, tgt_idx in enumerate(remaining_targets):
                goal = design_mols_nm[tgt_idx]
                c = rrt_star_cost(
                    start=start,
                    goal=goal,
                    obstacles=dynamic_obstacles,
                    bounds=bounds,
                    step_size=0.4,
                    max_iter=600,     # keep moderate for speed
                    goal_radius=0.5,
                    rewiring_radius=1.0,
                    goal_bias=0.2
                )
                cost_matrix[i, j] = c

        # Solve assignment for this reduced problem
        rows, cols = linear_sum_assignment(cost_matrix)
        pair_costs = cost_matrix[rows, cols]

        # Choose the best pair (greedy on this step)
        best_idx = int(np.argmin(pair_costs))
        row_sel = rows[best_idx]
        col_sel = cols[best_idx]

        mol_idx = remaining_mols[row_sel]
        tgt_idx = remaining_targets[col_sel]

        start = all_mols_nm[mol_idx]
        goal = design_mols_nm[tgt_idx]

        if display_progress:
            print(f"Selected mol {mol_idx} -> target {tgt_idx}, "
                  f"approx cost = {pair_costs[best_idx]:.3f}")

        # Now compute a detailed RRT* path (larger iterations)
        path = rrt_star(
            start=start,
            goal=goal,
            obstacles=dynamic_obstacles,
            step_size=0.4,
            max_iter=4000,
            goal_radius=0.4,
            rewiring_radius=1.0,
            bounds=bounds,
            goal_bias=0.15,
            verbose=display_progress
        )

        assign_sequence.append((mol_idx, tgt_idx))
        paths.append(path)

        # Update dynamic obstacles: add final position of moved molecule
        final_pos = path[-1]
        dynamic_obstacles = np.vstack([dynamic_obstacles, final_pos])

        # Remove used indices from remaining sets
        remaining_mols.remove(mol_idx)
        remaining_targets.remove(tgt_idx)

    return assign_sequence, paths




