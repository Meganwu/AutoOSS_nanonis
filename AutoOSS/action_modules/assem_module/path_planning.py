
import sys
import numpy as np
import pandas as pd
import time
from collections import namedtuple
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt


sys.path.append("c:\\Users\\wun2\\github\\AutoOSS_20251126")


from AutoOSS.action_modules.assem_module.assem_utils import collision_segment, extract_path, path_length
import heapq


import heapq
import numpy as np

# RRT-star



class Node:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.parent = None
        self.cost = 0.0

def rrt(start,
        goal,
        obstacles,
        step_size=2.0,
        max_iter=2000,
        goal_threshold=0.5,
        bound_margin=10,
        goal_bias=0.05):
    """
    Simple RRT (NOT RRT*) for 2D motion planning.

    Parameters
    ----------
    start, goal : array-like, shape (2,)
    obstacles   : list of [x, y, r]
    step_size   : extension distance per RRT step
    goal_threshold : distance threshold to goal
    bounds      : float: margin 
    goal_bias   : probability of sampling goal directly

    Returns
    -------
    path : (N, 2) array
    """
    
    min_x = np.min([start[0], goal[0]])-bound_margin
    max_x = np.max([start[0], goal[0]])+bound_margin
    min_y = np.min([start[1], goal[1]])-bound_margin
    max_y = np.max([start[1], goal[1]])+bound_margin    

    start_node = Node(start)
    nodes = [start_node]

    for it in range(max_iter):

        # --------------------------------------------------
        # 1. RANDOM SAMPLE (with goal bias)
        # --------------------------------------------------
        if np.random.random() < goal_bias:
            sample = np.array(goal)
        else:
            sample = np.array([
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y)
            ])
        
        # --------------------------------------------------
        # 2. NEAREST NODE
        # --------------------------------------------------
        dists = [np.linalg.norm(n.pos - sample) for n in nodes]
        nearest = nodes[int(np.argmin(dists))]

        direction = sample - nearest.pos
        dist = np.linalg.norm(direction)
        if dist == 0:
            continue
        direction /= dist

        # --------------------------------------------------
        # 3. EXTEND TOWARD SAMPLE
        # --------------------------------------------------
        new_pos = nearest.pos + step_size * direction

        # Check bounds
        if not (min_x <= new_pos[0] <= max_x and min_y <= new_pos[1] <= max_y):
            continue

        # --------------------------------------------------
        # 4. COLLISION CHECK
        # --------------------------------------------------
        if collision_segment(nearest.pos, new_pos, obstacles):
            continue

        # --------------------------------------------------
        # 5. ADD NEW NODE
        # --------------------------------------------------
        new_node = Node(new_pos)
        new_node.parent = nearest
        nodes.append(new_node)

        # --------------------------------------------------
        # 6. GOAL CHECK
        # --------------------------------------------------
        if np.linalg.norm(new_pos - goal) <= goal_threshold:
            print(f"RRT reached goal at iteration {it}")
            return extract_path(new_node)

    # If no direct goal, return best partial path
    print("RRT did not directly reach goal — returning nearest node path.")
    best = min(nodes, key=lambda n: np.linalg.norm(n.pos - goal))
    return extract_path(best)

def rrt_star(start,
             goal,
             obstacles,
             step_size=2,
             max_iter=3000,
             goal_threshold=0.5,
             rewiring_radius=1.0,
             bound_margin=10,
             goal_bias=0.1,
             verbose=False):
    """
    Compute a path from start -> goal with RRT* in 2D.

    Parameters
    ----------
    start, goal : array-like, shape (2,)
    obstacles   : list, array-like, shape (N, 3) (x, y, r)
    bounds      : (min_x, xmax, ymin, ymax)

    Returns
    -------
    path : (M, 2) np.ndarray
    """
    min_x = np.min([start[0], goal[0]])-bound_margin
    max_x = np.max([start[0], goal[0]])+bound_margin
    min_y = np.min([start[1], goal[1]])-bound_margin
    max_y = np.max([start[1], goal[1]])+bound_margin

    start_node = Node(start)
    nodes = [start_node]
    
    goal_nodes=[]

    for it in range(max_iter):

        # 1) Sample (with goal bias)
        if np.random.rand() < goal_bias:
            sample = np.array(goal)
        else:
            sample = np.array([
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y)
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
            dist_to_new = np.linalg.norm(n.pos - new_pos)
            if dist_to_new < rewiring_radius and (dist_to_new < step_size):
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
            dist_to_new = np.linalg.norm(n.pos - new_pos)
            if dist_to_new < rewiring_radius and (dist_to_new < step_size):
                if not collision_segment(n.pos, new_pos, obstacles):
                    c = new_node.cost + np.linalg.norm(n.pos - new_pos)
                    if c < n.cost:
                        n.parent = new_node
                        n.cost = c

        # 7) Goal reached?
        if np.linalg.norm(new_pos - goal) < goal_threshold:
            goal_nodes.append(new_node)
            if verbose:
                print(f"RRT*: goal reached at iteration {it}")
                path=extract_path(new_node)
                plt.plot(path[:, 0], path[:, 1], linewidth=2)

            # return extract_path(new_node)

    # Fallback: least steps to goal

    if len(goal_nodes)>0:
            best_goal_node = min(goal_nodes, key=lambda n: path_length(extract_path(n)))
            path=extract_path(best_goal_node)
            plt.scatter(path[:, 0], path[:, 1], color='blue',label='RRT* path')
            return extract_path(best_goal_node)
    else:
            # fallback: nearest node but NOT optimal
            print("Warning: no valid goal node found!")
            best = min(nodes, key=lambda n: path_length(extract_path(n)))
            return extract_path(best)
    



def astar_fast(start, goal, obstacles,
                      step_size=2.0,      # max movement per step
                      grid_size=0.2,      # resolution
                      bound_margin=10,    # margin around start/goal
                      goal_threshold=0.5  # the distance to goal to stop
                      ):
    """
    A* on a dense grid where each move must be <= step_size.
    Stops when within goal_threshold of target.
    """

    # -----------------------------
    # SNAP TO GRID
    # -----------------------------
    def snap_to_grid(p):
        return tuple(np.round(np.array(p) / grid_size).astype(int))

    start_i = snap_to_grid(start)
    goal_i  = snap_to_grid(goal)

    bound_margin_i = int(bound_margin / grid_size)

    goal_real = np.array(goal)  # exact goal coordinate

    print("start_i:", start_i, "goal_i:", goal_i)

    # -----------------------------
    # NEIGHBOR OFFSETS (<= step_size)
    # -----------------------------
    max_d = int(step_size / grid_size)  # e.g. 2 / 0.2 = 10 grid units

    neighbor_offsets = []
    for dx in range(-max_d, max_d + 1):
        for dy in range(-max_d, max_d + 1):
            if dx == 0 and dy == 0:
                continue
            if dx*dx + dy*dy <= max_d*max_d:
                neighbor_offsets.append((dx, dy))

    # -----------------------------
    # BOUNDING BOX
    # -----------------------------
    min_x = min(start_i[0], goal_i[0]) - bound_margin_i
    max_x = max(start_i[0], goal_i[0]) + bound_margin_i
    min_y = min(start_i[1], goal_i[1]) - bound_margin_i
    max_y = max(start_i[1], goal_i[1]) + bound_margin_i

    # -----------------------------
    # COLLISION CHECK
    # -----------------------------
    coll_cache = {}

    def is_free(a, b):
        key = (a, b)
        if key in coll_cache:
            return coll_cache[key]

        pa = np.array(a) * grid_size
        pb = np.array(b) * grid_size

        hit = collision_segment(pa, pb, obstacles)
        coll_cache[key] = not hit
        return not hit

    # -----------------------------
    # A* SEARCH
    # -----------------------------
    pq = []
    heapq.heappush(pq, (0, 0, start_i))
    came_from = {start_i: None}
    g_score = {start_i: 0}

    while pq:
        f, g, (x, y) = heapq.heappop(pq)
        pos_real = np.array([x * grid_size, y * grid_size])

        # ⭐ NEW GOAL CONDITION: reach within tolerance
        if np.linalg.norm(pos_real - goal_real) <= goal_threshold:
            # reconstruct path
            grid_path = []
            cur = (x, y)
            while cur is not None:
                grid_path.append(cur)
                cur = came_from[cur]
            grid_path.reverse()

            # convert to real coords
            path_real = [(ix*grid_size, iy*grid_size) for (ix, iy) in grid_path]
            path_real[-1] = tuple(goal_real)  # replace last with exact goal

            return path_real

        # expand neighbors
        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy

            # bounding box
            if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                continue

            nxt = (nx, ny)
            if not is_free((x, y), nxt):
                continue

            new_cost = g + 1

            if nxt not in g_score or new_cost < g_score[nxt]:
                g_score[nxt] = new_cost
                came_from[nxt] = (x, y)

                # Heuristic: Chebyshev distance in units of allowed-step
                hx = abs(goal_i[0] - nx)
                hy = abs(goal_i[1] - ny)
                h = max(hx, hy) / max_d  

                heapq.heappush(pq, (new_cost + h, new_cost, nxt))

    return None

