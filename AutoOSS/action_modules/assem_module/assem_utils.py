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

import sys
import numpy as np
import pandas as pd
import time
from collections import namedtuple
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import pyplot as plt, patches


sys.path.append("c:\\Users\\wun2\\github\\AutoOSS_20251126")


from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt



from scipy.spatial import distance


def update_obstacles_mols(all_mols_pos, obstacles_pos, threshold : float = 0.1):

    '''
    Update the obstacles list by adding molecules that are too close to the obstacles.
    Parameters:
    obstacles_pos (list or np.array): List of obstacle positions.
    all_mols_pos (list or np.array): List of all molecule positions.
    threshold (float): unit: nm, Distance threshold to consider a molecule too close to an obstacle.'''
    print("Initial number of obstacles:", len(obstacles_pos), "; Initial number of molecules:", len(all_mols_pos))
    if type(obstacles_pos) is list:
        obstacles_pos = np.array(obstacles_pos)
    if type(all_mols_pos) is list:
        all_mols_pos = np.array(all_mols_pos)
    updated_obstacles_pos = obstacles_pos.tolist()
    updated_all_mols_pos = all_mols_pos.tolist()
    done=False
    while not done:
        dist=cdist(np.array(updated_obstacles_pos), np.array(updated_all_mols_pos))
        obs_idx, mol_idx = np.where(dist < threshold)
        if len(obs_idx) == 0:
            done=True
        else:
            for o_idx, m_idx in zip(obs_idx, mol_idx):
                updated_obstacles_pos.append(updated_all_mols_pos[m_idx])
                updated_all_mols_pos.pop(m_idx)
                break

    print("Updated number of obstacles:", len(updated_obstacles_pos), "; Updated number of molecules:", len(updated_all_mols_pos))

    return updated_all_mols_pos, updated_obstacles_pos



def collision_segment(pos1, pos2, obstacles):
    """
    Check if the line segment p1 -> p2 intersects any obstacle (treated as a circle of radius r).
    obstacles: list of shape (N, 3) (x, y r)
    """
    if obstacles is None or len(obstacles) == 0:
        return False

    v = pos2 - pos1
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return False

    for obs in obstacles:
        obs_pos=np.array(obs[:2])
        obs_r=obs[2]
        u = obs_pos - pos1
        t = np.clip(np.dot(u, v) / (v_norm ** 2), 0.0, 1.0)
        proj = pos1 + t * v
        if np.linalg.norm(proj - obs_pos) < obs_r:
            return True

    return False


def extract_path(node):
    """Trace back through parents and return path as (N, 2) array."""
    path = []
    while node is not None:
        path.append(node.pos)
        node = node.parent
    return np.array(path[::-1])  # reverse order


def path_length(path):
    if path is None or len(path) < 2:
        return np.inf
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


