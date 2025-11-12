#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Author: Nian Wu
Date: 2025-07-29    
Description: Utility functions for AutoOSS assembly and design alignment.
"""

from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import time
from collections import namedtuple
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import pyplot as plt, patches
from scipy.spatial import distance

from AutoOSS.env_modules.rrt import RRT


def assignment(start, goal):
    """
    Assign start to goal with the linear_sum_assignment function and setting the cost matrix to the distance between each start-goal pair

    Parameters
    ----------
    start, goal: array_like
        start and goal positions

    Returns
    -------
    np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost: array_like
            sorted start and goal positions, and their distances

    total_cost: float
            total distances
    
    row_ind, col_ind: array_like
            Indexes of the start and goal array in sorted order
    """
    cost_matrix = cdist(np.array(start)[:,:2], np.array(goal)[:,:2])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind]
    total_cost = np.sum(cost)
    return np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost, total_cost, row_ind, col_ind

def align_design(atoms, design):
    """
    Move design positions and assign atoms to designs to minimize total manipulation distance 

    Parameters
    ----------
    atoms, design: array_like
        atom and design positions

    Returns
    -------
    atoms_assigned, design_assigned: array_like
            sorted atom and design (moved) positions
    
    anchor: array_like
            position of the atom that will be used as the anchor
    """
    assert atoms.shape == design.shape
    c_min = np.inf
    for i in range(atoms.shape[0]):
        for j in range(design.shape[0]):
            a = atoms[i,:]
            d = design[j,:]
            design_ = design+a-d
            a_index = np.delete(np.arange(atoms.shape[0]), i)
            d_index = np.delete(np.arange(design.shape[0]), j)
            a, d, _, c, _, _ = assignment(atoms[a_index,:], design_[d_index,:])
            if (c<c_min):
                c_min = c
                atoms_assigned, design_assigned = a, d
                anchor = atoms[i,:]
    return atoms_assigned, design_assigned, anchor

def align_deisgn_stitching(all_atom_absolute_nm, design_nm, align_design_params):
    """
    Shift the designs to match the atoms based on align_design_params. 
    Assign atoms to designs to minimize total manipulation distance.
    Get the obstacle list from align_design_params

    Parameters
    ----------
    all_atom_absolute_nm, design_nm: array_like
        atom and design positions

    align_design_params: dict
        {'atom_nm', 'design_nm', 'obstacle_nm'} 

    Returns
    -------
    atoms, designs: array_like
            sorted atom and design (moved) positions
    
    anchor_atom_nm: array_like
            position of the atom that will be used as the anchor
    """
    anchor_atom_nm = align_design_params['atom_nm']
    anchor_design_nm = align_design_params['design_nm']
    obstacle_nm = align_design_params['obstacle_nm']
    assert anchor_design_nm.tolist() in design_nm.tolist()
    dist = cdist(all_atom_absolute_nm, anchor_atom_nm.reshape((-1,2)))
    anchor_atom_nm = all_atom_absolute_nm[np.argmin(dist),:]
    atoms = np.delete(all_atom_absolute_nm, np.argmin(dist), axis=0)
    dist = cdist(design_nm, anchor_design_nm.reshape((-1,2)))
    designs = np.delete(design_nm, np.argmin(dist), axis=0)
    designs += (anchor_atom_nm - anchor_design_nm)
    if obstacle_nm is not None:
        obstacle_nm[:,:2] = obstacle_nm[:,:2]+(anchor_atom_nm - anchor_design_nm)
    return atoms, designs, anchor_atom_nm, obstacle_nm

def get_atom_and_anchor(all_atom_absolute_nm, anchor_nm):
    """
    Separate the positions of the anchor and the rest of the atoms 

    Parameters
    ----------
    all_atom_absolute_nm, anchor_nm: array_like
        positions of all the atoms and the anchor

    Returns
    -------
    atoms_nm, new_anchor_nm: array_like
            positions of all the atoms (except the anchor) and the anchor
    """
    new_anchor_nm, anchor_nm, _, _, row_ind, _ = assignment(all_atom_absolute_nm, anchor_nm)
    atoms_nm = np.delete(all_atom_absolute_nm, row_ind, axis=0)
    return atoms_nm, new_anchor_nm



def get_path_from_rl(atom_chosen, design_chosen, agent, env_toy):
    atom_chosen_ref = atom_chosen-(design_chosen-np.array([1, 1]))
    episode_steps = 0
    done_move = False
    state, info = env_toy.reset()
    max_steps = 1000  # Set a maximum number of steps to prevent infinite loops
    env_toy.atom_pos=atom_chosen_ref
    # env_toy.goal=np.array([1, 1])

    state = env_toy.atom_pos

    step=0
    # plt.scatter(env_toy.atom_pos[0], env_toy.atom_pos[1], color='blue',s=100)
    # plt.scatter(env_toy.goal[0], env_toy.goal[1], color='green',s=100)
    path = []
    while not done_move:
        if step>max_steps:
            break
        action = agent.select_action(state,eval=True)
        next_state, reward, done_move, info = env_toy.step(action)
        
        step+=1
        episode_steps+=1


        print('i_episode:', episode_steps, 'step:', step, 'state:', state, 'action:', action, 'reward:', reward, 'done:', done_move)

        state_orgin_coor=state+(design_chosen-np.array([1, 1]))
        path.append(state_orgin_coor.tolist())
        if done_move:
            break

        state = next_state         
        
        if done_move:
            break
    return path
    

def plot_move(start, goal, agent, env_toy, figsize=(12,12), label=None, path_plan_method='rrt'):
    """
    Plot the movement of the atoms to their assigned designs

    Parameters
    ----------
    start, goal: array_like
        start and goal positions
    """
    plt.figure(figsize=figsize)
    atoms_assigned, design_assigned, anchor = align_design(start, goal)
    atoms = atoms_assigned.copy()
    designs = design_assigned.copy()
    anchors= [anchor]
    atoms, designs, costs, _, _, _ = assignment(atoms, designs)
    exist_atoms_x=[]
    exist_atoms_y=[]
    for id in range(atoms.shape[0]):
        plt.subplot(atoms.shape[0], 3, id+1)
    # id=0
        safe_radius_nm = 0.5
        mode='anchor'

        if mode=='design':
            j = np.flip(np.argsort(costs))[id]
        else:
            j = (np.argsort(cdist(atoms, np.vstack(anchors)).min(axis = 1)))[id]
        atom_chosen = atoms[j,:]
        design_chosen = designs[j,:]
        obstacle_list = []
        for i in range(atoms.shape[0]):
            if i!=j:
                obstacle_list.append((atoms[i,0],atoms[i,1], safe_radius_nm))
        for a in anchors:
            obstacle_list.append((a[0], a[1], safe_radius_nm))
        print('atom_chosen:', atom_chosen, 'design_chosen:', design_chosen, 'obstacle_list:', obstacle_list)
        plt.scatter(atom_chosen[0], atom_chosen[1], s=50, c='b')
        plt.scatter(design_chosen[0], design_chosen[1], s=50, c='r')
        plt.plot([atom_chosen[0], design_chosen[0]], [atom_chosen[1], design_chosen[1]], 'b', linewidth=2)
        plt.text(design_chosen[0], design_chosen[1], str(id), fontsize=12, color='black')
        # for obstacle in obstacle_list:
        #     plt.scatter(obstacle[0], obstacle[1], s=20, c='g')
        #     circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='g', fill=False)
        #     plt.gca().add_artist(circle)
        # plt.xlim(-4, 10)
        # plt.ylim(-4, 10)
        # plt.scatter(atoms_assigned[:,0], atoms_assigned[:,1], s=20, c='b')
        # plt.scatter(design_assigned[:,0], design_assigned[:,1], s=20, c='purple')
        # plt.scatter(anchor[0], anchor[1], s=50, c='g')
        exist_atoms_x.append(design_chosen[0])
        exist_atoms_y.append(design_chosen[1])
        atoms_assigned=np.delete(atoms_assigned, np.where((atoms_assigned==atom_chosen).all(axis=1))[0][0], axis=0)
        plt.scatter(atoms_assigned[:,0], atoms_assigned[:,1], s=100, c='grey')
        plt.scatter(anchor[0], anchor[1], s=200, c='g')
        plt.scatter(exist_atoms_x, exist_atoms_y, s=100, c='red')
        plt.scatter(design_chosen[0], design_chosen[1], s=100, c='yellow')
        # plt.plot(exist_atoms_x, exist_atoms_y, 'g--', linewidth=2)
        if path_plan_method=='rl':
            path=get_path_from_rl(atom_chosen, design_chosen, agent, env_toy)
        elif path_plan_method=='rrt':
            rrt = RRT(
                start=atom_chosen, goal=design_chosen, rand_area=[-2, 2],
                obstacle_list=obstacle_list, expand_dis= 1, path_resolution=1)

            path = rrt.planning(animation=False)
        if path is None:
            print("Path not found")
            continue
        else:
            path = np.array(path)
            plt.scatter(path[:, 0], path[:, 1], s=20, c='purple')
            plt.plot(path[:, 0], path[:, 1], color='purple', linewidth=2)
            plt.xlabel('move steps %s' % len(path))
        if label is not None:
            plt.title(label)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    