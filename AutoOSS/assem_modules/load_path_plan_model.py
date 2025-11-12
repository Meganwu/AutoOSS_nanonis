#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Author: Nian Wu
Date: 2025-07-29
Description: Load path planning model for the environment.
"""


from AutoOSS.rl_modules.sac_agent import sac_agent
from AutoOSS.rl_modules.replay_memory import HerReplayMemory
from AutoOSS.env_modules.episode_memory import Episode_Memory


import torch
from collections import namedtuple
import numpy as np  
import os

from scipy.spatial import distance

np.random.seed(2)  # reproducible

class env_toy:
    def __init__(self):
        self.state = np.array([0, 0])


    def reset(self):
        self.atom_pos=np.array([0, 0])
        self.state=self.atom_pos
        self.length=0.3
        self.thresh=0.1
        self.goal=np.array([1, 1])
        self.dist_start=distance.euclidean(self.atom_pos, self.goal)
        info={'state':self.state}
        self.img_info = info
        return self.state, info
    
    def step(self, action):
        
        self.next_state = self.atom_pos+action*self.length
        

        reward, dist=self.compute_reward(self.next_state)
        
        if dist<self.thresh:
            done_move=True
        else:
            done_move=False
        

        info={'state':self.state, 'action': action, 'reward': reward, 'next_state': self.next_state}
        self.atom_pos=self.next_state
        return self.next_state, reward, done_move, info

    def compute_reward(self, pos):
        
        dist=distance.euclidean(pos, self.goal)
        if dist<self.thresh:
            reward=1
        
        else:
            reward= -dist+self.dist_start
            
    
            
        self.dist_start=dist
     
        return reward, dist
    


def load_path_plan_model(main_path='C:\\Users\\wun2\\github\\web_AutoOSS'):



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #TODO


    # replay_size=10000 #Set memory size        #TODO
    #Set the folder name to store training data and neural network weight
    # folder_name =  'C:/LocalUserData/User-data/phys-asp-lab/nian_auto_spm/test_nian_ddpg_new_1208'
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)


    torch.device("cuda" if torch.cuda.is_available() else "cpu")


    lr= 0.0001  # default 0.0003
    gamma = 0.9     # default 0.9
    tau = 0.005       # default 0.005

    

    #Set the action space range
    ACTION_SPACE = namedtuple('ACTION_SPACE', ['high', 'low'])
    # action_space = ACTION_SPACE(high = torch.tensor([1, 1, 1, 1]), low = torch.tensor([-1, -1, 0.25, 0.0]))  # 4 D
    
    action_space = ACTION_SPACE(high = torch.tensor([1, 1]), low = torch.tensor([-1, -1]))

    
    #Initialize the soft actor-critic agent
    alpha = 1.0

    
    agent = sac_agent(num_inputs = 2, num_actions = 2, action_space = action_space, device=device, hidden_size=256, lr=lr, gamma=gamma, tau=tau, alpha=alpha)


    folder_name =  'assemble_para'

    agent.critic.load_state_dict(torch.load(os.path.join(main_path, folder_name, 'sac_critic_best.pth')))
    agent.critic_target.load_state_dict(torch.load(os.path.join(main_path, folder_name, 'sac_critic_best.pth')))
    agent.policy.load_state_dict(torch.load(os.path.join(main_path, folder_name, 'sac_policy_best.pth')))
    agent.alpha=torch.load(os.path.join(main_path, folder_name, 'sac_alpha_best.pth'))

    return agent



