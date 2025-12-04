#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Nian Wu
# Date: 2025-11-26
# Description: Module for moving task with Nanonis controller

import sys
sys.path.append("c:\\Users\\wun2\\github\\AutoOSS_20251126")
import os
import time
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt

from AutoOSS.img_modules.img_attrib import mol_property
from AutoOSS.utils.extract_img_from_sxm import extract_scanimg_from_sxm, numpy_f4_to_grayscale
from AutoOSS.nanonisTCP.nanonisController import nanonisController
from tqdm import tqdm

class MoveModule(nanonisController):

    # Default parameters (class-level constant)
    DEFAULT_PARAMS = {
        "Move_tip_start_x": None,                      # (X_center) in nm
        "Move_tip_start_y": None,                      # (Y_center) in nm
        "Move_tip_end_x": None,                       # Reference point position x in nm during the movement
        "Move_tip_end_y": None,                       # Reference point position y in nm during the movement
        "Move_ref_point_x": None,                       # Tip position x refered to fix point in nm during the movement
        "Move_ref_point_y": None,                       # Tip position y refered to fix point in nm during the movement
        "Move_ref_dist_x": 4,                        # Tip position x distance refered to fix point in nm during the movement
        "Move_ref_dist_y": 4,                        # Tip position y distance refered to fix point in nm during the movement
        "Move_ref_dist_angle": 45,                        # Tip position angle refered to fix point in degrees during the movement
        "Move_speed": 1.2,                      # Move speed in nm/s
        "Move_scan_speed": 500,                 # Scan speed in nm/s
        "Move_bias": 1.0,                      # Sample bias in V
        "Move_current": 0.002,                    # Tunneling current setpoint in nA
        "Move_height": 0.01,                    # Scan height in nm (used in Constant Height mode)
        "Move_mode": "Constant Current",       # Scan mode: 'Constant Current' or 'Constant Height'
        "Move_from_ref": True,                      # Design move positions from reference point
        "Move_save_name": "SPM_Move_",  # Series name for saving scans
        "Move_info": "Params detail: Move_tip_start_x (y): nm, Move_tip_end_x (y): nm, Move_ref_point_x (y): nm, Move_ref_dist_x (y): nm, Move_ref_dist_angle: degrees, Move_speed: nm/s, Move_scan_speed: nm/s, Move_bias: V, Move_current: nA, Move_height: nm, Move_mode: 'Constant Current' or 'Constant Height'",  # Info string for saving scans


    }

    def __init__(self, nano_control: nanonisController, params_dict=None, *args, **kwargs):
        super().__init__()

        self.nano_control=nano_control


        # Start from default params
        
        tip_pos_x, tip_pos_y = self.nano_control.FolMe.XYPosGet(Wait_for_newest_data=True)[0]*1e9, self.nano_control.FolMe.XYPosGet(Wait_for_newest_data=True)[1]*1e9  # convert m to nm

        
        self.DEFAULT_PARAMS['Move_ref_point_x'], self.DEFAULT_PARAMS['Move_ref_point_y'] = tip_pos_x, tip_pos_y
        if self.DEFAULT_PARAMS['Move_from_ref']:
            self.DEFAULT_PARAMS['Move_tip_start_x'] = self.DEFAULT_PARAMS['Move_ref_point_x']-self.DEFAULT_PARAMS['Move_ref_dist_x']*np.cos(np.deg2rad(self.DEFAULT_PARAMS['Move_ref_dist_angle'])) 
            self.DEFAULT_PARAMS['Move_tip_start_y'] = self.DEFAULT_PARAMS['Move_ref_point_y']-self.DEFAULT_PARAMS['Move_ref_dist_y']*np.sin(np.deg2rad(self.DEFAULT_PARAMS['Move_ref_dist_angle']))
            self.DEFAULT_PARAMS['Move_tip_end_x'] = self.DEFAULT_PARAMS['Move_ref_point_x']+self.DEFAULT_PARAMS['Move_ref_dist_x']*np.cos(np.deg2rad(self.DEFAULT_PARAMS['Move_ref_dist_angle']))
            self.DEFAULT_PARAMS['Move_tip_end_y'] = self.DEFAULT_PARAMS['Move_ref_point_y']+self.DEFAULT_PARAMS['Move_ref_dist_y']*np.sin(np.deg2rad(self.DEFAULT_PARAMS['Move_ref_dist_angle']))
        self.current_params = dict(self.DEFAULT_PARAMS)


        # Override with user-provided kwargs
        self.current_params.update(kwargs)

        # Convert dict to attributes
        for key, value in self.current_params.items():
            setattr(self, key, value)


        if params_dict is not None:
            for key, value in params_dict.items():
                if key in self.current_params.keys():
                    setattr(self, key, value)


    def reset_params(self):
        # Reset to default parameters
        self.nano_control.BiasSpectr.Open()
        self.current_params = dict(self.DEFAULT_PARAMS)
        for key, value in self.current_params.items():
            setattr(self, key, value)

    def update_params(self, params_dict=None, print_params=False, **kwargs):
        # Update parameters with provided kwargs
        if params_dict is not None:
            for key, value in params_dict.items():
                if key in self.current_params.keys():
                    setattr(self, key, value)

        self.current_params.update(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.set_move_tip_positions()
        if print_params:
            self.print_params()

 

    def print_params(self):
        print("MoveModule parameters:")
        
        self.set_move_tip_positions()
       
        for key, value in self.current_params.items():
            getattr(self, key, value)
            print(f"{key}: {getattr(self, key)}")

        

    def set_move_tip_positions(self):
        # Set dissociation tip position
        if self.Move_from_ref:
            self.Move_tip_start_x = self.Move_ref_point_x - self.Move_ref_dist_x * np.cos(np.deg2rad(self.Move_ref_dist_angle)) 
            self.Move_tip_start_y = self.Move_ref_point_y - self.Move_ref_dist_y * np.sin(np.deg2rad(self.Move_ref_dist_angle))
            self.Move_tip_end_x = self.Move_ref_point_x + self.Move_ref_dist_x * np.cos(np.deg2rad(self.Move_ref_dist_angle)) 
            self.Move_tip_end_y = self.Move_ref_point_y + self.Move_ref_dist_y * np.sin(np.deg2rad(self.Move_ref_dist_angle))
        print(f"Movement tip from ({self.Move_tip_start_x}, {self.Move_tip_start_y}) nm to ({self.Move_tip_end_x}, {self.Move_tip_end_y}) nm")




    
    def perform_single_move(self):
        # Perform a single dissipation measurement
        # Move tip to dissociation position

        self.nano_control.ZController.OnOffSet(True)
        self.set_move_tip_positions()
        self.nano_control.FolMe.SpeedSet(self.Move_scan_speed*1e-9, False)
        self.nano_control.FolMe.XYPosSet(self.Move_tip_start_x*1e-9, self.Move_tip_start_y*1e-9, Wait_end_of_move=True)


        

        if self.Move_mode == 'Constant Height':
            start_z=self.nano_control.ZController.ZPosGet()          # obtain current z position, in meter
            self.nano_control.ZController.OnOffSet(False)
            self.nano_control.Bias.Set(self.Move_bias)

            z_value=start_z-self.Move_height*1e-9
            self.nano_control.ZController.ZPosSet(z_value)

        elif self.Move_mode == 'Constant Current':
            self.nano_control.ZController.OnOffSet(True)
            self.nano_control.Bias.Set(self.Move_bias)
            self.nano_control.ZController.SetpntSet(self.Move_current*1e-9)

        self.print_params()

        self.nano_control.FolMe.SpeedSet(self.Move_speed*1e-9, True)
        self.nano_control.FolMe.XYPosSet(self.Move_tip_end_x*1e-9, self.Move_tip_end_y*1e-9, Wait_end_of_move=True)


        self.nano_control.TipRec.DataSave('move_%s' % self.Move_save_name, False)
        channel_indexes, data = self.nano_control.TipRec.DataGet()

        fig, axs = plt.subplots(figsize=(5, 5))
        plt.plot([i for i in range(len(data[0]))], data[0])
        plt.title(f'Move Data {1}')
        plt.xlabel('Data Point Index')
        plt.ylabel('Data Value')

        self.nano_control.FolMe.SpeedSet(self.Move_scan_speed*1e-9, False)





