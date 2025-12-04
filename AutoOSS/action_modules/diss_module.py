#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Nian Wu
# Date: 2025-12-03
# Description: Module for dissociation task with Nanonis controller

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




class DissModule(nanonisController):

    # Default parameters (class-level constant)
    DEFAULT_PARAMS = {
        "Diss_tip_x": None,                      # (X_center) in nm
        "Diss_tip_y": None,                      # (Y_center) in nm
        "Diss_ref_point_x": None,                       # Reference point position x in nm during the dissociation
        "Diss_ref_point_y": None,                       # Reference point position y in nm during the dissociation
        "Diss_ref_dist_x": 0,                       # Tip position x refered to fix point in nm during the dissociation
        "Diss_ref_dist_y": 0,                       # Tip position y refered to fix point in nm during the dissociation
        "Diss_ref_dist_angle": 0,                        # Tip position angle refered to fix point in degrees during the dissociation
        "Diss_bias": 1.0,                      # Sample bias in V
        "Diss_current": 0.002,                    # Tunneling current setpoint in nA
        "Diss_height": 0.01,                    # Scan height in nm (used in Constant Height mode)
        "Diss_mode": "Constant Current",       # Scan mode: 'Constant Current' or 'Constant Height'
        "Diss_from_ref": True,                      # Design dissociation tip positions from reference point
        "Diss_sweep_mode": 'MLS',                     # 'Linear' or 'MLS'(Multi Segment)
        "Diss_save_name": "SPM_Diss_",  # Series name for saving scans
        "Diss_info": "Params detail: Diss_tip_x (y): nm, Diss_ref_point_x (y): nm, Diss_ref_dist_x (y): nm, Diss_ref_dist_angle: degrees, Diss_bias: V, Diss_current: nA, Diss_height: nm, Diss_mode: Constant Current or Constant Height, Diss_sweep_mode: Linear or Multi Segment",  # Series name for saving scans


    }

    def __init__(self, nano_control: nanonisController, params_dict=None, *args, **kwargs):
        super().__init__()

        self.nano_control=nano_control

        # Initialize BiasSpectr module

        self.nano_control.BiasSpectr.Open()


        # Start from default params
        
        tip_pos_x, tip_pos_y = self.nano_control.FolMe.XYPosGet(Wait_for_newest_data=True)[0]*1e9, self.nano_control.FolMe.XYPosGet(Wait_for_newest_data=True)[1]*1e9

        self.DEFAULT_PARAMS['Diss_tip_x'], self.DEFAULT_PARAMS['Diss_tip_y'] = tip_pos_x, tip_pos_y
        self.DEFAULT_PARAMS['Diss_ref_point_x'], self.DEFAULT_PARAMS['Diss_ref_point_y'] = tip_pos_x, tip_pos_y
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

        self.set_diss_tip_position()


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

        self.set_diss_tip_position()
        if print_params:
            self.print_params()

 

    def print_params(self):
        print("DissModule parameters:")
        for key, value in self.current_params.items():
            getattr(self, key, value)
            print(f"{key}: {getattr(self, key)}")

    def set_diss_tip_position(self):
        # Set dissociation tip position
        if self.Diss_from_ref:
            self.Diss_tip_x = self.Diss_ref_point_x + self.Diss_ref_dist_x * np.cos(np.deg2rad(self.Diss_ref_dist_angle)) 
            self.Diss_tip_y = self.Diss_ref_point_y + self.Diss_ref_dist_y * np.sin(np.deg2rad(self.Diss_ref_dist_angle))
        print(f"Dissociation tip position set to: ({self.Diss_tip_x}, {self.Diss_tip_y}) nm")
    

    def set_diss_biasspectr(self, diss_start=1.0, diss_end=1.0,initial_settling_time = [1.0,1.0,1.0,1.0,1.0], settling_time = [50e-6,50e-6,50e-6,50e-6,50e-6], integration_time = [50e-6,200e-6,200e-6,200e-6,50e-6], steps = [64,20,128,20,64], lockin_run=[0,0,0,0,0]):
        # Set BiasSpectr parameters based on current settings
        if self.Diss_sweep_mode == 'Linear':
            self.nano_control.BiasSpectr.MLSModeSet(sweep_mode="linear")
            self.nano_control.BiasSpectr.LimitsSet(start_value=diss_start,end_value=diss_end)
        elif self.Diss_sweep_mode == 'MLS':
            self.nano_control.BiasSpectr.MLSModeSet(sweep_mode="MLS")
            bias_max_select= self.Diss_bias
            bias_start = [diss_start,diss_start, bias_max_select, bias_max_select, diss_end]
            bias_end   = [diss_start, bias_max_select, bias_max_select, diss_end, diss_end]

            self.nano_control.BiasSpectr.MLSValsSet(bias_start,bias_end,initial_settling_time,settling_time,integration_time,steps,lockin_run)
 



    def perform_single_diss(self):
        # Perform a single dissipation measurement
        # Move tip to dissociation position
        self.set_diss_tip_position()
        self.nano_control.FolMe.XYPosSet(self.Diss_tip_x*1e-9, self.Diss_tip_y*1e-9, Wait_end_of_move=True)


        if self.Diss_mode == 'Constant Height':
            start_z=self.nano_control.ZController.ZPosGet()          # obtain current z position, in meter
            self.nano_control.ZController.OnOffSet(False)
            self.set_diss_biasspectr()
            z_value=start_z-self.Diss_height*1e-9
            self.nano_control.ZController.ZPosSet(z_value)
            self.nano_control.BiasSpectr.AdvPropsSet(z_controller_hold=1) # hold Z controller during the bias spectroscopy

        elif self.Diss_mode == 'Constant Current':
            self.nano_control.ZController.OnOffSet(True)
            self.set_diss_biasspectr()
            self.nano_control.ZController.SetpntSet(self.Diss_current*1e-9)
            self.nano_control.BiasSpectr.AdvPropsSet(z_controller_hold=2)   # not hold Z controller during the bias spectroscopy

        #self.print_params()

        # Select channels to save: Bias(V), Current(A), Z(m)
        self.nano_control.BiasSpectr.ChsSet([0, 8, 14], mode="set")


        vert_data=self.nano_control.BiasSpectr.Start(get_data=1,save_base_name="diss_%s" % (self.Diss_save_name))

        if self.Diss_mode == 'Constant Height':

            plt.plot(vert_data['data_dict']['Bias calc (V)'], vert_data['data_dict']['Current (A)'])
            plt.ylabel('Current (A)')
        elif self.Diss_mode == 'Constant Current':
            plt.plot(vert_data['data_dict']['Bias calc (V)'], vert_data['data_dict']['Z (m)']*1e9)
            plt.ylabel('Z (nm)')
        plt.xlabel('Bias (V)')
        plt.title('Dissipation Spectroscopy Result')