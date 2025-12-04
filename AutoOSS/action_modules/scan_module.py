#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Nian Wu
# Date: 2025-12-03
# Description: Module for scanning task with Nanonis controller

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




class ScanModule(nanonisController):

    # Default parameters (class-level constant)
    DEFAULT_PARAMS = {
        "Scan_area": [-30, 30, -30, 30],         # (X_start, X_end, Y_start, Y_end) in nm
        "Scan_center_x": 0,                      # (X_center) in nm
        "Scan_center_y": 0,                      # (Y_center) in nm
        "Scan_size_w": 20,                       # Scan image size in nm
        "Scan_size_h": 20,                       # Scan image size in nm
        "Scan_angle": 0,                        # Scan angle in degrees
        "Scan_pixels": 256,                     # Number of pixels in scan image
        "Scan_speed": 500,                      # Scan speed in nm/s
        "Scan_bias": 1.0,                      # Sample bias in V
        "Scan_current": 0.002,                    # Tunneling current setpoint in nA
        "Scan_height": 0.01,                    # Scan height in nm (used in Constant Height mode)
        "Scan_mode": "Constant Current",       # Scan mode: 'Constant Current' or 'Constant Height'
        "Scan_direction": "down",              # Scan direction: 'up' or 'down'
        "Scan_save_name": "SPM_before_",  # Series name for saving scans
    }

    def __init__(self, nano_control: nanonisController, params_dict=None, **kwargs):
        super().__init__()

        self.nano_control=nano_control

        # Start from default params
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

            # Feedback on or off based on scan mode
        if self.Scan_mode == 'Constant Height':
            self.nano_control.ZController.OnOffSet(False)
        elif self.Scan_mode == 'Constant Current':
            self.nano_control.ZController.OnOffSet(True)

        if print_params:
            self.print_params()

 

    def print_params(self):
        print("ScanModule parameters:")
        for key, value in self.current_params.items():
            getattr(self, key, value)
            print(f"{key}: {getattr(self, key)}")

    def perform_single_scan(self):

        self.nano_control.ZController.OnOffSet(True)

        # Set scan parameters (bias, current, speed)
        self.nano_control.Bias.Set(self.Scan_bias)
        self.nano_control.ZController.SetpntSet(self.Scan_current*1e-9)  # Convert nA to A
        self.nano_control.Scan.SpeedSet(fwd_speed=self.Scan_speed*1e-9,
                           bwd_speed=self.Scan_speed*1e-9)  # Convert nm/s to m/s
                
        # Set scan frame and move to start position
        self.nano_control.Scan.FrameSet(self.Scan_center_x*1e-9, self.Scan_center_y*1e-9,
                           self.Scan_size_w*1e-9, self.Scan_size_h*1e-9, self.Scan_angle)
        self.nano_control.FolMe.XYPosSet((self.Scan_center_x - self.Scan_size_w/2)*1e-9,
                            (self.Scan_center_y + self.Scan_size_h/2)*1e-9, Wait_end_of_move=True)

        print("Waiting for 1 seconds...")
        time.sleep(1)  # Waits for 1 seconds to make the tip stable

        # Feedback on or off based on scan mode
        if self.Scan_mode == 'Constant Height':
            self.nano_control.ZController.OnOffSet(False)
        elif self.Scan_mode == 'Constant Current':
            self.nano_control.ZController.OnOffSet(True)



        # Start scan
        self.nano_control.Scan.PropsSet(continuous_scan=0, bouncy_scan=0,
                           autosave=1, series_name=self.Scan_save_name, comment="")
        self.nano_control.Scan.Action('start', scan_direction=self.Scan_direction)
        self.nano_control.Scan.WaitEndOfScan()
        print(f"Scan completed and saved with the name {self.Scan_save_name}.")
        self.nano_control.ZController.OnOffSet(True)

    def get_scan_img(self, channel=0):
        # Analyze scanning results
        if self.Scan_mode == 'Constant Current':
            print("Show height z image under constant current mode")
            channel = 14  # Height channel
            img = self.nano_control.Scan.FrameDataGrab(channel, 0)[1]
            plt.imshow(img)
        elif self.Scan_mode == 'Constant Height':
            print("Show current image under constant height mode")
            img = self.nano_control.Scan.FrameDataGrab(channel, 0)[1]
            plt.imshow(img)
        return img
    
    def detect_molecules(self, channel=14,mol_area_limit=[1.3, 3.5]):
        '''dectect molecule based on size or distance in a scanning image'''
        img_large = self.get_scan_img(channel=channel)
        img_large = numpy_f4_to_grayscale(img_large)
        img_large_prop=mol_property(img_large, pixels=256, offset_x_nm= self.Scan_center_x, offset_y_nm=(self.Scan_center_y + self.Scan_size_h/2), len_nm=self.Scan_size_w*1e-9)
        # img_prop.detect_contours()
        img_large_prop.center_points_from_contour(mol_area_limit=mol_area_limit)
        
        # mol_candidates=img_large_prop.selected_points_from_contours # nm
        mol_candidates=img_large_prop.detect_mols_center_from_contours
        print('selective ***** points:', mol_candidates)
        plt.show()

        return mol_candidates
    
    def perform_multi_scan(self, plot_scan_points=True):
        # seperate the approach area into small scanning regions
        num_grid_x=int((self.Scan_area[1]-self.Scan_area[0])/self.Scan_size_w)
        num_grid_y=int((self.Scan_area[3]-self.Scan_area[2])/self.Scan_size_h)




        for i in tqdm(range(num_grid_x)):
            for j in range(num_grid_y):
                # when scanning from top to bottom, even rows scan downwards, odd rows scan upwards, it can reduce the moving distance of tip
                if i%2==0:
                    self.Scan_center_x=self.Scan_area[0]+self.Scan_size_w/2+i*self.Scan_size_w          # center x position
                    self.Scan_center_y=self.Scan_area[3]-self.Scan_size_h/2-j*self.Scan_size_h          # center y position
                else:
                    self.Scan_center_x=self.Scan_area[0]+self.Scan_size_w/2+i*self.Scan_size_w          # center x position
                    self.Scan_center_y=self.Scan_area[2]+self.Scan_size_h/2+j*self.Scan_size_h          # center y position
                print('scanning regions', i, j, self.Scan_center_x, self.Scan_center_y, self.Scan_size_w, self.Scan_size_h)
                self.update_params(Scan_center_x=self.Scan_center_x, Scan_center_y=self.Scan_center_y)
                if plot_scan_points:
                    plt.scatter(self.Scan_center_x, self.Scan_center_y, c='r')


                self.perform_single_scan()

   
        if plot_scan_points:
            plt.xlim(self.Scan_area[0], self.Scan_area[1])
            plt.ylim(self.Scan_area[2], self.Scan_area[3])