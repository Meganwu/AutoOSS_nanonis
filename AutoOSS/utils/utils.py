
import numpy as np

from AutoOSS.nanonisTCP.nanonisController import nanonisController
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.spatial.distance import cdist

import pandas as pd
import os
import io
import time


import sys
# sys.path.append("z:\\asp\\STM_AUTO_OSS\\AUTOOSS_NANONIS\\utils") 
# sys.path.append("z:\\asp\\STM_AUTO_OSS\\AUTOOSS_NANONIS") 
# from AutoOSS.REACTRL.env_modules.img_attrib import *


def scan_approach_area(nano_control: nanonisController, x_lim: list =[-300, 300], y_lim: list =[-300, 300], scan_size: float = 200, scan_degree: float=0, save_path='save_scan_imgs_20250626'):
    ''' unit: nm'''

    # convert nm to m
    x_lim=np.array(x_lim)*1e-9
    y_lim=np.array(y_lim)*1e-9
    scan_size=float(f'{scan_size}e-9')

    num_col=int((x_lim[-1]-x_lim[0])/scan_size)
    num_row=int((y_lim[-1]-y_lim[0])/scan_size)

    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

    for i in range(num_col):
        for j in range(num_row):  
            if i%2==1:
                nano_control.FolMe.XYPosSet(x_lim[0]+(i-1/2)*scan_size, y_lim[1]-(num_row-1-j-1/2)*scan_size, Wait_end_of_move=True)
                nano_control.Scan.FrameSet(x_lim[0]+i*scan_size, y_lim[1]-(num_row-1-j)*scan_size, scan_size, scan_size, scan_degree)

            else:
                nano_control.FolMe.XYPosSet(x_lim[0]+(i-1/2)*scan_size, y_lim[1]-(j-1/2)*scan_size, Wait_end_of_move=True)
                nano_control.Scan.FrameSet(x_lim[0]+i*scan_size, y_lim[1]-j*scan_size, scan_size, scan_size, scan_degree)
                
        
            time.sleep(2)  # Waits for 10 seconds

            nano_control.Scan.Action('start', scan_direction="down")
            nano_control.Scan.WaitEndOfScan()
            img=nano_control.Scan.FrameDataGrab(14, 0)[1]
            plt.imsave('%s/img_%s_%s.png' % (save_path, i, j), img)



def tip_form_region(nano_control: nanonisController, tip_form_z_range: list = [-0.9, -1.2], tip_form_len_nm: float = 30, tip_form_region_candidates: list = [[-32, 132], [-32, 19], [-32, -80], [-32, -215], [-32, -300]]):
    ''' unit: nm'''  
    
    print('start tip forming now')

    # tip_form_region_candidates=np.array([[-170, 180.0], [150, 180.0], [-150, -180.0], [150.0, -180]])

    tip_form_region_candidates=np.array(tip_form_region_candidates)
    scan_mid_x, scan_mid_y, scan_len_x, scan_len_y, scan_angle = nano_control.Scan.FrameGet()
    scan_midtop_x, scan_midtop_y=scan_mid_x*1e9, (scan_mid_y+scan_len_y/2)*1e9  

    tip_current_pos=np.array([scan_midtop_x, scan_midtop_y]).reshape(1,-1)
    select_region=np.argmin(cdist(tip_form_region_candidates, tip_current_pos))

    tip_form_region_final=tip_form_region_candidates[np.argmin(cdist(tip_form_region_candidates, tip_current_pos))]
    tip_form_ref_x=tip_form_region_final[0]
    tip_form_ref_y=tip_form_region_final[1]
    print('Tip forming region:', tip_form_ref_x, tip_form_ref_y)

    # tip_form_x=tip_form_ref_x-tip_form_len_nm/2+tip_form_len_nm*np.random.rand()
    tip_form_x=tip_form_ref_x+tip_form_len_nm*np.random.rand()
    tip_form_y=tip_form_ref_y+tip_form_len_nm*np.random.rand()

    tip_form_x, tip_form_y=float(f'{tip_form_x}e-9'), float(f'{tip_form_y}e-9')

    upper_limit=tip_form_z_range[1]
    lower_limit=tip_form_z_range[0]
    tip_form_z=lower_limit+np.random.rand()*(upper_limit-lower_limit)              
    tip_form_z=float(f'{tip_form_z}e-9')


    
    nano_control.FolMe.XYPosSet(tip_form_x, tip_form_y, Wait_end_of_move=True)
    nano_control.TipShaper.PropsSet(Tip_lift=tip_form_z, Bias_lift=1, Restore_feedback=1)

    Switch_off_delay,Change_bias,Bias,Tip_lift,Lift_time_1,Bias_lift,Bias_settling_time,Lift_height,Lift_time_2,End_wait_time,Restore_feedback = nano_control.TipShaper.PropsGet()


    # print("Tip Lift: %f m" % Tip_lift)
    # print("Lift_time_1: %f s" % Lift_time_1)
    # print("Bias_lift: %f V" % Bias_lift)
    # print("Bias settling time: %f s" % Bias_settling_time)
    # print("Lift_height: %f m" % Lift_height)

    nano_control.TipShaper.Start(wait_until_finished=1,timeout=30)



def detect_move_from_current(file_path):
    # file_path=r'\\LT-AFM3\Data\250623\Lateral_Manipulation003_%s.dat' % str(i).zfill(3)

    # Read entire file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find [DATA] marker
    for i, line in enumerate(lines):
        if line.strip() == '[DATA]':
            header_index = i + 1
            break

    # Get header line and clean it
    raw_header = lines[header_index].strip()
    columns = [col.strip() for col in raw_header.split('\t') if col.strip()]
    if len(columns) < 2:  # Possibly space-separated instead
        columns = [col.strip() for col in raw_header.split('  ') if col.strip()]

    # Join the data part and read into DataFrame
    data = ''.join(lines[header_index + 1:])
    df = pd.read_csv(io.StringIO(data), delim_whitespace=True, header=None)
    df.columns = columns

    diff_max_mean = df['Current (A)'].max()-df['Current (A)'].mean()
    diff_max_min = df['Current (A)'].max()-df['Current (A)'].min()

    if diff_max_mean > 2 or (diff_max_min > 2):
        print(f"Warning: Large variation in Current (A) detected in {file_path}")
        print(f"Max - Mean: {diff_max_mean}, Max - Min: {diff_max_min}")
        return True
    else:
        print(f"No significant variation in Current (A) detected in {file_path}")
        return False

        