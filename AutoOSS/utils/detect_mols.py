
import numpy as np

from AutoOSS.nanonisTCP.nanonisController import nanonisController
from AutoOSS.utils.extract_img_from_sxm import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.spatial.distance import cdist

import pandas as pd
import os
import io
import time
from datetime import datetime
import shutil as shulti


import sys


def gene_detect_mols_folder(folder_id: str='current_time',
        total_output_folder: str = 'all_output',
        subtask_folder: str = 'get_new_region',
        save_scan_img_large_folder: str = 'scan_img_large',
        save_scan_img_small_folder: str = 'scan_img_small',
        save_scan_data_large_folder: str = 'scan_data_large',
        save_scan_data_small_folder: str = 'scan_data_small',
        gene_current_time: bool = True,):
    """
    Generate a folder to save the detected molecules
    """
    if gene_current_time:
            current_time = datetime.datetime.now()
            folder_id = '%s_%s_%s_%s' % (current_time.month, current_time.day, current_time.hour, current_time.minute) 


    subtask_path='%s/%s_%s' % (total_output_folder, subtask_folder, folder_id)
    save_img_large='%s/%s' % (subtask_path, save_scan_img_large_folder)
    save_img_small='%s/%s' % (subtask_path, save_scan_img_small_folder)
    save_data_large='%s/%s' % (subtask_path, save_scan_data_large_folder)
    save_data_small='%s/%s' % (subtask_path, save_scan_data_small_folder)

    # check if these folders exist

    if not os.path.exists(subtask_folder):
            os.mkdir(subtask_path)
    if not os.path.exists(save_img_large):
            os.mkdir(save_img_large)
    if not os.path.exists(save_img_small):
            os.mkdir(save_img_small)
    if not os.path.exists(save_data_large):
            os.mkdir(save_data_large)
    if not os.path.exists(save_data_small):
            os.mkdir(save_data_small)

    return folder_id, subtask_path, save_img_large, save_img_small, save_data_large, save_data_small

    


def detect_mols_from_large_img(nano_control: nanonisController= None, format='nanonis', img_path: str='test_imgs_1_ZnBr2\example1.png',mol_area_limit: list =[0.2, 1.5], mol_circularity_limit: list =[0.1, 1.2], scan_mid_x_nm: float = 0, scan_mid_y_nm: float = 0, scan_size_nm: float = 10, scan_angle: float=0):
    '''Analyze the current scan image and return the molecular candidates. '''

    if format in ['png', 'jpg', 'jpeg']:
        img_path= img_path
        img_large = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
        scan_top_center_x, scan_top_center_y = scan_mid_x_nm, scan_mid_y_nm + scan_size_nm / 2  # nm, the center of upper line in a square scan
    

    elif format == 'sxm':
        img_path= img_path
        img_large, scan_mid_x, scan_mid_y, scan_len_x, scan_len_y, scan_angle=extract_scanimg_from_sxm(img_path, params_show=True)
        
        # convert m to nm
        scan_top_center_x, scan_top_center_y=scan_mid_x*1e9, (scan_mid_y+scan_len_y/2)*1e9                       # nm, the center of upper line in a square scan
        scan_size_nm = scan_len_x * 1e9  # Convert to nm


    elif format == 'nanonis':
        
        img_large=nano_control.Scan.FrameDataGrab(14, 1)[1]
        scan_mid_x, scan_mid_y, scan_len_x, scan_len_y, scan_angle = nano_control.Scan.FrameGet()   
        img_large=numpy_f4_to_grayscale(img_large)           # Convert to grayscale for mol_property analysis#
        
        assert scan_len_x== scan_len_y, "Scan length in x and y direction should be equal for square scans."

        # convert m to nm
        scan_top_center_x, scan_top_center_y=scan_mid_x*1e9, (scan_mid_y+scan_len_y/2)*1e9                       # nm, the center of upper line in a square scan
        scan_size_nm = scan_len_x * 1e9  # Convert to nm
    
    pixels=img_large.shape[0]  # Assuming img_large is a square image, pixels is the number of pixelss in one dimension

    try:

        img_large_prop=mol_property(img_large, pixels=pixels, offset_x_nm=scan_top_center_x, offset_y_nm=scan_top_center_y, len_nm=scan_size_nm)
        img_large_prop.center_points_from_contour(mol_area_limit=mol_area_limit, mol_circularity_limit=mol_circularity_limit)
        
        # mol_candidates=img_large_prop.selected_points_from_contours # nm
        mol_candidates=img_large_prop.detect_mols_center_from_contours
        print('selective ***** points:', mol_candidates)
        plt.show()

    except Exception as e:
        print(f"Error during scan processing: {e}")
        mol_candidates = []


    return mol_candidates

def save_scan_image(
        nano_control: nanonisController,
        x_nm: float = 0,
        y_nm: float = 0,
        scan_size_nm: float = 10,
        pixels: int = 128,
        scan_bias: float = -2.0,          # V
        scan_current: float = 2,          # pA
        scan_speed: float = 500,          # nm/s
        scan_line_time: float = 0.06,  # s 
        save_img_folder: str = None,
        save_data_folder: str = None,
        img_name: str = None,
        data_name: str = None,
        data_name_suffix: int = 1,
        save_img: bool = True,
        save_data: bool = True,
        nanonis_folder: str = 'c://Users//wun2//Downloads//nanonis_result',
        detect_mol: bool = False,
        mol_area_limit=[0.2, 1.5], 
        mol_circularity_limit=[0.1, 1.2]
        ) -> tuple:
        """scanning a image and save

        Args:
                env (_type_): createc environment.
                x_nm (_type_, optional): the offset_x_nm of scanning region.
                y_nm (_type_, optional): the offset_y_nm of scanning region.
                scan_len_nm (_type_, optional): the length of scanning region.
                save_img_folder (_type_, optional): the folder for saving images.
                filename (_type_, optional): the filename for saving images.

        Returns:
                _type_: _description_
        """ 
        # convert nm to m
        scan_x_center=float(f'{x_nm}e-9')
        scan_y_center=float(f'{y_nm}e-9')
        scan_size=float(f'{scan_size_nm}e-9')

        # convert pA to A
        scan_current=float(f'{scan_current}e-12')

        # convert nm/s to m/s
        scan_speed=float(f'{scan_speed}e-9')

        # Set scanning parameters

        nano_control.Bias.Set(scan_bias)
        nano_control.ZController.SetpntSet(scan_current)
        nano_control.Scan.SpeedSet(fwd_speed=scan_speed,bwd_speed=scan_speed,fwd_line_time=scan_line_time,bwd_line_time=scan_line_time)
        nano_control.Scan.FrameSet(scan_x_center, scan_y_center, scan_size, scan_size, 0)
        nano_control.Scan.BufferSet(pixels=pixels, lines=pixels)

        nano_control.FolMe.XYPosSet(scan_x_center,scan_y_center, Wait_end_of_move=True)
        time.sleep(3)  # Waits for 10 seconds

        # save nanonis output file .sxm

        if save_data:
            nano_control.Scan.PropsSet(autosave=2, series_name="scan_data_%s" % data_name)


        # Start scanning
        
        nano_control.Scan.Action('start', scan_direction="down")
        nano_control.Scan.WaitEndOfScan()

        # save scanning image

        img_large=nano_control.Scan.FrameDataGrab(14, 1)[1]
        if save_img:
            plt.imsave('%s/img_forward_%s.png' % (save_img_folder, img_name), img_large)

        if save_data:
            # modify it in terms of the name in Nanonis
            shulti.copyfile('%s/scan_data_%s%s.sxm' % (nanonis_folder, data_name, str(data_name_suffix).zfill(4)), '%s/scan_data_%s.sxm' % (save_data_folder, data_name))

        if detect_mol:
            mol_candidates=detect_mols_from_large_img(nano_control=nano_control, mol_area_limit=mol_area_limit, mol_circularity_limit=mol_circularity_limit)
            if len(mol_candidates)>0:
                print('mol_candidates:', mol_candidates)
            else:
                print('No molecules detected in the image.')



def get_state(self,
        x_nm: float = None,
        y_nm: float = None,
        scan_len_nm_large: int = 20,
        pixels_large: int = 128,
        scan_len_nm_small: float = 3.5,
        pixels_small: int = 128,
        new_scan_region: bool = False,
        check_similarity: list = None,
        candidate_mols: list = None,               # existing_mols is a list of possible individual mols based on rought detection
        real_mols: list = None,                # checked_mols is a list of mols that have been checked
        fake_mols: list = None,                    # mol_fake is a list of mols that are not real mols
        max_seek_time: int = 3,
        fix_state: bool = True,
        gene_save_folder: bool = True,
        gene_current_time: bool = False,
        folder_id: str = 'current_time',
        img_name: str = None,
        data_name: str = None,
        ):

        """
        Get the state of the environment

        Returns
        -------
        self.state: array_like
        """


        if gene_save_folder:
            if gene_current_time:
                    current_time = datetime.datetime.now()
                    folder_id = '%s_%s_%s_%s' % (current_time.month, current_time.day, current_time.hour, current_time.minute) 
            folder_id, subtask_path, save_img_large, save_img_small, save_data_large, save_data_small = gene_detect_mols_folder(folder_id=folder_id, gene_current_time= gene_current_time)


        if check_similarity is None or len(check_similarity)==0:
                check_similarity = [[x_nm, y_nm]]

        if real_mols is None:
                real_mols = []
        if fake_mols is None:
                fake_mols = []


        if self.check_similarity is None:
                self.check_similarity = check_similarity


        
        done = False
        max_time_limit=5
        seek_time=0

        if new_scan_region:
                found_mol=False
                while not found_mol:
                        self.tip_form_region()   
                        while (candidate_mols is None) or len(candidate_mols)==0:  
                                    
                                seek_time+=1
                                print('start scanning region: seek_time', seek_time, self.createc_controller.get_offset_nm())
        
                                if img_name is None:
                                        self.save_scan_image(x_nm=x_nm, y_nm=y_nm, pixels=pixels_large, scan_len_nm=scan_len_nm_large, scan_speed=1000, save_img_folder=save_img_large, save_data_folder=save_data_large, img_name=str(seek_time), data_name=str(seek_time))
                                        img_large=cv2.imread('%s/img_forward_%s.png' % (save_img_large, str(seek_time)), cv2.IMREAD_GRAYSCALE)
                                else:
                                        self.save_scan_image(x_nm=x_nm, y_nm=y_nm, pixels=pixels_large, scan_len_nm=scan_len_nm_large, scan_speed=1000, save_img_folder=save_img_large, save_data_folder=save_data_large, img_name=img_name+'_'+str(seek_time), data_name=data_name+'_'+str(seek_time))
                                        img_large=cv2.imread('%s/img_forward_%s.png' % (save_img_large, img_name+'_'+str(seek_time)), cv2.IMREAD_GRAYSCALE)
                                img_prop_large=mol_property(img_large, offset_x_nm=x_nm, offset_y_nm=y_nm, len_nm=scan_len_nm_large)
                                # select mol from points
                                detect_mols_1=img_prop_large.detect_mol_from_points()
                                # detect_mols_1=img_prop_large.detect_mol_from_points()

                                # select mol from contours
                                img_prop_large.center_points_from_contour(mol_area_limit=[1.0, 2.5])
                                detect_mols_2=img_prop_large.detect_mols_center_from_contours

                                # detect_mols=[i for i in detect_mols_1 if i in detect_mols_2]
                                
                                if len(detect_mols_1)>0 and len(detect_mols_2)>0:
                                        detect_mols_1_array=np.array(detect_mols_1)
                                        detect_mols_2_array=np.array(detect_mols_2)
                                        candidate_mols=[i for i in detect_mols_2 if np.sqrt((detect_mols_1_array[:, 0]-i[0])**2+(detect_mols_1_array[:, 1]-i[1])**2).min()<1.5]
                                                # candidate_mols=detect_mols_1
                                else:
                                        candidate_mols=[]

                                if len(candidate_mols)>0:
                                        print('candidate_mols', candidate_mols)
                                
                                else:
                                        forbid_radius=20
                                        print('start finding new region')
                                        candidate_mols=None 
                                        if seek_time>max_seek_time:
                                                if seek_time%3==0:
                                                        self.tip_form_region()
                                                x_nm, y_nm=self.get_next_region(x_nm, y_nm, detect_similarity=True, forbid_radius=forbid_radius*2, check_similarity=check_similarity)
                                        else:
                                                x_nm, y_nm=self.get_next_region(x_nm, y_nm, detect_similarity=True, check_similarity=check_similarity)
                                        check_similarity.append([x_nm, y_nm])
                                        print('looking for new region x_nm, y_nm', x_nm, y_nm)

                        candidate_mols_num=len(candidate_mols)

                        for i in range(candidate_mols_num):
                                print('mol num:', i)
                                
                                mol_pos=candidate_mols[0]
                                candidate_mols.remove(mol_pos)

                                        
                                if img_name is None:
                                        self.adjust_mol_pos(mol_pos, scan_len_nm_small=scan_len_nm_small, pixels_small=128, scan_speed=200, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name='%s_%s' % (seek_time, i), data_name='%s_%s' % (seek_time, i))                
                                        img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, '%s_%s' % (seek_time, i)), cv2.IMREAD_GRAYSCALE)
                                else:
                                        self.adjust_mol_pos(mol_pos, scan_len_nm_small=scan_len_nm_small, pixels_small=128, scan_speed=200, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name='%s_%s_%s' % (img_name, seek_time, i), data_name='%s_%s_%s' % (img_name, seek_time, i))              

                                        img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, '%s_%s_%s' % (img_name, seek_time, i)), cv2.IMREAD_GRAYSCALE)
                                img_feat=self.extract_state_feat(img_small)
                                img_cnn_detect_mol=self.detect_mol_cnn(img_small)
                                detect_messy=self.detect_messy_cnn(img_small)
                                if  img_cnn_detect_mol and self.detect_product_cnn(img_small)!='messy_mol' and self.detect_product_cnn_2(img_small)=='origin_mol' and detect_messy:
                                        img_cnn_detect_mol_2=True
                                else:
                                        img_cnn_detect_mol_2=False
                                scan_offset_nm=self.createc_controller.get_offset_nm()
                                img_prop_small=mol_property(img_small, offset_x_nm=scan_offset_nm[0], offset_y_nm=scan_offset_nm[1], len_nm=scan_len_nm_small)
                                img_prop_small.center_points_from_contour()
                                



                                

                                if len(img_prop_small.contours)>0 and len(img_prop_small.contours_max)>10 and  len(img_prop_small.detect_mols_from_contours)>0  and img_cnn_detect_mol_2:
                                        img_prop_small.contour_property()
                                        if self.add_state_time:
                                                if self.add_state_feat:
                                                        self.state=img_feat+[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]+[0.0]
                                                else:
                                                        self.state=[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]+[0.0]
                                        else:
                                                if self.add_state_feat:
                                                        self.state=img_feat+[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]
                                                else:
                                                        self.state=[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]
                                        self.tip_pos_expert_nm=[0.3*img_prop_small.ellipse_height*np.sin(img_prop_small.ellipse_angle/180*np.pi), -0.3*img_prop_small.ellipse_height*np.cos(img_prop_small.ellipse_angle/180*np.pi)]
                                        real_mols.append(mol_pos)  
                                        forbid_radius=20
                                        found_mol=True
                                        if fix_state:
                                                self.state=[1]
                                        break

                                else:
                                        fake_mols.append(mol_pos)
                                        found_mol=False
                                        print('No Molecule detected')

                        if not found_mol:
                                if seek_time>max_seek_time:

                                        if seek_time%3==0:
                                                self.tip_form_region(max_fix_pos=True)
                                        
                                        forbid_radius=20+20*(seek_time-max_seek_time)
                                        x_nm, y_nm=self.get_next_region(x_nm, y_nm, detect_similarity=True, forbid_radius=forbid_radius, check_similarity=check_similarity)
                                else:
                                        x_nm, y_nm=self.get_next_region(x_nm, y_nm, detect_similarity=True, check_similarity=check_similarity)
                                        check_similarity.append([x_nm, y_nm])
                                        print('looking for new region x_nm, y_nm', x_nm, y_nm)


                                


        else:
                self.save_scan_image(x_nm=x_nm, y_nm=y_nm, pixels=pixels_small, scan_len_nm=scan_len_nm_small, scan_speed=200, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name=img_name, data_name=data_name)
                img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, img_name), cv2.IMREAD_GRAYSCALE)
                img_feat=self.extract_state_feat(img_small)

                img_prop_small=mol_property(img_small, offset_x_nm=x_nm, offset_y_nm=y_nm, len_nm=scan_len_nm_small)
                img_prop_small.center_points_from_contour()
                img_prop_small.contour_property()
                # img_cnn_detect_mol=self.detect_mol_cnn(img_small)

                try:
                        if len(img_prop_small.contours)>0 and len(img_prop_small.contours_max)>5 and len(img_prop_small.detect_mols_from_contours)>0:
                                if self.add_state_time:
                                        if self.add_state_feat:
                                                self.state=img_feat+[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]+[0.0]
                                        else:
                                                self.state=[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]+[0.0]
                                else:
                                        if self.add_state_feat:
                                                self.state=img_feat+[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]
                                        else:
                                                self.state=[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]
                                self.tip_pos_expert_nm=[0.3*img_prop_small.ellipse_height*np.sin(img_prop_small.ellipse_angle/180*np.pi), -0.3*img_prop_small.ellipse_height*np.cos(img_prop_small.ellipse_angle/180*np.pi)]
                                if fix_state:
                                        self.state=[1]
                except:
                        print('No Molecule detected')

        return img_prop_small.ellipse_x, img_prop_small.ellipse_y, check_similarity, candidate_mols




def reset_max_radius_cellsize(self, cellsize: float=10, max_radius: float=300) -> None:
        """
        Reset the max_radius and cellsize
        """
        self.max_radius = max_radius
        self.cellsize = cellsize
        self.num_cell = int(self.max_radius/self.cellsize)

def get_next_region(self, tip_x, tip_y, scan_ref_x_center=None, scan_ref_y_center=None, forbid_radius=20, check_simi_forbid_radius=25, move_upper_limit=400, approach_limit=[-300, 300, -300, 300], spiralreg=1.0, mn=100, detect_similarity=True, check_similarity=None):
        """
        Get the next good closest tip position
        """
        if scan_ref_x_center is None:
                scan_ref_x_center=self.scan_ref_x_center
        if scan_ref_y_center is None:
                scan_ref_y_center=self.scan_ref_y_center
        if forbid_radius is None:
                forbid_radius=self.forbid_radius
        if check_simi_forbid_radius is None:
                check_simi_forbid_radius=self.check_simi_forbid_radius
        if approach_limit is None:
                approach_limit=self.approach_limit
        if move_upper_limit is None:
                move_upper_limit=self.move_upper_limit

        move_limit=move_upper_limit/float(self.cellsize)
        found=False
        self.get_approach_area()
        self.forbidden_area(forbid_radius=forbid_radius)
        for i in range(-self.num_cell, self.num_cell+1):
                for j in range(-self.num_cell,self.num_cell+1):
                        # plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='orange',s=1)
                        if self.mask[i+self.num_cell,j+self.num_cell] == True:
                                
                        # plt.gca().add_patch(patches.Rectangle((x+i*self.cellsize-4,  y+j*self.cellsize), 8, 8, fill=False, edgecolor='grey', lw=2))
                                # plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='g',s=1)
                                continue

                        dist_euclidian = np.sqrt(float((i*self.cellsize)**2)+ float((j*self.cellsize)**2)) #Euclidian distance
                        if (dist_euclidian>self.move_upper_limit):
                                # plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='blue')
                                continue

                        if tip_x+i*self.cellsize<approach_limit[0] or tip_x+i*self.cellsize>approach_limit[1] or tip_y+j*self.cellsize<approach_limit[2] or tip_y+j*self.cellsize>approach_limit[3]:
                        # new_x_all.append(ref_x+i*radius*1.5)
                        # new_y_all.append(ref_y+j*radius*1.5)
                                plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='blue', s=1)
                                continue

                        # plt.scatter(x+i*self.cellsize,y+j*self.cellsize,color='yellow')

                        #dist_manhattan = abs(i*self.cellsize)+abs(j*self.cellsize) #Manhattan distance

                        dist_manhattan = max(abs(i*self.cellsize), abs(j*self.cellsize))  #Manhattan distance
                                        
                        dist=(spiralreg*dist_euclidian+dist_manhattan)

                        if detect_similarity:
                                if check_similarity is None:
                                        check_similarity = [[tip_x, tip_y]]
                                # print('check_similarity:', check_similarity)
                                check_similarity_array=np.array(check_similarity)-np.array([tip_x+i*self.cellsize, tip_y+j*self.cellsize])
                                # print(check_similarity_array)
                                simi_points_dist=np.array([np.sqrt(check_similarity_array[k][0]**2+check_similarity_array[k][1]**2) for k in range(len(check_similarity_array))]).min()

                        else:
                                simi_points_dist=1000000

                        # print('ssss', similarity_dist, sim_forbiden_radius)
                        # print('dist', dist, move_limit)
        
                        if simi_points_dist > check_simi_forbid_radius:
                                if dist < move_limit  or (not found):
                                        found=True
                                        move_limit=dist
                                        tip_x_move=i
                                        tip_y_move=j
                                        plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='y')
                                elif dist==move_limit:
                                        if np.sqrt(float((tip_x+i*self.cellsize-scan_ref_x_center)**2)+float((tip_y+j*self.cellsize-scan_ref_y_center)**2))<np.sqrt(float((tip_x+tip_x_move*self.cellsize-scan_ref_x_center)**2)+ float((tip_y+tip_y_move*self.cellsize-scan_ref_y_center)**2)):
                                                found=True
                                                move_limit=dist
                                                tip_x_move=i
                                                tip_y_move=j
                                                plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='y')
                        # print(i, j, move_limit, dist_manhattan, dist_euclidian, dist)

        try:
                plt.scatter(tip_x+tip_x_move*self.cellsize,tip_y+tip_y_move*self.cellsize,color='r')            
                return tip_x+tip_x_move*self.cellsize, tip_y+tip_y_move*self.cellsize

        except:
                print('No good cloest point found, return the previous tip position')

                tip_x=approach_limit[0]+100+450*np.random.random()
                tip_y=approach_limit[2]+100+450*np.random.random()

                return tip_x, tip_y
        



def get_approach_area(self):
        """
        Get the approach area
        """
        print("starting new approach area...")
        self.mask = np.zeros((2*self.num_cell+1,2*self.num_cell+1),dtype=np.bool_)

        
def forbidden_area(self, forbid_radius: float = 100) -> tuple:
        """
        Check if the coordinates x, y is in the forbidden area

        Parameters
        ----------
        forbiden_r: float
        forbidden area radius in nm

        Return
        ------
        mask: array_like
        whether the coordinates is in the forbidden area
        """
        for i in range(-self.num_cell, self.num_cell+1):
                for j in range(-self.num_cell, self.num_cell+1):
                        if self.mask[i+self.num_cell, j+self.num_cell] == True:
                                continue
                        dist=np.sqrt((i)**2+(j)**2) #Euclidian distance
                        max_dist=forbid_radius/self.cellsize
                        if dist<max_dist:
                                self.mask[i+self.num_cell, j+self.num_cell] = True
        np.save("mask.npy", self.mask)
