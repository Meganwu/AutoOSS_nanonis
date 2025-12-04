import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist


class mol_property:

    def __init__(self, img, pixels=None, offset_x_nm=0, offset_y_nm=0, len_nm=None) -> None:
        '''Img format: img=cv2.imread(img_path); img = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)'''
        if pixels is None:
            pixels = img.shape[0]
        if len_nm is None:
            len_nm = img.shape[0]

        self.pixels = pixels
        self.offset_x_nm = offset_x_nm
        self.offset_y_nm = offset_y_nm
        self.len_nm = len_nm
        self.unit_nm = self.len_nm/self.pixels
        self.img = img

    def gradient_2d(self, img):
        # Calculate derivatives (differences)  # Like gray
        img=self.remove_line(img)

        denoised = cv2.fastNlMeansDenoising(img, h=10)
        dx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(dx**2 + dy**2)

        magnitude_f32 = magnitude.astype(np.float32)

        # Optional: Normalize magnitude to 0-255 and convert to uint8 for thresholding
        magnitude_norm = cv2.normalize(magnitude_f32, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_uint8 = magnitude_norm.astype(np.uint8)
            
        return magnitude_uint8 
    

    def remove_line(self, img, method='z_score', z_thres_value=3.0, perc_thres_value=99.5):
        ''' Automatically identify rows with anomalously high intensity and fix them, method default: 'z_score' ('percentile' as a option)'''

        # Compute row-wise mean
        row_means = img.mean(axis=1)

        # Detect anomalous rows using z-score ===
    
        if method=='percentile':
            threshold = np.percentile(row_means, perc_thres_value)
            anomaly_rows = np.where(row_means > threshold)[0]
        elif method=='z_score':
            z_scores = (row_means - row_means.mean()) / row_means.std()
            threshold = z_thres_value  # Z-score threshold, tune as needed
            anomaly_rows = np.where(z_scores > threshold)[0]
        print(f"Anomalous rows: {anomaly_rows}")

        # Inpainting or interpolation ===
        img_fixed = img.copy()
        for r in anomaly_rows:
            if 1 <= r < img.shape[0] - 1:
                img_fixed[r] = ((img[r-1].astype(np.int32) + img[r+1].astype(np.int32)) // 2).astype(np.uint8)

        return img_fixed
    
    def detect_contours(self, method='otsu', thres_global=[50, 255], thres_otsu_blur_kernel=5, BlockSize=31, C=-5, gradient=True):
        if gradient:
            self.gradient_img=self.gradient_2d(self.img)

            # blurred = cv2.bilateralFilter(self.gradient_img, d=9, sigmaColor=75, sigmaSpace=75)
            blurred = cv2.GaussianBlur(self.gradient_img, (5, 5), 1.5)
            self.thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=BlockSize, C=C)

        else:
            if method == 'otsu':
                self.ret,self.thresh = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            elif method == 'global':
                self.ret,self.thresh = cv2.threshold(self.img, thres_global[0], thres_global[1], cv2.THRESH_BINARY)
            elif method == 'otsu_gaussian':
                blur = cv2.GaussianBlur(self.img, (thres_otsu_blur_kernel, thres_otsu_blur_kernel), 0)
                self.ret,self.thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # self.contours, _ = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.contours, _ = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours_points_num = [len(cnt) for cnt in self.contours]
        self.contours_max = self.contours[np.argmax(self.contours_points_num)]
    
    # def detect_contours(self, method='otsu', thres_global=[50, 255], thres_otsu_blur_kernel=5, gradient=True):
    #     if gradient:
    #         self.gradient_img=self.gradient_2d(self.img)
    #         self.ret,self.thresh = cv2.threshold(self.gradient_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     else:
    #         if method == 'otsu':
    #             self.ret,self.thresh = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #         elif method == 'global':
    #             self.ret,self.thresh = cv2.threshold(self.img, thres_global[0], thres_global[1], cv2.THRESH_BINARY)
    #         elif method == 'otsu_gaussian':
    #             blur = cv2.GaussianBlur(self.img, (thres_otsu_blur_kernel, thres_otsu_blur_kernel), 0)
    #             self.ret,self.thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     self.contours, _ = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     self.contours_points_num = [len(cnt) for cnt in self.contours]
    #     self.contours_max = self.contours[np.argmax(self.contours_points_num)]

    # def detect_contours(self, method='otsu', thres_global=[50, 255], thres_otsu_blur_kernel=5):
    #     if method == 'otsu':
    #         self.ret,self.thresh = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     elif method == 'global':
    #         self.ret,self.thresh = cv2.threshold(self.img, thres_global[0], thres_global[1], cv2.THRESH_BINARY)
    #     elif method == 'otsu_gaussian':
    #         blur = cv2.GaussianBlur(self.img, (thres_otsu_blur_kernel, thres_otsu_blur_kernel), 0)
    #         self.ret,self.thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     self.contours, _ = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     self.contours_points_num = [len(cnt) for cnt in self.contours]
    #     self.contours_max = self.contours[np.argmax(self.contours_points_num)]


    def contour_property(self, cnt=None, round_digit=2):
        
        if cnt is None:
            self.detect_contours()
            cnt = self.contours_max
        self.cnt = cnt
        self.area = cv2.contourArea(cnt)*self.unit_nm*self.unit_nm
        self.perimeter = cv2.arcLength(cnt, True)*self.unit_nm
        self.convexity = cv2.isContourConvex(cnt)
        # approximate as a rectangule
        self.rect_x, self.rect_y, self.rect_w, self.rect_h = cv2.boundingRect(cnt)
        self.rect_x, self.rect_y, self.rect_w, self.rect_h = round(self.rect_x*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(-self.rect_y*self.unit_nm+self.offset_y_nm, round_digit), round(self.rect_w*self.unit_nm, round_digit), round(self.rect_h*self.unit_nm, round_digit)

        # approximate as a minimum area rectangle
        self.rect_min = cv2.minAreaRect(cnt)
        self.rect_min_x, self.rect_min_y = round(self.rect_min[0][0]*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(-self.rect_min[0][1]*self.unit_nm+self.offset_y_nm, round_digit)
        self.rect_min_w, self.rect_min_h = round(self.rect_min[1][0]*self.unit_nm, round_digit), round(self.rect_min[1][1]*self.unit_nm, round_digit)
        self.rect_min_angle = round(self.rect_min[2], round_digit)

        # approximate as a circle
        self.circle = cv2.minEnclosingCircle(cnt)
        (self.circle_x, self.circle_y), self.circle_radius = self.circle
        self.circle_x, self.circle_y, self.circle_radius = round(self.circle_x*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(-self.circle_y*self.unit_nm+self.offset_y_nm, round_digit), round(self.circle_radius*self.unit_nm, round_digit)

        # approximate as a ellipse
        self.ellipse = cv2.fitEllipse(cnt) # (center_x, center_y), (width, height), angle
        self.ellipse_x, self.ellipse_y = round(self.ellipse[0][0]*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(-self.ellipse[0][1]*self.unit_nm+self.offset_y_nm, round_digit)
        self.ellipse_width, self.ellipse_height = round(self.ellipse[1][0]*self.unit_nm, round_digit), round(self.ellipse[1][1]*self.unit_nm, round_digit)
        self.ellipse_angle = round(self.ellipse[2], round_digit)

    def center_points_from_contour(self, dist_thres=1.8, mol_area_limit=[1.5, 2.8], mol_circularity_limit=[0.1, 1.2], round_digit=2, plot_graph=True, BlockSize=31, C=-5, gradient=False):
        '''select contours based on the area and circularity'''
        selected_points_from_contours=[]
        detect_mols_from_contours=[]
        detect_mols_center_from_contours=[]
        self.detect_contours(BlockSize=BlockSize, C=C, gradient=gradient)  # Detect contours from the image
        for i in range(len(self.contours)):
            if self.contours_points_num[i]>10:
                cnt = self.contours[i]
                ellipse = cv2.fitEllipse(cnt) # (center_x, center_y), (width, height), angle
                ellipse_x, ellipse_y = round(ellipse[0][0]*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(-ellipse[0][1]*self.unit_nm+self.offset_y_nm, round_digit)
                ellipse_width, ellipse_height = round(ellipse[1][0]*self.unit_nm, round_digit), round(ellipse[1][1]*self.unit_nm, round_digit)
                ellipse_angle = round(ellipse[2], round_digit)
                selected_points_from_contours.append([ellipse_x, ellipse_y])
                area = cv2.contourArea(cnt)*self.unit_nm*self.unit_nm
                perimeter = cv2.arcLength(cnt, True)*self.unit_nm
                circularity = 4 * np.pi * area / (perimeter**2)
                if plot_graph:
                    plt.scatter(ellipse_x, ellipse_y, s=10, c='b')
                if area>mol_area_limit[0] and area<mol_area_limit[1] and circularity>mol_circularity_limit[0] and circularity<mol_circularity_limit[1]:
                    detect_mols_from_contours.append(ellipse)
                    detect_mols_center_from_contours.append([ellipse_x, ellipse_y])
                    # if plot_graph:
                    #     cv2.ellipse(self.img, ellipse, (255, 0, 0), 2)
                    #     plt.gca().add_patch(Ellipse((ellipse_x, ellipse_y), width=ellipse_width, height=ellipse_height, angle=ellipse_angle, color='yellow', fill=False))
                    self.plot_contour(cnt)

        if plot_graph:
            plt.imshow(self.img, extent=[self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2, self.offset_y_nm-self.len_nm, self.offset_y_nm])
            plt.ylim(self.offset_y_nm-self.len_nm, self.offset_y_nm)
            plt.xlim(self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2)
        self.selected_points_from_contours = selected_points_from_contours
        self.detect_mols_from_contours = detect_mols_from_contours
        self.detect_mols_center_from_contours = detect_mols_center_from_contours




    

    def plot_contour(self, cnt=None, color=(255, 0, 0), thickness=1, ellipse=True, rect=False, circle=False, text=True):
        self.contour_property(cnt=cnt)
        cv2.drawContours(self.img, self.cnt, 0, color, thickness)
        cv2.ellipse(self.img, self.ellipse, color, thickness)
        plt.imshow(self.img, extent=[self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2, self.offset_y_nm-self.len_nm, self.offset_y_nm])
        plt.scatter(self.ellipse_x, self.ellipse_y, c='blue', s=20)
        if text:
            # plt.text(self.ellipse_x, self.ellipse_y, 'w: %.2f h: %.2f ang: %.2f area: %.2f' % (self.ellipse_width, self.ellipse_height, self.ellipse_angle, self.area), color='red', fontsize=10)
             plt.text(self.ellipse_x, self.ellipse_y, 'area: %.2f' % (self.area), color='red', fontsize=10)

    

    def detect_edges(self, light_limit=[100, 200], blur_kernel=5) -> tuple:

        # Display original image


        # Blur the image for better edge detection
        img_blur= cv2.GaussianBlur(self.img, (blur_kernel,blur_kernel), 0) 
        
        # Sobel Edge Detection
        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        # Display Sobel Edge Detection Images
        
        # Canny Edge Detection
        self.edges = cv2.Canny(image=img_blur, threshold1=light_limit[0], threshold2=light_limit[1]) # Canny Edge Detection

    
    
    def select_points(self, shape_type='blob', dist_thres=1.8, light_limit=[50, 150], s=10, plot_graph=True):
        '''selct points from detected edges for large image (such as 250 nm) and select points from dectected blobs for small image (such as 10 nm)
        shape_type: 'blob' or 'edge'
        dist_thres: the distance threshold between two points (approximate to molecule size)
        light_limit: the light limit for selecting blob points
        plot_graph: plot the selected points on the image
        '''
        
        if shape_type=='edge':
            self.detect_edges()
            detect_mols=np.where(self.edges>0)
        else:
            detect_mols=np.where(self.img>light_limit[1])

        
        data_mols ={'x': detect_mols[1], 'y': detect_mols[0]}
        data_mols=pd.DataFrame(data_mols)
        data_mols=data_mols.drop_duplicates(ignore_index=True)

        selected_points=[]
        selected_points.append([data_mols.x[0], data_mols.y[0]])
        if plot_graph:
            plt.imshow(self.img, extent=[self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2, self.offset_y_nm-self.len_nm, self.offset_y_nm])
            plt.scatter(self.offset_x_nm-self.len_nm/2+self.unit_nm*data_mols.x[0], self.offset_y_nm-self.unit_nm*data_mols.y[0])
        for i in range(len(data_mols)):
            selected_points_array=np.array(selected_points)
            if np.sqrt((data_mols.x[i]-selected_points_array[:, 0])**2+(data_mols.y[i]-selected_points_array[:, 1])**2).min()*self.unit_nm>dist_thres:
                selected_points.append([data_mols.x[i], data_mols.y[i]])
                if plot_graph:
                    plt.scatter(self.offset_x_nm-self.len_nm/2+self.unit_nm*data_mols.x[i], self.offset_y_nm-self.unit_nm*data_mols.y[i], s=s)

        if plot_graph:
            plt.xlim(self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2)
            plt.ylim(self.offset_y_nm-self.len_nm, self.offset_y_nm)


        self.selected_points_nm=[[selected_points[i][0]*self.unit_nm+self.offset_x_nm-self.len_nm/2, -selected_points[i][1]*self.unit_nm+self.offset_y_nm] for i in range(len(selected_points))]
        return self.selected_points_nm
    
    def detect_mol_from_points(self, mol_dist_thres=2.5):
        '''select indivisal molecules from selected points (absolute position in points))'''
        self.select_points(plot_graph=False)
        mol_points=[]
        dist=cdist(np.array(self.selected_points_nm), np.array(self.selected_points_nm))
        for i in range(len(self.selected_points_nm)):
            if len(np.nonzero(dist[i])[0])>0:
                mol_min_dist = np.min(dist[i][np.nonzero(dist[i])])
                if mol_min_dist>mol_dist_thres:
                    mol_points.append(self.selected_points_nm[i])
        return mol_points








