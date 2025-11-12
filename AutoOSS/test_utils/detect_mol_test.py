import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from AutoOSS.img_modules.img_attrib import *


def remove_line(img, method='z_score', z_thres_value=3.0, perc_thres_value=99.5):
    ''' Automatically identify rows with anomalously high intensity and fix them, method default: 'z_score' ('percentile' as a option)'''

    # Compute row-wise mean
    row_means = img.mean(axis=1)

    # === Step 3: Detect anomalous rows using z-score ===

    if method=='percentile':
        threshold = np.percentile(row_means, perc_thres_value)
        anomaly_rows = np.where(row_means > threshold)[0]
    elif method=='z_score':
        z_scores = (row_means - row_means.mean()) / row_means.std()
        threshold = z_thres_value  # Z-score threshold, tune as needed
        anomaly_rows = np.where(z_scores > threshold)[0]
    print(f"Anomalous rows: {anomaly_rows}")

    # === Step 4: Inpainting or interpolation ===
    img_fixed = img.copy()
    for r in anomaly_rows:
        if 1 <= r < img.shape[0] - 1:
            img_fixed[r] = ((img[r-1].astype(np.int32) + img[r+1].astype(np.int32)) // 2).astype(np.uint8)

    return img_fixed


def plot_img(img_path='save_imgs_6_25_2\save_large_imgs\large_0_2.png', min_area=0.2, max_area=5, len_nm=20, pixel=256):


    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    gray=remove_line(gray)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    dx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(dx**2 + dy**2)



    magnitude_f32 = magnitude.astype(np.float32)

    # Optional: Normalize magnitude to 0-255 and convert to uint8 for thresholding
    magnitude_norm = cv2.normalize(magnitude_f32, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_uint8 = magnitude_norm.astype(np.uint8)


    blurred = cv2.bilateralFilter(magnitude_uint8, d=9, sigmaColor=75, sigmaSpace=75)

    # # Adaptive Thresholding / Local Thresholding, it captures local protrusions better, especially in uneven backgrounds.
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=-5)   # blockSize should be odd, and C is a constant subtracted from the mean or weighted mean.

    # thresh = cv2.adaptiveThreshold(magnitude_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=15, C=-5)

    # Find contours of white regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Filter small contours (noise)

    unit_nm=len_nm/pixel
    print([cv2.contourArea(cnt)*unit_nm*unit_nm for cnt in contours])
    protrusions = [cnt for cnt in contours if (cv2.contourArea(cnt)*unit_nm*unit_nm > min_area) & (cv2.contourArea(cnt)*unit_nm*unit_nm < max_area)]

    # Show mask
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    
    plt.title("Binary Mask")
    plt.axis("off")
    filtered_contours=[]
    for cnt in protrusions:
        p=cnt.squeeze()[:, 0].mean()
        q=cnt.squeeze()[:, 1].mean()
        plt.scatter(p, q)
        area=cv2.contourArea(cnt)*unit_nm*unit_nm
        plt.text(p+1, q, np.round(area, 2), color='red', fontsize=14)
        perimeter = cv2.arcLength(cnt, True)*unit_nm
        circularity = 4 * np.pi * area / (perimeter**2)
        plt.text(p-1, q+5, np.round(circularity, 2), color='green', fontsize=14)
        if (circularity>0.1) & (circularity<1.2):
            filtered_contours.append(cnt)

    mask = np.zeros_like(gray)
    # Draw filled contours on the mask
    cv2.drawContours(mask, filtered_contours, -1, color=255, thickness=cv2.FILLED)
    masked_result = cv2.bitwise_and(image, image, mask=mask)

    plt.imshow(mask, cmap="gray")

    # Show masked original image
    # plt.subplot(1, 3, 3)
    # plt.imshow(cv2.cvtColor(masked_result, cv2.COLOR_BGR2RGB))
    # plt.title("Masked Protrusions")
    # plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("magnitute_uint8")
    plt.axis("off")
    plt.imshow(magnitude_uint8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = 'c://Users//wun2//github//AutoOSS_Nanonis_clean_20250704//AutoOSS_Nanonis_clean//test_imgs//test_19.png'
    plot_img(img_path=img_path, min_area=0.2, max_area=5, len_nm=20, pixel=256)
   
