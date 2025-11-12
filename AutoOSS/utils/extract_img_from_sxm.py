from .nanonis_sxm import SXM, STS, create_map, open_file, organise_spectra
import cv2
import numpy as np

def numpy_f4_to_grayscale(img_array):
    """Converts a big-endian float32 numpy array to an OpenCV grayscale image."""
    # Convert from big-endian to little-endian float32
    img_little_endian = img_array.astype('<f4')  # '<f4' is little-endian float32

    # Normalize the array to 0-255
    img_normalized = cv2.normalize(img_little_endian, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 8-bit grayscale (uint8)
    img_gray = cv2.convertScaleAbs(img_normalized)

    return img_gray


def extract_scanimg_from_sxm(filepath=None, params_show=False):
    sxm_file=SXM(filepath)
    sxm_img=sxm_file.data[0]
    img_gray=numpy_f4_to_grayscale(sxm_img)        # convert to grayscale within (0, 255) for opencv analysis
    if params_show:
        scan_mid_x, scan_mid_y = sxm_file.header['SCAN_OFFSET']
        scan_len_x, scan_len_y =sxm_file.header['SCAN_RANGE']
        scan_angle=sxm_file.header['SCAN_ANGLE']
        return img_gray, scan_mid_x, scan_mid_y, scan_len_x, scan_len_y, scan_angle
    else:
        return img_gray
    



