import cv2
import numpy as np


def translate_and_crop(img, x_offset, y_offset):
    """Translate the image by x and y offsets, and then crop the empty border"""
    M = np.float32([[1,0,x_offset],[0,1,y_offset]])
    resulting_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    if x_offset >= 0 and y_offset >= 0:
        resulting_img = resulting_img[y_offset:, x_offset:]
    elif x_offset >= 0 and y_offset < 0:
        resulting_img = resulting_img[:y_offset, x_offset:]
    elif x_offset < 0 and y_offset >= 0:
        resulting_img = resulting_img[y_offset:, :x_offset]
    elif x_offset < 0 and y_offset < 0:
        resulting_img = resulting_img[:y_offset:, :x_offset]
    return resulting_img


def downsampling(img, reduction_factor):
    """Downsample the image by a reduction factor"""
    if reduction_factor < 1:
        raise Exception('The reduction factor must be greater than 1')
    return img[0::reduction_factor, 0::reduction_factor]
