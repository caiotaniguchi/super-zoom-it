"""Module for type conversion of numpy ndarrays"""
import numpy as np


def im2double(im):
    """Converts a ndarray from uint8 to float32."""
    im_type = im.dtype.type
    if im_type is np.uint8:
        return (np.float32(im)) / 255
    raise Exception('Expected uint8 data type, got ' + im.dtype.name)


def im2uint8(im):
    """Converts a ndarray from float32 to uint8."""
    im_type = im.dtype.type
    if im_type is np.float32:
        im = (im * 255)
        im = np.rint(im)
        return np.uint8(im)
    raise Exception('Expected float32 data type, got ' + im.dtype.name)