import numpy as np
from math import pi


def _gaussian2d(xlen, ylen=None, xstd=0.5, ystd=None):
    # Calculate the unnormalized Gaussian 2D matrix
    #
    # Parameters
    # ----------
    # xlen : int
    #     Length of x / width of the window
    # ylen : int
    #     Length of y / height of the window
    # xstd : float
    #     Standard deviation of x
    # xstd : float
    #     Standard deviation of y
    #
    x0 = ((xlen + 1) / 2.0) - 1
    if ylen == None:
        ylen = xlen
        y0 = x0
    else:
        y0 = ((ylen + 1) / 2.0) - 1
    if ystd == None:
        ystd = xstd

    result = np.zeros(shape = (xlen, ylen))
    for xi in range(0, xlen):
        for yj in range(0, ylen):
            result[xi, yj] = np.exp(-((xi - x0) ** 2 / (2 * xstd ** 2) + (yj - y0) ** 2 / (2 * ystd ** 2)))
    return result


def gaussian_fspecial(hsize=3, sigma=0.5):
    """Calculate the Gaussian 2D matrix, normalized by the sum of its terms. Mimics MATLAB's fspecial function

    Parameters
    ----------
    hsize : int
        Size of the square matrix
    sigma : float
        Standard deviation of the gaussian distribution
    """
    gaussian_m = _gaussian2d(hsize, hsize, sigma, sigma)
    return gaussian_m / gaussian_m.sum()


def gaussian_window(hsize=3, sigma=0.5):
    """Calculate the normalized Gaussian 2D matrix, as defined in [1]

    Parameters
    ----------
    hsize : int
        Size of the square matrix
    sigma : float
        Standard deviation of the gaussian distribution

    References
    ----------
    .. [1] Weisstein, Eric W. "Gaussian Function."
    Retrieved January 13, 2016, from MathWorld--A Wolfram Web Resource.
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    gaussian_m = _gaussian2d(hsize, hsize, sigma, sigma)
    return gaussian_m / (2 * pi * sigma ** 2)
