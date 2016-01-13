import numpy as np
import warnings
from skimage.measure import structural_similarity
from math import log10


def mse(matrix1, matrix2):
    """Calculate the Mean Squared Error (MSE) between two matrices

    Both matrices must have the same dimensions. Can handle 2D or 3D matrices.
    """
    def inner_mse(matrix1, matrix2):
        acc = (matrix1.astype("float") - matrix2.astype("float")) ** 2
        return np.sum(acc) / float(matrix1.shape[0] * matrix2.shape[1])

    dimensions = len(matrix1.shape)
    if dimensions == 3:
        total_mse = 0.
        for i in range(matrix1.shape[2]):
            total_mse = mse(matrix1[:,:,i], matrix2[:,:,i])
        return total_mse
    return inner_mse(matrix1, matrix2)


def psnr(matrix1, matrix2, maxValue=None):
    """Calculate the Peak Signal to Noise Ratio (PSNR) between two matrices

    Parameters
    ----------
    matrix1, matrix2 : ndarray
        Both must be of same dimensions and types.
    maxValue : int, float or None
        The maximum value an element can assume in the matrices. If not specified,
        it will assume 0-1 range for float32 matrices and 0-255 range for uint8
        matrices.
    """
    mse_result = mse(matrix1, matrix2)
    if maxValue == None:
        m_type = matrix1.dtype.type
        if m_type is np.uint8:
            maxValue = 255
        elif m_type == np.float32:
            maxValue = 1
        else:
            raise Exception('Expected float32 or  data type, got ' + im.dtype.name)
        with np.errstate(divide='ignore'):
            return 20 * log10((maxValue ** 2) / mse_result)


def ssim(image1, image2):
    """Calculate the Structural Similarity (SSIM) index between two images

    Uses the SSIM implemented in the Scikit-image package with the same parameters used in [1].

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
    """
    return structural_similarity(image1, image2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)