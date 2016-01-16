import cv2
import numpy as np
import math
from scipy.ndimage.filters import correlate
from scipy.signal import convolve2d
import superzoomit.misc.typeconv as tc
import superzoomit.misc.geotransform as gt
import superzoomit.misc.filter as filter


def _generate_ds_images(img, reduction_factor, mov, hfilter, noise_amp):
    num_imgs = len(mov)
    imgs = []
    img = tc.im2double(img)

    def maxabs(x):
        # Return the maximum absolute value in x
        return math.fabs(max(x.min(), x.max(), key=abs))

    vx_max = maxabs(mov[:, 0])
    vy_max = maxabs(mov[:, 1])
    img_width = img.shape[0]
    img_height = img.shape[1]

    for i in range(0, num_imgs):
        translated_img = gt.translate_and_crop(img, mov[i, 0], mov[i, 1])
        translated_img = translated_img[:img_width - vx_max, :img_height - vy_max]
        if len(translated_img.shape) == 3:
            blurred_img = np.zeros(
                (translated_img.shape[0], translated_img.shape[1], translated_img.shape[2]),
                dtype=np.float32
            )
            temp = correlate(translated_img[:, :, 0], hfilter)
            blurred_img[:, :, 0] = correlate(translated_img[:, :, 0], hfilter)
            blurred_img[:, :, 1] = correlate(translated_img[:, :, 1], hfilter)
            blurred_img[:, :, 2] = correlate(translated_img[:, :, 2], hfilter)
            downsampled_img = gt.downsampling(blurred_img, reduction_factor)
            noisy_img = downsampled_img + noise_amp * np.random.randn(
                downsampled_img.shape[0],
                downsampled_img.shape[1],
                downsampled_img.shape[2]
            )
        else:
            blurred_img = correlate(translated_img, hfilter)
            downsampled_img = gt.downsampling(blurred_img, reduction_factor)
            noisy_img = downsampled_img + noise_amp * np.random.randn(
                downsampled_img.shape[0],
                downsampled_img.shape[1]
            )
        imgs.append({
            'hr': translated_img,
            'ds': downsampled_img,
            'lr': noisy_img
        })
    return imgs


# %   rff: autocorrelation matrix for the observation vector.
# %   dy: Vector with vertical motion/displacement
# %   dx: Vector with horizontal motion/displacement
# %   nrows: (Dy) Number of rows of the desired vector
# %   ncols: (Dx) Number of columns of the desired vector
#     wx: window height
#     wy: window width
# rff, Dy, Dx, Wy, Wx, dy, dx
def _calculate_r(rff, dx, dy, wx, wy, nrows, ncols):
    # Center of the matrix - point with maximum value
    center_line, center_column = np.unravel_index(rff.argmax(), rff.shape)
    # Number of images
    p = len(dx)
    k = (p * wx * wy) / (nrows * ncols)

    # Reference vectors
    xv = np.arange(0, wx/nrows) + 1
    yv = np.arange(0, wy/ncols) + 1
    ref_line, ref_column = np.meshgrid(xv, yv)
    ref_line = ref_line.reshape(ref_line.shape[0] * ref_line.shape[1])
    ref_column = ref_column.reshape(ref_column.shape[0] * ref_column.shape[1])

    # Initialize matrix
    tmp_line = np.meshgrid(np.zeros(9), ref_line)[1]
    tmp_column = np.meshgrid(np.zeros(9), ref_column)[1]
    ref_v_line = tmp_line.T.reshape(tmp_line.shape[0] * tmp_line.shape[1])
    ref_v_column = tmp_column.T.reshape(tmp_column.shape[0] * tmp_column.shape[1])

    # Inner loop - fill R's columns - line vectors
    inner_ref_m_column = np.meshgrid(ref_v_column, ref_v_column)[0] * 3
    inner_ref_m_line = np.meshgrid(ref_v_line, ref_v_line)[0] * 3

    # Outer loop - fill R's lines - column vectors
    outer_ref_m_line = inner_ref_m_line.T
    outer_ref_m_column = inner_ref_m_column.T

    # Include motion matrices
    motion_line = np.meshgrid(dy, np.arange(k/p) + 1)[0]
    motion_column = np.meshgrid(dx, np.arange(k/p) + 1)[0]
    motion_line = motion_line.T.reshape(motion_line.shape[0] * motion_line.shape[1])
    motion_column = motion_column.T.reshape(motion_column.shape[0] * motion_column.shape[1])

    # Matrices with reference values of the motion vectors
    # Each image block is a line vector with dx*dy width
    tmp_v = np.arange(k) + 1
    ref_motion_line = np.meshgrid(tmp_v, motion_line)[1]
    ref_motion_column = np.meshgrid(tmp_v, motion_column)[1]

    # Matrices with reference values of the motion vectors shifted
    # Each image block is a line vector with dx*dy width
    shifted_motion_line = np.meshgrid(motion_line, tmp_v)[0]
    shifted_motion_column = np.meshgrid(motion_column, tmp_v)[0]

    # Sum to calculate the position in rff
    # rff center not included
    # values_line = k + -dy(i) -m2 + dy(l)
    values_lines = inner_ref_m_line - outer_ref_m_line + ref_motion_line - shifted_motion_line
    values_columns = inner_ref_m_column - outer_ref_m_column + ref_motion_column - shifted_motion_column
    # Shift values to rff center
    # Add 1 to account for diferences in indexing between MATLAB and Python
    values_lines += center_line + 1
    values_columns += center_column + 1
    # Flatten matrices to vectors
    values_lines = values_lines.T.reshape(values_lines.shape[0] * values_lines.shape[1])
    values_columns = values_columns.T.reshape(values_columns.shape[0] * values_columns.shape[1])

    # Calculate position considering rff a column vector
    positions = (rff.shape[0] * (values_columns - 1)) + values_lines
    rff_v = rff.reshape(rff.shape[0] * rff.shape[1])

    return np.reshape(rff_v[positions.astype(int) - 1], (k, k))


def bw_single_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # TODO FIX
    # img = tc.im2double(img)

    # Motion vector. Considers a 3:1 resolution increase
    mov = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1], [-1, 1], [1, 0], [1, -1], [1, 1]])
    # Number of images
    num_imgs = len(mov)
    # Interpolation factor
    td = 3
    # Noise std
    sigma_e = 0.03
    # Splits the motion vector between x and y displacements
    dx = mov[:, 0]
    dy = mov[:, 1]
    # TODO Variance of noise ?
    vn = sigma_e ** 2
    # TODO ???
    sz = 3
    # Gaussian standard deviation. Uses MATLAB's fspecial() default = 0.5
    gaussian_std = 0.5
    # Gaussian Filter
    gaussian_filter = filter.gaussian_fspecial(sz + 1)
    # LR images
    imgs = _generate_ds_images(img, td, mov, gaussian_filter, sigma_e)

    # TODO Initialization of parameters ???
    # TODO FIX
    # ml = m = len(imgs[0]['hr'].shape[0])
    # mc = n = len(imgs[0]['hr'].shape[1])
    # x = np.zeros(m, n, num_imgs)
    # xd = np.zeros(ml/td, mc/td, num_imgs)
    # for i in range(0, num_imgs):
    #     x[:, :, i] = imgs[i]['hr']
    #     xd[:, :, i] = imgs[i]['lr']

    # TODO Values to calculate W ???
    nl = 3
    nc = 3

    # PSF auto correlation function - rdd
    v = 0.75
    a = 10 * td
    xv = yv = (np.arange(0, a) + 1) - a/2
    x, y = np.meshgrid(xv, yv)
    rdd = v ** (np.sqrt(x ** 2 + y ** 2))

    # PSF representation - h
    m = 10 * td
    xv = yv = (np.arange(0, m) + 1) - m/2
    x, y = np.meshgrid(xv, yv)
    h = ((np.fabs(x) <= 3) & (np.fabs(y)<= 3)).astype(float)
    h *= filter.gaussian_fspecial(hsize=m)

    # Compute the cross-correlation and auto correlation
    # r_df(x,y) = r_dd(x,y) * h(x,y)
    rdf = convolve2d(rdd, h)
    # r_ff(x,y) = r_dd(x,y) * h(x,y) * h(-x,-y)
    rff = convolve2d(rdf, h)

    # Adaptative process
    wx = nc * td
    wy = nl * td
    nrows = ncols = lx = ly = td
    k = (num_imgs * wx * wy) / (lx * ly)
    p = nc * nl     # number of pixels in R

    return _calculate_r(rff, dx, dy, wx, wy, nrows, ncols)





# %%%%%%%%%%%%%%%%%%%%%%%% Processo Adaptativo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Wx = nc*TD; Wy = nl*TD;
# Dx = TD; Dy = TD;
# K = ni*Wx*Wy/TD/TD;
#
# p = Wx/TD*Wy/TD * ni; % pixels de R
#
# R = GetR(rff, dy, dx, Dy, Dx, Wy, Wx);
# P = GetP(rdf, dy, dx, Dy, Dx, Wy, Wx);


