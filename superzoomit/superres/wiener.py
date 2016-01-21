import cv2
import numpy as np
import math
from scipy.ndimage.filters import correlate
from scipy.signal import convolve2d
import superzoomit.misc.typeconv as tc
import superzoomit.misc.geotransform as gt
import superzoomit.misc.mattransform as mt
import superzoomit.misc.filter as filter


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
    ref_line = mt.vectorize_matrix(ref_line)
    ref_column = mt.vectorize_matrix(ref_column)

    # Initialize matrix
    zeros = np.zeros(9)
    ref_v_line = mt.generate_and_flatten_grid(zeros, ref_line, 'Y')
    ref_v_column = mt.generate_and_flatten_grid(zeros, ref_column, 'Y')

    # Inner loop - fill R's columns - line vectors
    inner_ref_m_column = np.meshgrid(ref_v_column, ref_v_column)[0] * 3
    inner_ref_m_line = np.meshgrid(ref_v_line, ref_v_line)[0] * 3

    # Outer loop - fill R's lines - column vectors
    outer_ref_m_line = inner_ref_m_line.T
    outer_ref_m_column = inner_ref_m_column.T

    # Include motion matrices
    yv = np.arange(k/p) + 1
    motion_line = mt.generate_and_flatten_grid(dy, yv, 'X')
    motion_column = mt.generate_and_flatten_grid(dx, yv, 'X')

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
    # values_line = k + -dy(i) -m2 + dy(l) + rff_center
    # Add 1 to account for diferences in indexing between MATLAB and Python
    values_lines = inner_ref_m_line - outer_ref_m_line + ref_motion_line - shifted_motion_line + center_line + 1
    values_columns = inner_ref_m_column - outer_ref_m_column + ref_motion_column - shifted_motion_column + center_column + 1

    # Flatten matrices to vectors
    values_lines = mt.vectorize_matrix(values_lines.T)
    values_columns = mt.vectorize_matrix(values_columns.T)


    # Calculate position considering rff a column vector
    positions = (rff.shape[0] * (values_columns - 1)) + values_lines
    rff_v = mt.vectorize_matrix(rff)

    return np.reshape(rff_v[positions.astype(int) - 1], (k, k))


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


# %   rdf: Cross correlation matrix between the desired vector and the
# %        observation vector.
# %   dy: Vector with vertical motion/displacement
# %   dx: Vector with horizontal motion/displacement
# %   nrows: (Dy) Number of rows of the desired vector
# %   ncols: (Dx) Number of columns of the desired vector
# %   wy: Number of rows of the observation window
# %   wx: Number of columns of the observation window
def _calculate_p(rdf, dx, dy, wx, wy, nrows, ncols):
    # Given an observation matrix (W), calculate the distance between
    # its points and the subwindow (D).
    # Each column of the P matrix represents a subwindow D.

    # Center of the matrix - point with maximum value
    center_line, center_column = np.unravel_index(rdf.argmax(), rdf.shape)
    # Number of images
    nimgs = len(dx)
    k = (nimgs * wx * wy) / (nrows * ncols)

    # Outer loop
    # Inner matrix (subwindow) - varies in the lines of the P matrix
    pos_x = np.arange(ncols) + 1
    pos_y = np.arange(nrows) + 1

    half_x = np.fix((ncols + 1) / 2)
    half_y = np.fix((nrows + 1) / 2)

    pos_x -= half_x
    pos_y -= half_y

    v_pos_x = np.meshgrid(pos_x, pos_x)[0]
    v_pos_y = np.meshgrid(pos_y, pos_y)[1]

    # TODO TESTAR
    v_pos_x = v_pos_x.T.reshape(v_pos_x.shape[0] * v_pos_x.shape[1])
    v_pos_y = v_pos_x.T.reshape(v_pos_x.shape[0] * v_pos_x.shape[1])

    m_pos_x = np.meshgrid(v_pos_x, np.zeros(k))[0]
    m_pos_y = np.meshgrid(v_pos_y, np.zeros(k))[0]

    v_dx = np.meshgrid(dx, np.arange(k/nimgs) + 1)[0]
    v_dy = np.meshgrid(dy, np.arange(k/nimgs) + 1)[0]

    v_dx = v_dx.T.reshape(v_dx.shape[0] * v_dx.shape[1])
    v_dy = v_dy.T.reshape(v_dy.shape[0] * v_dy.shape[1])

    m_dx = np.meshgrid(np.zeros(ncols * nrows), v_dx)[1]
    m_dy = np.meshgrid(np.zeros(ncols * nrows), v_dy)[1]

    w_pos_x = np.arange(wx/ncols) + 1
    w_pos_y = np.arange(wy/nrows) + 1

    half_x = np.fix((wx/ncols + 1) / 2)
    half_y = np.fix((wy/nrows + 1) / 2)

    w_pos_x = (w_pos_x - half_x) * ncols
    w_pos_y = (w_pos_y - half_y) * nrows

    v_w_pos_x = np.meshgrid(w_pos_x, w_pos_x)[0]
    v_w_pos_y = np.meshgrid(w_pos_y, w_pos_y)[1]

    v_w_pos_x = v_w_pos_x.T.reshape(v_w_pos_x.shape[0] * v_w_pos_x.shape[1])
    v_w_pos_y = v_w_pos_y.T.reshape(v_w_pos_y.shape[0] * v_w_pos_y.shape[1])

    c_pos_x = np.meshgrid(np.zeros(nimgs), v_w_pos_x)[1]
    c_pos_y = np.meshgrid(np.zeros(nimgs), v_w_pos_y)[1]

    c_pos_x = c_pos_x.T.reshape(c_pos_x.shape[0] * c_pos_x.shape[1])
    c_pos_y = c_pos_y.T.reshape(c_pos_y.shape[0] * c_pos_y.shape[1])

    m_c_pos_x = np.meshgrid(np.zeros(ncols * nrows), c_pos_x)[1]
    m_c_pos_y = np.meshgrid(np.zeros(ncols * nrows), c_pos_y)[1]


    # [~, colPosXmat] = meshgrid(zeros(Dx*Dy,1), colPosX); % 81x9
    # [~, colPosYmat] = meshgrid(zeros(Dx*Dy,1), colPosY);


    # % Deslocamento externo (par de for)
    # % Matriz interna (pequena) - Varia nas linhas da matriz P
    # posX = 1:Dx;
    # posY = 1:Dy;
    #
    # meioX = fix((Dx + 1)/2);
    # meioY = fix((Dy + 1)/2);
    #
    # posX = posX - meioX;
    # posY = posY - meioY;
    #
    # [posXvec, ~] = meshgrid(posX);
    # [~, posYvec] = meshgrid(posY);
    #
    # posXvec = reshape(posXvec, 1, size(posXvec,1)*size(posXvec,2));
    # posYvec = reshape(posYvec, 1, size(posYvec,1)*size(posYvec,2));
    #
    # % Matriz pequena, sem centralizar
    # [posXmat, ~] = meshgrid(posXvec, zeros(K,1));
    # [posYmat, ~] = meshgrid(posYvec, zeros(K,1));
    #
    # % Descolamento interno (trinca de for)
    #
    # % 1º for - Deslocamento
    #     % 2º for - Matrix pequena deslocando nas colunas da grande
    # % 3º for - Matrix pequena deslocando nas linhas da grande
    #
    # % Etapa de deslocamento
    # [vec_dx ~] = meshgrid(dx, 1:K/ni);
    # [vec_dy ~] = meshgrid(dy, 1:K/ni);
    #
    # vec_dx = reshape(vec_dx, size(vec_dx,1)*size(vec_dx,2), 1); % 81x1
    # vec_dy = reshape(vec_dy, size(vec_dy,1)*size(vec_dy,2), 1);
    #
    # [~, mat_dx] = meshgrid(zeros(Dx*Dy,1), vec_dx); % 81x9
    # [~, mat_dy] = meshgrid(zeros(Dx*Dy,1), vec_dy);
    #
    # % Colunas e linhas de
    # posWinX = 1:Wx/Dx;
    # posWinY = 1:Wy/Dy;
    #
    # centroX = fix((Wx/Dx + 1)/2);
    # centroY = fix((Wy/Dy + 1)/2);
    #
    # posWinX = (posWinX - centroX)*Dx;
    # posWinY = (posWinY - centroY)*Dy;
    #
    # [posWinXvec, ~] = meshgrid(posWinX); % 3x3
    # [~, posWinYvec] = meshgrid(posWinY);
    #
    # posWinXvec = reshape(posWinXvec, size(posWinXvec,1)*size(posWinXvec,2), 1); % 9x1
    # posWinYvec = reshape(posWinYvec, size(posWinYvec,1)*size(posWinYvec,2), 1);
    #
    # [~, colPosX] = meshgrid(zeros(ni,1), posWinXvec); % 9x9
    # [~, colPosY] = meshgrid(zeros(ni,1), posWinYvec);
    #
    # colPosX = reshape(colPosX, size(colPosX,1)*size(colPosX,2), 1); % 81x1
    # colPosY = reshape(colPosY, size(colPosY,1)*size(colPosY,2), 1);
    #
    # [~, colPosXmat] = meshgrid(zeros(Dx*Dy,1), colPosX); % 81x9
    # [~, colPosYmat] = meshgrid(zeros(Dx*Dy,1), colPosY);


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


