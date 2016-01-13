import cv2
import numpy as np
import math
from scipy.ndimage.filters import correlate
import superzoomit.misc.typeconv as tc
import superzoomit.misc.geotransform as gt
import superzoomit.misc.filter as filter


def _generate_ds_images(img, reduction_factor, mov, hfilter, noise_amp):
    num_imgs = len(mov)
    imgs = []
    img = tc.im2double(img)

    def maxabs(x):
        # Return the maximum absolute value in x
        #
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


def bw_single_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = tc.im2double(img)

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
