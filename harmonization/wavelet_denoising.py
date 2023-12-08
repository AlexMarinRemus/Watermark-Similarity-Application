"""
Module for denoising images using wavelets
"""

import skimage
from skimage.restoration import denoise_wavelet
import numpy as np
import pywt
import cv2
# import matplotlib.pyplot


def wavelet_denoise(image):
    """
    Takes the watermark image as input and applies simple wavelet denoising.
    """
    float_img = skimage.img_as_float(image)
    clipped = np.clip(float_img, 0, 1)
    # denoised_image = denoise_wavelet(clipped, wavelet = 'db30', rescale_sigma = True, wavelet_levels = 3)
    denoised_image = denoise_wavelet(
        clipped, wavelet='sym4', rescale_sigma=True)

    # dd = denoise_wavelet(clipped, wavelet = 'coif4', rescale_sigma = True)
    # cv2.imshow("dd", dd)
    return denoised_image


def wavelet_fourier_vertical(image, levels=4, wavelet='sym4', sigma=3):
    """
    Args:
    image: the loaded image.
    levels: the number of levels of decomposition.
    wavelet: the type of wavelet transform used.
    sigma: the dampening coefficient.

    Returns: the denoised image.

    Takes the watermark image as input and applies applies Fourier transform
    in wavelet domain in order to remove the vertical lines.
    The implementation follows the idea in https://opg.optica.org/oe/fulltext.cfm?uri=oe-17-10-8567&id=179485

    Args:
        image: The watermark image
        levels: Optional, the number of levels of decomposition. The default is 4.
        wavelet: Optional, the type of wavelet. The default is 'sym4'.
        sigma: Optional, the damping coefficient. The default is 3.

    Returns: The denoised image
    """

    coeff_horizontal = []
    coeff_vertical = []
    coeff_diagonal = []

    # For each level of decomposition we perform wavelet decomposition which recursively
    # splits the input image into a low frequency and a details band. The low
    # frequency band represents the approximation coefficients at a coarser scale.
    # the detail bands correspond to the horizontal, vertical, respectively diagonal
    # coefficients. We save the coefficients at each decomposition level in an array
    # of coefficients which will be used for reconstruction after the DFT.
    for i in range(0, levels):
        image = skimage.img_as_float(image)
        (image, (coeff_horiz_i, coeff_vert_i, coeff_diag_i)) = pywt.dwt2(image, wavelet)
        coeff_horizontal.append(coeff_horiz_i)
        coeff_vertical.append(coeff_vert_i)
        coeff_diagonal.append(coeff_diag_i)

    # For each level of decomposition, the vertical details coefficients are
    # Fourier transformed. We multiply the resulting coefficients in the DFT
    # domain by a Gaussian function. This has the effect of dampening the
    # coefficients close to the x-axis in the Fourier domain, which eliminates the
    # vertical lines.
    for i in range(levels):
        fCv = np.fft.fftshift(np.fft.fft2(coeff_vertical[i]))
        (my, mx) = fCv.shape
        indices = np.arange(-np.floor(my/2), -np.floor(my/2) + my)
        damp = 1 - np.exp(-indices**2 / (2 * np.power(sigma, 2.)))
        fCv = fCv * np.tile(damp, (mx, 1)).T
        # After the dampening, the coefficients will be transformed back to the
        # wavelet space.
        coeff_vertical[i] = np.fft.ifft2(np.fft.ifftshift(fCv)).real

    # The destriped image is reconstructed from the refined coefficients.
    img = image.copy()
    for i in range(levels-1, -1, -1):
        img = img[:coeff_horizontal[i].shape[0], :coeff_horizontal[i].shape[1]]
        img = pywt.idwt2(
            (img, (coeff_horizontal[i], coeff_vertical[i], coeff_diagonal[i])), wavelet)

    return img.real


def wavelet_fourier_horizontal(image, levels=4, wavelet='db2', sigma=3):
    """
    Args:
    image: the loaded image.
    levels: the number of levels of decomposition.
    wavelet: the type of wavelet transform used.
    sigma: the dampening coefficient.

    Returns: the denoised image.
    
    This performs wavelet-fourier vertical line removal for an image rotated by 90 degrees, 
    which is equivalent to performing horizontal line removal. 

    Args:
        image: The watermark image
        levels: Optional, the number of levels of decomposition. The default is 4.
        wavelet: Optional, the type of wavelet. The default is 'db2'.
        sigma: Optional, the damping coefficient. The default is 3.

    Returns: The denoised image
    """
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = wavelet_fourier_vertical(image, levels, wavelet, sigma)
    # The image is rotated back after the line removal to have the initial orientation.
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def wavelet_fourier(image, levels=2, wavelet='haar', sigma=3):
    """
    This does more wavelet line removal and denoising for different parameters. 
    TODO: Adjust parameters

    Args:
        image: The watermark image
        levels: Optional, the number of levels of decomposition. The default is 2.
        wavelet: Optional, the type of wavelet. The default is 'haar'.
        sigma: Optional, the damping coefficient. The default is 3.

    Returns: The denoised image
    """
    image = wavelet_fourier_vertical(image, levels, wavelet, sigma)
    image = denoise_wavelet(image, wavelet='haar', rescale_sigma=True)
    return image

def wavelet_traced(image, levels=8, wavelet='dmey', sigma=2.5, option=1):
    """
    Args:
        image: The watermark image
        levels: Optional, the number of levels of decomposition. The default is 8.
        wavelet: Optional, the type of wavelet. The default is 'dmey'.
        sigma: Optional, the damping coefficient. The default is 2.5.
        option: integer determining if the vertical, horizontal, both, or no lines
                should be removed from the image.

    Returns: The denoised image

    Method similar to wavelet_fourier, except that operations are done in a different 
    order and it is used in denoising the traced watermarks. There are some other 
    values for the wavelet parameters that achieve good results for different images. 
    These have been commented out since the current parameters perform the best with 
    the current dataset. They could become useful later on in case other parts from 
    the harmonization are changed, and stronger or weaker line removal than the current
    one is needed.
    """
    image = wavelet_denoise(image)
    # If the option is equal to 1, remove only the vertical lines.
    if option == 1:
        image = wavelet_fourier_vertical(image, levels = 8, wavelet='dmey', sigma=2.5)
        return image
    # If the option is equal to 2, remove only the horizontal lines.
    elif option == 2:
        image = wavelet_fourier_horizontal(image, levels = 8, wavelet='db10', sigma=3)
        # image = wavelet_fourier_horizontal(image, levels = 7, wavelet='bior4.4', sigma=9)
        # image = wavelet_fourier_horizontal(image, levels = 3, wavelet='dmey', sigma=3)
        return image
    # If the option is equal to 3, remove both the horizontal and the vertical lines.
    elif option == 3:
        image = wavelet_fourier_vertical(image, levels = 8, wavelet='dmey', sigma=2.5)
        image = wavelet_fourier_horizontal(image, levels = 8, wavelet='db10', sigma=2.5)
        return image
    # Otherwise, no line is removed from the image.
    # image = wavelet_fourier_vertical(image, levels = 8, wavelet='sym4', sigma=0.4)
    # image = wavelet_fourier_vertical(image, levels = 7, wavelet='bior4.4', sigma=9)
    # image = wavelet_fourier_vertical(image, levels = 3, wavelet='dmey', sigma=3)
    return image

# print(pywt.wavelist('dmey'))
# print(pywt.wavelist(kind='discrete'))
# print(pywt.wavemngr("read"))
