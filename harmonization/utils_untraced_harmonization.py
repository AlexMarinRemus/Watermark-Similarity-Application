"""
Utility file used by the harmonization.py file for the denoising of
untraced watermark images.
"""
import bm3d
import cv2
import numpy as np
from pykuwahara import kuwahara

from harmonization.binarize import sauvola_thresholding
from harmonization.contrast_enhancement import contrast_stretch
from harmonization.sharpen import unsharp_masking_gaussian
from harmonization.wavelet_denoising import (wavelet_denoise,
                                             wavelet_fourier_horizontal,
                                             wavelet_fourier_vertical)


def denoise_untraced(img, sigma_psd=25):
    """
    Performs denoising on an untraced watermark image
    1. Sharpening
    2. Wavelet denoising
    3. Contrast stretching
    4. BM3D denoising - takes the sigma user-provided argument for noise variation
    5. Kuwahara filtering
    Args:
        img: image that should be processed
        sigma_psd: noise variation expectation in the image; argument for BM3D
    Returns: the denoised image
    """
    original = unsharp_masking_gaussian(img)
    pic = wavelet_remove_all_lines(original)
    pic = contrast_stretch(pic)
    denoised = bm3d.bm3d(pic, sigma_psd= sigma_psd, stage_arg= bm3d.BM3DStages.ALL_STAGES)
    denoised = kuwahara(denoised, "gaussian", radius=3)
    return denoised

def threshold_untraced(image, window_size=25, k=0.1, morph_kernel= (3,3), iterations=3):
    """
    Performs thresholding by taking the user-provided arguments
    1. Sauvola thresholding
    2. Morphological closing
    3. Connected components filtering
    Args:
        image: denoised image that should be thresholded
        min_size: minimal amount of connected pixels needed for confidence
        window_size: window size parameter for the sauvola thresholding
        k: argument for the sauvola thresholding
    Returns: the thresholded image
    """
    thresh = sauvola_thresholding(image.astype(np.uint8), window_size=window_size, k=k)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                              kernel= cv2.getStructuringElement(cv2.MORPH_CROSS, morph_kernel), iterations=iterations)
    return thresh

def wavelet_remove_all_lines(img):
    """
    Performs general denoising, then removes all horizontal and vertical lines
    Args:
        img: image that should be processed
    Returns: the processed image
    """
    wavelet_first = wavelet_denoise(img)
    wavelet_image = wavelet_fourier_vertical(wavelet_first)
    theoth = wavelet_fourier_horizontal(wavelet_image)
    return theoth

def connected_component_analysis(image, min_size=200):
    """
    Remove connected components in the image with fewer pixels than min_size
    Args:
        min_size: minimal number of connected pixels for a connected component to stay. Default to 200
    Returns: thresholded image with the smaller components removed
    """
    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(image)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a
    # component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines.
    # You'll have to update the ranges in the for loop below.
    sizes = sizes[1:]
    nb_blobs -= 1

    # output image with only the kept components
    result = np.zeros_like(im_with_separated_blobs)
    # for every component in the image, keep it only if it's above min_size
    indices = np.where(sizes >= min_size)[0] + 1
    mask = np.isin(im_with_separated_blobs, indices)
    result[mask] = 255

    return result
