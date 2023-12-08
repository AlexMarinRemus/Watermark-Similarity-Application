"""
This file contains several methods related to thresholding an image in order 
to binarize it.
"""

import numpy as np
import skimage.filters as Filters
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte


def niblack_thresholding(image, window_size=25, k=0.2):
    """
    Performs Niblack thresholding, which is a form of local thresholding, on an image.

    Args:
        image: The image to be thresholded.
        window_size: Optional, a window size is specified that represents the size of the
                     sliding window that is used. The default is 25.
        k: Optional, value k is specified that represents the constant that the standard deviation of the
        window is scaled by. The default is 0.2.

    Returns: The image thresholded with a niblack threshold.
    """

    thresh_niblack = Filters.threshold_niblack(
        image, window_size=window_size, k=k)
    binary_niblack = (image < thresh_niblack).astype(np.uint8)
    binary_niblack *= 255
    return binary_niblack


def sauvola_thresholding(image, window_size=25, k=0.2, r=None):
    """
    Performs Sauvola thresholding, which is a form of local thresholding, on an image.
    Sauvola thresholding is a modification of Niblack, that tends to reduce the amount of background
    texture.

    Args:
        image: The image to be thresholded.
        window_size: Optional, a window size is specified that represents the size of the
                     sliding window that is used. The default is 25.
        k: Optional, a value k is specified that represents a scaling factor. The default is 0.2.
        r: Optional, the r value is the normalization factor for the standard deviation.. The default is None.

    Returns: the image thresholded with a sauvola threshold.
    """

    thresh_sauvola = Filters.threshold_sauvola(
        image, window_size=window_size, k=k, r=r)
    binary_sauvola = (image < thresh_sauvola).astype(np.uint8)
    binary_sauvola *= 255
    return binary_sauvola


def otsu_thresholding(image):
    """
    Performs Otsu thresholding, a global thresholding technique, on an image.
    Otsu thresholding works by finding the weighted variance that minimizes the
    difference between white (foreground) and black (background) pixels.

    Args:
        image: The image to be thresholded.

    Returns: The image thresholded with an otsu threshold.
    """

    binary_otsu = (image < Filters.threshold_otsu(image)).astype(np.uint8)
    binary_otsu *= 255
    return binary_otsu


def entropy_thresholding(image, radius=2, threshold=0.8):
    """
    Performs thresholding using the image's entropy. An images entropy
    is the measure of how complex the image is in certain areas. Thresholding
    the image on entropy keeps the more complex parts of the image. Note that
    this isn't as effective at caputring fine details as the other thresholds.

    Args:
    image: The image to be thresholded.
    radius: Optional, the radius to which entropy is calculated. The default is 2.
    threshold: Optional, the threshold to split the entropy with. The default is 0.8.

    Returns: The image thresholded with an entropy threshold.

    """

    entropy_image = entropy(img_as_ubyte(image), disk(radius))
    scaled_entropy = entropy_image
    if entropy_image.max() != 0:
        scaled_entropy = entropy_image / entropy_image.max()
    binary_entropy = (scaled_entropy > threshold).astype(np.uint8)
    binary_entropy *= 255
    return binary_entropy
