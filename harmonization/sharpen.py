"""
This file contains several methods of varying strength that can
be used to sharpen an image.
"""

import numpy as np
import cv2

def filter_sharpening_strong(image):
    """
    Args:
    image: the loaded image

    Returns: the image with the sharpening applied.

    Uses traditional sharpening filter that is convolved with the image
    to achieve a sharpened image. In this method the sharpening filter is
    strong.
    """

    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened

def filter_sharpening_light(image):
    """
    Args:
    image: the loaded image

    Returns: the image with the sharpening applied.

    Uses traditional sharpening filter that is convolved with the image
    to achieve a sharpened image. In this method the sharpening filter is
    light.
    """

    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened

def unsharp_masking_gaussian(image):
    """
    Args:
    image: the loaded image

    Returns: the image with the sharpening applied.

    Uses unsharp masking to sharpen the image using a gaussian blur.
    The idea is that the blur is subtracted from the image so the edges
    are more defined. The leads to less of an over-sharpening effect
    than what can happen with traditional filter based sharpening, and
    the lines also tend to be a bit thicker.
    """

    #For the gaussian blur, the kernel size is 0, so is
    # instead the kernel is computed from the sigma
    blurred = cv2.GaussianBlur(image,(0,0),3.0)
    unsharp_image = cv2.addWeighted(image, 2, blurred, -1, 0)
    return unsharp_image

def unsharp_masking_median(image):
    """
    Args:
    image: the loaded image

    Returns: the image with the sharpening applied.

    This method is very similar to the gaussian unsharp masking
    method, but instead it uses a median blur instead of a
    gaussian blur. The result is very similar, but has fewer
    high intensity areas, in other words it's a bit more gray.
    """

    blurred = cv2.medianBlur(image,7)
    unsharp_image = cv2.addWeighted(image, 2, blurred, -1, 0)
    return unsharp_image

def unsharp_masking_laplacian(image):
    """
    Args:
    image: the loaded image

    Returns: the image with the sharpening applied.
    
    Similar to the unsharp masking with gaussian, but using a
    laplacian filter instead. This sharpens a bit more effectively
    than the guassian, but the lines are thinner.
    """

    blurred = cv2.Laplacian(image,-1)
    unsharp_image = cv2.addWeighted(image, 1, blurred, -0.7, 0)
    return unsharp_image
