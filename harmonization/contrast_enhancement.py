"""
Contrast enhancement module
"""
import cv2
import numpy as np

def CLAHE(image, clip_limit=2, tile_grid_size=(8,8)):
    """
    Args:
    image: the loaded image.
    clip_limit: the threshold value for contrast limiting.
    tile_grid_size: size of the window used for histogram equalization.

    Returns: the image with CLAHE applied.

    CLAHE is a form of adaptive histogram equalization that ensures that
    noise is not over-amplified by limiting it. CLAHE is used as a local
    contrast stretching technique.
    """
    #Creating CLAHE
    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)

    # Apply CLAHE to the original image
    img = clahe.apply(image)
    return img

def flat_field_correction(image, flat_file_path= "./harmonization/flat.png"):
    """
    Flat field correction is the technique of removing a specific type of shading
    from an image. For this case, the shading is in the form of a vignette around
    the edges of an image, as can be seen in the flat.png image that is the defaul
    path for flat_file_path.

    Args:
        image: The image to be corrected.
        flat_file_path: Optional, the path to the flat field image. The default is "./harmonization/flat.png".

    Returns: The image with flat field correction appplied
    """
    flat = cv2.imread(flat_file_path, cv2.IMREAD_GRAYSCALE)
    flat = 255 - flat
    flat = cv2.resize(flat, (image.shape[1], image.shape[0]))
    # Flat is added to image to enhance the edges' intensities and
    # reduce the intensities towards the middle
    result = cv2.add(image, flat)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, \
        cv2.CV_8U).astype(np.uint8)
    return result


def normal_stretch(image, median_size=9, gaussian_size=(3,3), gaussian_sigma=3):
    """
    Perform normal stretching. This is typically a precursor to flat field correction.
    Note: this method may create strange wave-like fragments in the image.

    Args:
        img: The image to be stretched.
        medianSize: Optional, the size of the median filter. The default is 9.
        gaussianSize: Optional, the size of the gaussian filter. The default is (3, 3).
        gaussianSigma: Optional, the sigma of the gaussian filter. The default is 3.

    Returns: The stretched image

    """
    img = cv2.medianBlur(image,ksize=median_size)
    img = cv2.GaussianBlur(image,ksize=gaussian_size, sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
    img = CLAHE(image)
    img = cv2.equalizeHist(image)
    return img


def contrast_stretch(image):
    """
    Perform contrast stretching on an image
    Args:
        image: Image to be contrast stretched.

    Returns: Contrast stretched image

    Contrast stretching takes an image, and scale's the intensity histogram
    of the image such that the intensities in the image scale the full range
    of possible pixel values, from 0 to 255.
    """

    flatten = image.flatten()
    # Isolate the minimum and maximum intensities
    min_hist = np.amin(flatten)
    max_hist = np.amax(flatten)
    # Stretch the minimum and maximum intensities to fill whole range of histogram
    stretched_img = ((255 / (max_hist - min_hist)) *
                     (image - min_hist)).astype(np.uint8)

    # Clamp image so that pixel values are valid
    stretched_img = np.clip(stretched_img, 0, 255)
    return stretched_img


def remove_shadows(image, dilation_mask, medianblur_k):
    """
    Removes the shadows of an image by isolating the background and removing it
    Args:
        image: the image to remove shadows from.
        dilation_mask: the dilation mask to be used for the dilation operation.
        medianblur_k: the k used for the median blur operation.

    Returns: the image with shadows removed
    """
    dilate = cv2.dilate(image, dilation_mask)
    background = cv2.medianBlur(dilate, medianblur_k)
    diff = 255 - cv2.absdiff(image, background)
    normalized = cv2.normalize(
        diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized

def stretch_flat_field(image, median_size=9, gaussian_size=(3,3), gaussian_sigma=3):
    """
    Args:
        image: the image to be stretched.
        median_size: the k value (size) used for median blurring.
        gaussian_size: the k value (size) used for gaussian blurring.
        gaussian_sigma: the standard deviation of the gaussian kernel used for the blur.

    Returns: Image that is normally stretched then has flat field correction applied.

    Perform normal stretching with flat field correction.
    """
    image = normal_stretch(image, median_size, gaussian_size, gaussian_sigma)
    result = flat_field_correction(image)
    return result

def top_hat(image, iterations= 3, operation= cv2.MORPH_OPEN, struct= cv2.MORPH_CROSS, \
    k_size=(7, 7), morph_iterations= 4):
    """
    Args:
    image: the loaded image.
    iterations: the number of times that the top_hat operation is performed
    operation: the type of operation that is performed. Can be either an opening
            or closing operation.
    struct: the type of structuring element to use. Can be a cross, ellipse, or rect.
            For more information on morphological structures refer to opencv documentation.
    k_size: The size of the structuring element to be used.
    morph_iterations: the number of times the morphological operation is performed during
            one top_hat operation.


    Performs the top-hat transform to an image with hyperparameters that can be edited
    """
    result = image
    for _ in range(iterations):
        result = result - cv2.morphologyEx(result, operation,
                                           cv2.getStructuringElement(struct,k_size),
                                           iterations= morph_iterations)
    return result


def ameliorate_contrast_on_margins(image):
    """
    Args:
        image: the image to be processed.

    Returns: The image with lower contrast on the margins.   

    Takes self.image. It calculates the histogram and then replaces all values that
    are larger than the most common value with the most common value in the image. 
    The new image is then returned.
    """
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    most_common_bin = np.argmax(histogram)
    most_common_value = int(most_common_bin)
    image[image >= most_common_value] = most_common_value
    return image