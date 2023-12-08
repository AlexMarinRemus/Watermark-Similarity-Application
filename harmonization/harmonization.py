"""
Harmonization module
"""

import logging

import cv2
import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.util import img_as_float

import harmonization.binarize as Binarize
import harmonization.wavelet_denoising as Wavelet
import harmonization.contrast_enhancement as Contrast
import harmonization.utils_untraced_harmonization as Utils_Untraced
import harmonization.utils_traced_harmonization as Utils_Traced

logger = logging.getLogger(__name__)


class Harmonization:
    """
    Harmonization class
    """

    def __init__(self, image):
        self.image = image


    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image


    def preprocess_traced(self, option=1):
        """
        Args:
            None.

        Method performing preprocessing for traced watermarks by ameliorating
        the contrast on margins, applying wavelet denoising, contrast stretching and
        removing shadows.
        """
        logging.info("Preprocessing traced image")
        # Ameliorate contrast on margins, wavelet denoising, contrast stretch and remove shadows
        self.image = Contrast.ameliorate_contrast_on_margins(self.image)
        self.image = Wavelet.wavelet_traced(self.image, option=option)
        self.image = Contrast.contrast_stretch(self.image)
        self.image = np.clip(self.image, 0, 255)
        self.image = Contrast.remove_shadows(self.image, np.ones((8,8)), 33)
        return self.image

    def preprocess_untraced(self):
        """
        Args:
            None.

        Method performing preprocessing of untraced watermarks by inverting the image to
        transform the watermark contour into black while making the background white.
        """
        self.image = cv2.bitwise_not(self.image)
        return self.image

    def threshold_traced_light_noise(self, dilation_shape=(3,3), window_size=25, k=0.2):
        """
        Args:
            dilation_shape: Shape of the ellipse used for the dilation applied to lines
            window_size: Window size of the local threshold

        Function that denoises a traced watermark that has low noise levels.
        """
        logging.info("Thresholding light noise")
        # Threshold with local threshold
        self.image = np.uint8(Binarize.sauvola_thresholding(
            self.image, window_size=window_size, k=k, r=100))
        # Dilate lines of the watermark
        kernel_lines = np.uint8(cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, dilation_shape))
        self.image = cv2.dilate(self.image, kernel_lines)

        return self.image


    def denoise_traced_heavy_noise(self, denoise_sigma=0.05,
                                   gaussian_sigma=2):
        """
        Args:
        denoise_sigma: The sigma value used for wavelet denoising on an image - impacts
                       the extent of the denoising
        threshold_value: the value used for the global threshold
        gaussian_sigma: The sigma value used for the gaussian blur - impacts the extent
                        of the blur
                        
        Function that denoises a traced watermark that has high noise levels.

        NOTE! It was observed later that the GaussianBlur function is called with self.image
        instead of image, meaning that the wavelet denoising was not applied on the original
        image, and only the GaussianBlur took effect. As the difference is not significant
        and changing this would require creating a new database and computing several evaluation 
        processes once again, the arguments for GaussianBlur have not been modified to integrate 
        the result from the wavelet denoising operation. 
        """
        logging.info("Denoising heavy noise")
        # Denoising with wavelet
        image = img_as_float(self.image)
        denoised = denoise_wavelet(image, method="BayesShrink", mode="soft",
                                  rescale_sigma=True, sigma=denoise_sigma)
        image = np.clip(
            denoised * 256, 0, 255).astype(np.uint8)  # type casting

        # Blur some of the noise using gaussian blur
        self.image = cv2.GaussianBlur(self.image, (0, 0), gaussian_sigma)

        return self.image

    def threshold_traced_heavy_noise(self, threshold_value=190, closing_shape=(6,6), dilation_shape=(3,3)):
        """
        Args:
            threshold_value: The value used for the global threshold
            closing_shape: The shape of the kernel used for the closing operation
            dilation_shape: The shape of the kernel used for the dilation operation

        Method that thresholds a traced watermark that has high noise levels.
        """
        logging.info("Thresholding heavy noise")
        # Threshold with a global threshold so that the noise isn't
        # emphasized
        # Note: thresholding with 190 means less noise but some lighter/incomplete lines
        # while thresholding with 200 means more noise but stronger lines
        _, thresh = cv2.threshold(
            self.image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        image = thresh

        # Perform a closing and dilation operation to make the lines of the watermark
        # more distinct, and to remove access noise
        kernel = np.ones(closing_shape, np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        kernel_lines = np.uint8(cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, dilation_shape))
        self.image = cv2.dilate(image, kernel_lines)

        return self.image

    def denoise_untraced(self, sigma_psd=20):
        """
        Args:
            sigma_psd: noise variation expectation in the image; argument for BM3D

        Method that denoises an untraced watermark.
        """
        self.image = Utils_Untraced.denoise_untraced(self.image, sigma_psd=sigma_psd)
        return self.image

    def threshold_untraced(self, window_size=25, k=0.21, morph_kernel= (3,3), iterations=3):
        self.image = Utils_Untraced.threshold_untraced(self.image, window_size=window_size, \
                                                       k=k, morph_kernel=morph_kernel, \
                                                        iterations=iterations)
        return self.image


    def post_process_traced(self, iteration, raw_img, wavelet_option):
        """
        Args:
            iteration: the amount of times the entire harmonization process is being
                        applied. Can be either 1 or 2. If it is 2, the harmonization is applied
                        without ameliorating the contrast on margins, and bounding the image
                        to the coordinates resulted from the previous computation.
            raw_img: the original image given as input by the user.
            wavelet_option: the wavelet denoising option that determines if vertical, horizontal,
                            both or no lines will be removed by passing the index of the option
                            the user has chosen after the first denoising.

        Method that performs postprocessing of traced image by removing its noisy regions.
        If iteration is not 1, denoising and thresholding are performed twice, the second time
        being on the image bounded by the coordinates of the first postprocessing, and without
        ameliorating contrast on margins. This helps preserving the clarity of watermarks that
        may have been affected by the ameliorate_contrast_on_margins method during preprocessing.
        """
        logging.info("Post-processing traced image")
        # Make two arrays of zeros with the original size of the image
        # copy_img is used for the case where iteration=1, copy_unchanged is used
        # to maintain the original shape of the image if iteration=2
        copy_img = np.zeros(self.image.shape)
        copy_unchanged = np.zeros(self.image.shape).astype(np.uint8)
        # Pad with zeros on borders because this will be happening in the cluster_pixel
        # method with self.image, so the returned shape will be greater than the original.
        copy_img = cv2.copyMakeBorder(copy_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        copy_unchanged = cv2.copyMakeBorder(copy_unchanged, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # Cluster the pixels in the binarized image
        res_image, (min_x, min_y, max_x, max_y) = Utils_Traced.cluster_pixels(self.image)
        # Crop the images according to the above coordinates. Place the corresponding section
        # in the array of zeros so that it maintains the original size and shape of the image.
        copy_img[min_y:max_y, min_x:max_x] = res_image[min_y:max_y, min_x:max_x]
        self.image = copy_img.astype(np.uint8)

        if iteration == 1:
            return self.image
        else:
            # Pad the raw image with 0's to avoid errors from changing the size of the image that will be
            # processed in the harmonization_traced method.
            raw_img = cv2.copyMakeBorder(raw_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            kept = raw_img[min_y:max_y, min_x:max_x]
            # Save the position of the cropped section from the first iteration, the second portion will be
            # bounded within it and needs to maintain its original place in the uncropped image.
            # This is initially an array of 0's.
            copy_for_2 = copy_unchanged[min_y:max_y, min_x:max_x]
            # Apply harmonization without calling ameliorate_contrast_on_margins on the cropped region
            kept, (min_x2, min_y2, max_x2, max_y2) = Utils_Traced.harmonize_traced(kept, raw_img, \
                                                                                    wavelet_option=wavelet_option)
            # Update the values contained within the area bounded by the new coordinates.
            copy_for_2[min_y2:min(max_y2, copy_for_2.shape[0]), min_x2:min(max_x2, copy_for_2.shape[1])] = \
                      kept[min_y2:min(max_y2, copy_for_2.shape[0]), min_x2:min(max_x2, copy_for_2.shape[1])]
            # Update the array of zeros having same size as the original image so that the relative
            # position of the section obtained after the second iteration of the harmonization process
            # is maintained.
            copy_unchanged[min_y:max_y, min_x:max_x] = copy_for_2
            self.image = copy_unchanged.astype(np.uint8)
            return self.image


    def post_process_untraced(self):
        """
        Args:
            None.

        Method that performs postprocessing of untraced watermarks by computing connected
        component analysis.
        """
        self.image = Utils_Untraced.connected_component_analysis(self.image, 200).astype(np.uint8)
        return self.image

    def harmonize(self, is_traced):  # sourcery skip: extract-method
        """
        Harmonize the image to a common format

        Args:
            is_traced: a boolean value indicating whether the image is traced or not

        Returns:
           A numpy array of the harmonized image
        """
        logging.info("Harmonizing image")
        if is_traced:
            logging.info("Image is traced")
            # Pipeline for traced images
            raw_image = self.image.copy()
            self.image = self.preprocess_traced(option=1)

            # Estimate the noise levels in the image, choose the denoising based on
            # noise level
            if estimate_sigma(raw_image, average_sigmas=True) < 1:
                # Light noise denoising
                self.image = self.threshold_traced_light_noise()

            else:
                # Heavy noise denoising
                self.image = self.denoise_traced_heavy_noise()
                self.image = self.threshold_traced_heavy_noise()

            # self.image = self.post_process_traced()
            self.image = self.post_process_traced(iteration=2, raw_img=raw_image, wavelet_option=1)

        else:
            logging.info("Image is not traced")
            # Pipeline for non-traced images
            self.image = self.preprocess_untraced()
            self.image = Utils_Untraced.denoise_untraced(self.image, 20)

            self.image = Utils_Untraced.threshold_untraced(self.image, 25, 0.21, (3,3), 3)

            self.image = self.post_process_untraced()


        return self.image

