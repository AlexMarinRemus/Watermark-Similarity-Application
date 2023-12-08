"""
This file contains the class that tests all denoising methods.
Denoising methods can be found in 'harmonization/utils_untraced_harmonization.py'.
"""

import pytest
import numpy as np
import cv2

import harmonization.utils_untraced_harmonization as Utils_Untraced
import harmonization.binarize as Binarize
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


class TestUtilsUntracedHarmonization:
    """
    Class that tests all denoising methods for untraced watermarks.
    """
    @pytest.fixture(scope="session", autouse=True)
    def define_vars(self):
        """
        Defines global image variable used in all tests.
        """
        pytest.img_grayscale = cv2.imread("./testing//harmonization/1_1.jpg", 0)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_wavelet_remove_all_lines_with_vertical_line(self):
        """
        Test that the general all lines wavelet denoising denoises vertical lines
        in an image, such that the line becomes closer to the background intensity.
        """

        input_img = np.array([[180, 180, 180, 180, 180],
                            [180, 180, 50, 180, 180],
                            [180, 180, 50, 180, 180],
                            [180, 180, 50, 180, 180],
                            [180, 180, 180, 180, 180],
                            ]).astype(np.uint8)
        denoised_img = Utils_Untraced.wavelet_remove_all_lines(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Assert all of the pixels are above 100, meaning that the
        # darker line has been lightened to some extent
        assert (denoised_img > 100).all()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_wavelet_remove_all_lines_with_horizontal_line(self):
        """
        Test that the general all lines wavelet denoising denoises horizontal
        lines.
        """

        input_img = np.array([[180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 50, 50, 50, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            ]).astype(np.uint8)
        denoised_img = Utils_Untraced.wavelet_remove_all_lines(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Note: these assertions work on a 6x6 matrix because the wavelet transform
        # appends another column and row to the denoised image

        # Assert that where the horizontal line is is no longer dark, meaning that
        # the horizontal lines are removed and become lighter, as
        # expected
        assert (denoised_img[2][1:4] >= 100).all()

        # Assert that everywhere else is larger than 100, meaning that the
        # rest of the image is still relatively the same as the input, as
        # expected
        assert (denoised_img[0:2] > 100).all()
        assert (denoised_img[3:6] > 100).all()
        assert denoised_img[2][0] > 100 and denoised_img[2][4] > 100

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_wavelet_remove_all_lines_vertical_homogeneous_image(self):
        """
        Test that when the general all lines wavelet denoising is applied to an image
        that is homogeneous, that all of the values stay roughly within the
        same range.
        """

        input_img = np.array([[180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            ]).astype(np.uint8)
        denoised_img = Utils_Untraced.wavelet_remove_all_lines(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        assert (denoised_img < 190).all()
        assert (denoised_img > 150).all()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_wavelet_remove_all_lines_no_line_in_image(self):
        """
        Test that when the general all lines wavelet denoising is applied to an image
        that has no lines, that all of the values stay roughly within the
        same range as in the input.
        """

        input_img = np.array([[180, 180, 180, 180],
                            [180, 180, 50, 180],
                            [180, 180, 180, 180],
                            [180, 180, 180, 180]
                            ]).astype(np.uint8)
        denoised_img = Utils_Untraced.wavelet_remove_all_lines(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Assert that the dark dot gets blurred and lighter
        assert denoised_img[1, 2] > 100
        # Assert that the light parts stay light
        assert (denoised_img[1,3] > 100).all()
        assert (denoised_img[1,0:2] > 100).all()
        assert (denoised_img[0] > 100).all()
        assert (denoised_img[2] > 100).all()
        assert (denoised_img[3] > 100).all()
        
    def test_is_binarized_thresholding(self, define_vars):
        """
        Checks if all values in binarized image are either black (0) or white (255).
        """

        binarized_img = Utils_Untraced.threshold_untraced(pytest.img_grayscale, window_size=3)
        assert all(px in (0, 255)for px in binarized_img.flatten())

    def test_thresholding_window_size(self, define_vars):
        """
        Tests that the window size doesn't impact size of the image.
        """

        binarized_img_small_window = Utils_Untraced.threshold_untraced(pytest.img_grayscale, \
            window_size=5)
        binarized_sauvola_small_window = Binarize.sauvola_thresholding(pytest.img_grayscale, \
            window_size=5)
        binarized_sauvola_small_window = list(binarized_sauvola_small_window.flatten())
        binarized_img_small_window = list(binarized_img_small_window.flatten())
        # Checks that the thresholding also has the morphological closing
        assert np.all(binarized_sauvola_small_window<=binarized_img_small_window)
        binarized_img_large_window = Utils_Untraced.threshold_untraced(pytest.img_grayscale, \
            window_size=25)
        binarized_img_large_window = list(binarized_img_large_window.flatten())
        binarized_sauvola_large_window = Binarize.sauvola_thresholding(pytest.img_grayscale, \
            window_size=25)
        binarized_sauvola_large_window = list(binarized_sauvola_large_window.flatten())
        # Checks that the thresholding also has the morphological closing
        assert np.all(binarized_sauvola_large_window<=binarized_img_large_window)
        # Ensures that different window sizes still binarize to the same image size
        assert len(binarized_img_large_window) == len(binarized_img_small_window)
        assert np.any(binarized_img_large_window!=binarized_img_small_window)

    def test_thresholding_k(self, define_vars):
        """
        Tests that the k constant changes the count of white pixels as expected
        using the thresholding formula.
        """

        # Thresholds with small k value and counts white pixels
        binarized_img_small_k = Utils_Untraced.threshold_untraced(pytest.img_grayscale, k=0.1)
        binarized_img_small_k = list(binarized_img_small_k.flatten())
        binarized_sauvola_small_k = Binarize.sauvola_thresholding(pytest.img_grayscale, k=0.1)
        binarized_sauvola_small_k = list(binarized_sauvola_small_k.flatten())
        count_white_pixels_small_k = binarized_img_small_k.count(255)
        # Checks that the thresholding also has the morphological closing
        assert np.all(binarized_sauvola_small_k<=binarized_img_small_k)
        # Thresholds with larger k value and counts white pixels
        binarized_img_large_k = Utils_Untraced.threshold_untraced(pytest.img_grayscale, k=0.5)
        binarized_img_large_k = list(binarized_img_large_k.flatten())
        binarized_sauvola_large_k = Binarize.sauvola_thresholding(pytest.img_grayscale, k=0.5)
        binarized_sauvola_large_k = list(binarized_sauvola_large_k.flatten())
        count_white_pixels_large_k = binarized_img_large_k.count(255)
        # Checks that the thresholding also has the morphological closing
        assert np.all(binarized_sauvola_large_k<=binarized_img_large_k)
        # Ensures that different window sizes still binarize to the same image size
        assert len(binarized_img_large_k) == len(binarized_img_small_k)
        # Asserts that smaller values of k lead to more white pixels
        assert count_white_pixels_small_k >= count_white_pixels_large_k
        assert np.any(binarized_img_large_k!=binarized_img_small_k)
        
    def test_denoising(self, define_vars):
        """
        Tests that the denoising is of satisfactory quality using the PSNR and SSIM metric
        The higher this metric the better. A score of above 20 is considered satisfactory and a score of 30 or more
        is considered optimal. Similarly, 87% of structural similarity (SSIM) is great. We expect denoising with higher noise
        variation expected to produce smoother images and hence better scores for the 2 metrics
        """
        # Denoise the image and resize to fit the original dimensions
        denoised_img_big_noise = cv2.resize(Utils_Untraced.denoise_untraced(pytest.img_grayscale, 35), \
                                            (pytest.img_grayscale.shape[1], pytest.img_grayscale.shape[0]))
        denoised_img_small_noise = cv2.resize(Utils_Untraced.denoise_untraced(pytest.img_grayscale, 21), \
                                              (pytest.img_grayscale.shape[1], pytest.img_grayscale.shape[0]))
        psnr_big_noise = psnr(pytest.img_grayscale, denoised_img_big_noise.astype(np.uint8))
        psnr_small_noise = psnr(pytest.img_grayscale, denoised_img_small_noise.astype(np.uint8))
        ssim_big_noise = ssim(pytest.img_grayscale, denoised_img_big_noise.astype(np.uint8))
        ssim_small_noise = ssim(pytest.img_grayscale, denoised_img_small_noise.astype(np.uint8))
        assert ssim_big_noise != ssim_small_noise
        assert psnr_big_noise != psnr_small_noise
        assert psnr_big_noise >= psnr_small_noise >= 20
        assert ssim_big_noise >= ssim_small_noise >= 0.87


        # Connected component analysis tests
    
    def test_connected_component_analysis_3(self):
        """
        Tests that an image with small cluster has them removed
        """
        input_img = np.array([[255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
                            [255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]]).astype(np.uint8)
        expected_img = np.array([[255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
        processed_img = Utils_Untraced.connected_component_analysis(input_img, min_size= 3)
        assert np.all(processed_img == expected_img)
        
    def test_connected_component_analysis_1(self):
        """
        Tests that an image with only 1 small cluster has it removed
        """
        input_img = np.array([[255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
                            [255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]]).astype(np.uint8)
        expected_img = np.array([[255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
                            [255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
        processed_img = Utils_Untraced.connected_component_analysis(input_img, min_size= 2)
        assert np.all(processed_img == expected_img)
        
    def test_connected_component_analysis_nothing_left(self):
        """
        Tests that an image with only too big threshold everything gets removed
        """
        input_img = np.array([[255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
                            [255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]]).astype(np.uint8)
        expected_img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
        processed_img = Utils_Untraced.connected_component_analysis(input_img, min_size= 5)
        assert np.all(processed_img == expected_img)
    
    def test_connected_component_analysis_nothing_changed(self):
        """
        Tests that an image with a threshold of only 1 everything stays
        """
        input_img = np.array([[255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
                            [255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]]).astype(np.uint8)
        processed_img = Utils_Untraced.connected_component_analysis(input_img, min_size= 1)
        assert np.all(processed_img == input_img)
    

if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
