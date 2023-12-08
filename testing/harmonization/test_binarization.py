"""
This file contains the class that tests all binarization methods.
Binarization methods can be found in 'harmonization/binarize.py'.
"""

import pytest
import numpy as np

import harmonization.binarize as Binarize

class TestBinarization:
    """
    Class that tests all binarization methods.
    """

    @pytest.fixture(scope="session", autouse=True)
    def define_vars(self):
        """
        Defines global image variable used in all tests.
        """
        # Note: the values of the grayscale image have been randomized
        pytest.img_grayscale = np.array([[161, 80, 187, 47, 138, 192, 245, 77, 19, 203],
                                        [72, 112, 160, 229, 40, 132, 57, 185, 163, 198],
                                        [7, 56, 238, 75, 111, 104, 144, 26, 199, 90],
                                        [45, 5, 1, 145, 72, 197, 108, 90, 18, 0],
                                        [98, 82, 56, 237, 90, 79, 49, 164, 20, 252],
                                        [224, 226, 130, 4, 228, 48, 30, 42, 12, 217],
                                        [58, 118, 203, 72, 210, 66, 240, 252, 20, 255],
                                        [60, 222, 207, 104, 234, 189, 61, 233, 107, 173],
                                        [139, 125, 131, 155, 44, 150, 206, 96, 137, 173],
                                        [96, 164, 241, 67, 212, 132, 148, 7, 196, 255]])

    # Test Niblack

    def test_is_binarized_niblack(self, define_vars):
        """
        Checks if all values in binarized image are either black (0) or white (255).
        """

        binarized_img = Binarize.niblack_thresholding(pytest.img_grayscale, window_size=3)
        assert all(px in (0, 255) for px in binarized_img.flatten())

    def test_niblack_window_size(self, define_vars):
        """
        Tests that the window size doesn't impact size of the image.
        """

        binarized_img_small_window = Binarize.niblack_thresholding(pytest.img_grayscale, \
            window_size=3)
        binarized_img_small_window = list(binarized_img_small_window.flatten())

        binarized_img_large_window = Binarize.niblack_thresholding(pytest.img_grayscale, \
            window_size=5)
        binarized_img_large_window = list(binarized_img_large_window.flatten())

        # Ensures that different window sizes still binarize to the same image size
        assert len(binarized_img_large_window) == len(binarized_img_small_window)
        # Checks that the images are not the same
        assert binarized_img_small_window != binarized_img_large_window

    def test_niblack_k(self, define_vars):
        """
        Tests that the k constant changes the count of white pixels as expected
        using the niblack thresholding formula.
        """

        # Thresholds with small k value and counts white pixels
        binarized_img_small_k = Binarize.niblack_thresholding(pytest.img_grayscale, k=0.1)
        binarized_img_small_k = list(binarized_img_small_k.flatten())
        count_white_pixels_small_k = binarized_img_small_k.count(255)

        # Thresholds with larger k value and counts white pixels
        binarized_img_large_k = Binarize.niblack_thresholding(pytest.img_grayscale, k=0.5)
        binarized_img_large_k = list(binarized_img_large_k.flatten())
        count_white_pixels_large_k = binarized_img_large_k.count(255)

        # Ensures that different window sizes still binarize to the same image size
        assert len(binarized_img_large_k) == len(binarized_img_small_k)
        # Asserts that smaller values of k lead to more white pixels
        assert count_white_pixels_small_k >= count_white_pixels_large_k

    # Test Sauvola

    def test_is_binarized_sauvola(self, define_vars):
        """
        Checks if all values in binarized image are either black (0) or white (255).
        """

        binarized_img = Binarize.sauvola_thresholding(pytest.img_grayscale, window_size=3)
        assert all(px in (0, 255)for px in binarized_img.flatten())

    def test_sauvola_window_size(self, define_vars):
        """
        Tests that the window size doesn't impact size of the image.
        """

        binarized_img_small_window = Binarize.sauvola_thresholding(pytest.img_grayscale, \
            window_size=3)
        binarized_img_small_window = list(binarized_img_small_window.flatten())

        binarized_img_large_window = Binarize.sauvola_thresholding(pytest.img_grayscale, \
            window_size=5)
        binarized_img_large_window = list(binarized_img_large_window.flatten())

        # Ensures that different window sizes still binarize to the same image size
        assert len(binarized_img_large_window) == len(binarized_img_small_window)
        # Checks that the images are not the same
        assert binarized_img_small_window != binarized_img_large_window

    def test_sauvola_k(self, define_vars):
        """
        Tests that the k constant changes the count of white pixels as expected
        using the sauvola thresholding formula.
        """

        # Thresholds with small k value and counts white pixels
        binarized_img_small_k = Binarize.sauvola_thresholding(pytest.img_grayscale, k=0.1)
        binarized_img_small_k = list(binarized_img_small_k.flatten())
        count_white_pixels_small_k = binarized_img_small_k.count(255)

        # Thresholds with larger k value and counts white pixels
        binarized_img_large_k = Binarize.sauvola_thresholding(pytest.img_grayscale, k=0.5)
        binarized_img_large_k = list(binarized_img_large_k.flatten())
        count_white_pixels_large_k = binarized_img_large_k.count(255)

        # Ensures that different window sizes still binarize to the same image size
        assert len(binarized_img_large_k) == len(binarized_img_small_k)
        # Asserts that smaller values of k lead to more white pixels
        assert count_white_pixels_small_k >= count_white_pixels_large_k

    def test_sauvola_r(self, define_vars):
        """
        Tests that the r constant changes the count of white pixels as expected
        using the sauvola thresholding formula.
        """

        # Thresholds with small r value and counts white pixels
        binarized_img_small_r = Binarize.sauvola_thresholding(pytest.img_grayscale, r=20)
        binarized_img_small_r = list(binarized_img_small_r.flatten())
        count_white_pixels_small_r = binarized_img_small_r.count(255)

        # Thresholds with larger r value and counts white pixels
        binarized_img_large_r = Binarize.sauvola_thresholding(pytest.img_grayscale, r=50)
        binarized_img_large_r = list(binarized_img_large_r.flatten())
        count_white_pixels_large_r = binarized_img_large_r.count(255)

        # Ensures that different window sizes still binarize to the same image size
        assert len(binarized_img_large_r) == len(binarized_img_small_r)
        # Asserts that smaller values of k lead to more white pixels
        assert count_white_pixels_small_r >= count_white_pixels_large_r

    # Test Otsu

    def test_is_binarized_otsu(self, define_vars):
        """
        Checks if all values in binarized image are either black (0) or white (255).
        """

        binarized_img = Binarize.otsu_thresholding(pytest.img_grayscale)
        assert all(px in (0, 255) for px in binarized_img.flatten())

    # Test Entropy
    # Note: These tests filter out user warnings since they kept being flagged
    # for a certain conversion warning that couldn't be found and wasn't leading
    # to any problems.

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_is_binarized_entropy(self, define_vars):
        """
        Checks if all values in binarized image are either black (0) or white (255).
        """

        binarized_img = Binarize.entropy_thresholding(pytest.img_grayscale)
        assert all(px in (0, 255) for px in binarized_img.flatten())

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_entropy_radius(self, define_vars):
        """
        Checks if changing radius doesn't change the image size.
        """

        # Thresholds with radius three and counts white pixels
        binarized_img_small_radius = Binarize.entropy_thresholding(pytest.img_grayscale, \
            radius=0.9)
        binarized_img_small_radius = list(binarized_img_small_radius.flatten())

        # Thresholds with radius size five and counts white pixels
        binarized_img_large_radius = Binarize.entropy_thresholding(pytest.img_grayscale, \
            radius=2)
        binarized_img_large_radius = list(binarized_img_large_radius.flatten())

        # Ensures that different radius sizes still binarize to the same image size
        assert len(binarized_img_large_radius) == len(binarized_img_small_radius)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_entropy_threshold(self, define_vars):
        """
        Check that different thresholds result in expected difference in white pixels.
        """

        # Thresholds with small threshold value and counts white pixels
        binarized_img_small_thresh = Binarize.entropy_thresholding(pytest.img_grayscale, \
             threshold=50)
        binarized_img_small_thresh = list(binarized_img_small_thresh.flatten())
        count_white_pixels_small_thresh = binarized_img_small_thresh.count(255)

        # Thresholds with larger threshold value and counts white pixels
        binarized_img_large_thresh = Binarize.entropy_thresholding(pytest.img_grayscale, \
             threshold=100)
        binarized_img_large_thresh = list(binarized_img_large_thresh.flatten())
        count_white_pixels_large_thresh = binarized_img_large_thresh.count(255)

        # Ensures that different thresholds still binarize to the same image size
        assert len(binarized_img_large_thresh) == len(binarized_img_small_thresh)
        # Asserts that smaller thresholds lead to more white pixels
        assert count_white_pixels_small_thresh >= count_white_pixels_large_thresh


if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
