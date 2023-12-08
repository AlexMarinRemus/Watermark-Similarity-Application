"""
This file contains the class that tests all denoising methods.
Denoising methods can be found in 'harmonization/wavelet_denoising.py'.
"""

import pytest
import numpy as np
from skimage.restoration import estimate_sigma
from unittest.mock import patch, mock_open, call, Mock

import harmonization.wavelet_denoising as Denoise

class TestDenoising:
    """
    Class that tests all denoising methods.
    """

    def test_wavelet_denoising_vertical_image_with_vertical_line(self):
        """
        Test that the vertical line wavelet denoising denoises vertical lines
        in an image, such that the line becomes closer to the background intensity.
        """

        input_img = np.array([[180, 180, 180, 180, 180],
                            [180, 180, 50, 180, 180],
                            [180, 180, 50, 180, 180],
                            [180, 180, 50, 180, 180],
                            [180, 180, 180, 180, 180],
                            ]).astype(np.uint8)
        denoised_img = Denoise.wavelet_fourier_vertical(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Assert all of the pixels are above 100, meaning that the
        # darker line has been lightened to some extent
        assert (denoised_img > 100).all()

    def test_wavelet_denoising_vertical_image_with_horizontal_line(self):
        """
        Test that the vertical line wavelet denoising doesn't denoise horizontal
        lines.
        """

        input_img = np.array([[180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 50, 50, 50, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            ]).astype(np.uint8)
        denoised_img = Denoise.wavelet_fourier_vertical(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Note: these assertions work on a 6x6 matrix because the wavelet transform
        # appends another column and row to the denoised image

        # Assert that where the horizontal line is is still dark, meaning that
        # the vertical line denoising doesn't remove horizontal lines, as
        # expected
        assert (denoised_img[2][1:4] < 100).all()

        # Assert that everywhere else is larger than 100, meaning that the
        # rest of the image is still relatively the same as the input, as
        # expected
        assert (denoised_img[0:2] > 100).all()
        assert (denoised_img[3:6] > 100).all()
        assert denoised_img[2][0] > 100 and denoised_img[2][4] > 100

    def test_wavelet_denoising_vertical_homogeneous_image(self):
        """
        Test that when the vertical line denoising is applied to an image
        that is homogeneous, that all of the values stay roughly within the
        same range.
        """

        input_img = np.array([[180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            ]).astype(np.uint8)
        denoised_img = Denoise.wavelet_fourier_vertical(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        assert (denoised_img < 190).all()
        assert (denoised_img > 150).all()

    def test_wavelet_denoising_vertical_no_line_in_image(self):
        """
        Test that when the vertical line denoising is applied to an image
        that has no lines, that all of the values stay roughly within the
        same range as in the input.
        """

        input_img = np.array([[180, 180, 180, 180],
                            [180, 180, 50, 180],
                            [180, 180, 180, 180],
                            [180, 180, 180, 180]
                            ]).astype(np.uint8)
        denoised_img = Denoise.wavelet_fourier_vertical(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Assert that the dark dot stays dark
        assert denoised_img[1, 2] < 100
        # Assert that the light parts stay light
        assert (denoised_img[1,3] > 100).all()
        assert (denoised_img[1,0:2] > 100).all()
        assert (denoised_img[0] > 100).all()
        assert (denoised_img[2] > 100).all()
        assert (denoised_img[3] > 100).all()

    def test_wavelet_denoising_horizontal_image_with_horizontal_line(self):
        """
        Test that the horizontal line wavelet denoising denoises horizontal lines
        in an image, such that the line becomes closer to the background intensity.
        """

        input_img = np.array([[180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 50, 50, 50, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            ]).astype(np.uint8)
        denoised_img = Denoise.wavelet_fourier_horizontal(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Assert all of the pixels are above 100, meaning that the
        # darker line has been lightened to some extent
        assert (denoised_img > 100).all()
    
    def test_wavelet_denoising_horizontal_image_with_vertical_line(self):
        """
        Test that the horizontal line wavelet denoising doesn't denoise vertical
        lines.
        """

        input_img = np.array([[180, 180, 180, 180, 180],
                            [180, 180, 50, 180, 180],
                            [180, 180, 50, 180, 180],
                            [180, 180, 50, 180, 180],
                            [180, 180, 180, 180, 180],
                            ]).astype(np.uint8)
        denoised_img = Denoise.wavelet_fourier_horizontal(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Note: these assertions work on a 6x6 matrix because the wavelet transform
        # appends another column and row to the denoised image

        # Assert that where the vertical line is is still dark, meaning that
        # the horizontal line denoising doesn't remove vertical lines, as
        # expected
        assert (denoised_img[2:5, 2] < 100).all()

        # Assert that everywhere else is larger than 100, meaning that the
        # rest of the image is still relatively the same as the input, as
        # expected
        assert (denoised_img[:, 0:2] > 100).all()
        assert (denoised_img[:, 3:6] > 100).all()
        assert (denoised_img[0:2, 2] > 100).all() and denoised_img[5, 2] > 100, denoised_img

    def test_wavelet_denoising_horizontal_homogeneous_image(self):
        """
        Test that when the horizontal line denoising is applied to an image
        that is homogeneous, that all of the values stay roughly within the
        same range.
        """

        input_img = np.array([[180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            [180, 180, 180, 180, 180],
                            ]).astype(np.uint8)
        denoised_img = Denoise.wavelet_fourier_horizontal(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        assert (denoised_img < 190).all()
        assert (denoised_img > 150).all()

    def test_wavelet_denoising_horizontal_no_line_in_image(self):
        """
        Test that when the horizontal line denoising is applied to an image
        that has no lines, that all of the values stay roughly within the
        same range as in the input.
        """

        input_img = np.array([[180, 180, 180, 180],
                            [180, 180, 50, 180],
                            [180, 180, 180, 180],
                            [180, 180, 180, 180]
                            ]).astype(np.uint8)
        denoised_img = Denoise.wavelet_fourier_horizontal(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Assert that the dark dot stays dark
        assert denoised_img[1, 2] < 100
        # Assert that the light parts stay light
        assert (denoised_img[1,3] > 100).all()
        assert (denoised_img[1,0:2] > 100).all()
        assert (denoised_img[0] > 100).all()
        assert (denoised_img[2] > 100).all()
        assert (denoised_img[3] > 100).all()

    # Test for the wavelet_denoise method.

    def test_wavelet_denoise(self):
        """
        Test that after the wavelet denoising, the image keeps its shape 
        and it becomes less noisy.
        """

        # Initialize random input image.
        input_img = np.random.rand(256, 256) * 100
        # Perform wavelet denoising on the input image.
        denoised_img = Denoise.wavelet_denoise(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Assert that the result keeps the same shape but it is less noisy.
        assert input_img.shape == denoised_img.shape
        assert estimate_sigma(input_img, average_sigmas=True) > \
               estimate_sigma(denoised_img, average_sigmas=True)
        
    # Test for the wavelet_fourier method.

    def test_wavelet_fourier(self):
        """
        Test that after denoising the vertical lines from the image are removed
        and the result is less noisy.
        """

        # Initialize a noisy image with vertical lines.
        input_img = np.array([[1, 255, 33, 255, 23, 255, 1],
                              [3, 255, 40, 255, 18, 255, 5],
                              [7, 255, 22, 255, 33, 255, 8],
                              [6, 255, 49, 255, 19, 255, 3],
                              [2, 255, 18, 255, 11, 255, 9],
                              [0, 255, 26, 255, 43, 255, 6],
                              [7, 255, 67, 255, 12, 255, 2]]).astype(np.uint8)
        # Perform denoising for the given image.
        denoised_img = Denoise.wavelet_fourier(input_img)
        denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
        # Assert that the contrast has lowered and that the noise levels are 
        # smaller than before.
        assert (denoised_img < 175).all()
        assert estimate_sigma(input_img, average_sigmas=True) > \
                 estimate_sigma(denoised_img, average_sigmas=True)    
        
    # Tests for the wavelet_traced method.

    def test_wavelet_traced_option_1(self):
        """
        Test that if option is 1 then the method only calls the wavelet
        for vertical line removal.
        """
        
        # Initialize the input image.
        input_img = np.array([[1, 255, 33, 255, 23, 255, 1],
                              [3, 255, 40, 255, 18, 255, 5],
                              [7, 255, 22, 255, 33, 255, 8],
                              [6, 255, 49, 255, 19, 255, 3],
                              [2, 255, 18, 255, 11, 255, 9],
                              [0, 255, 26, 255, 43, 255, 6],
                              [7, 255, 67, 255, 12, 255, 2]]).astype(np.uint8)
        # Mock the methods that may be called.
        with patch.object(Denoise, "wavelet_fourier_vertical") as mock_f_vert, \
             patch.object(Denoise, "wavelet_fourier_horizontal") as mock_f_horiz, \
             patch.object(Denoise, "wavelet_denoise") as mock_wavelet_denoise:
            # Call the method and assert that it only uses vertical lines removal.
            denoised_img = Denoise.wavelet_traced(input_img, option=1)
            denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
            mock_wavelet_denoise.assert_called_once_with(input_img)
            mock_f_vert.assert_called_once_with(mock_wavelet_denoise.return_value, \
                                                levels=8, wavelet='dmey', \
                                                    sigma=2.5)
            mock_f_horiz.assert_not_called()


    def test_wavelet_traced_option_2(self):
        """
        Test that if option is 2 then the method only calls the wavelet
        for horizontal line removal.
        """
        
        # Initialize the input image.
        input_img = np.random.rand(1000, 1000) * 100
        # Mock the methods that may be called.
        with patch.object(Denoise, "wavelet_fourier_vertical") as mock_f_vert, \
             patch.object(Denoise, "wavelet_fourier_horizontal") as mock_f_horiz, \
             patch.object(Denoise, "wavelet_denoise") as mock_wavelet_denoise:
            # Call the method and assert that it only uses horizontal lines removal.
            denoised_img = Denoise.wavelet_traced(input_img, option=2)
            denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
             # Assert that just the horizontal line removal was kept together
             # with wavelet_denoise.
            mock_wavelet_denoise.assert_called_once_with(input_img)
            mock_f_horiz.assert_called_once_with(mock_wavelet_denoise.return_value, \
                                    levels = 8, wavelet='db10', sigma=3)
            mock_f_vert.assert_not_called()

    def test_wavelet_traced_option_3(self):
        """
        Test that if option is 3 then the method calls the both wavelet
        for horizontal and the vertical line removal.
        """
        
        # Initialize the input image.
        input_img = np.random.rand(1000, 1000) * 100
        # Mock the methods that may be called.
        with patch.object(Denoise, "wavelet_fourier_vertical") as mock_f_vert, \
             patch.object(Denoise, "wavelet_fourier_horizontal") as mock_f_horiz, \
             patch.object(Denoise, "wavelet_denoise") as mock_wavelet_denoise:
            # Call the method and assert that it uses both horizontal and vertical 
            # lines removal.
            denoised_img = Denoise.wavelet_traced(input_img, option=3)
            denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
             # Assert that both the horizontal and the vertical line removal were 
             # kept, and so did wavelet_denoise.
            mock_wavelet_denoise.assert_called_once_with(input_img)
            mock_f_vert.assert_called_once_with(mock_wavelet_denoise.return_value, \
                                                levels=8, wavelet='dmey', sigma=2.5)
            mock_f_horiz.assert_called_once_with(mock_f_vert.return_value, \
                                    levels = 8, wavelet='db10', sigma=2.5)

    def test_wavelet_traced_option_4(self):
        """
        Test that if option is 4 then the method calls the no wavelet
        line removal.
        """
        
        # Initialize the input image.
        input_img = np.random.rand(1000, 1000) * 100
        # Mock the methods that may be called.
        with patch.object(Denoise, "wavelet_fourier_vertical") as mock_f_vert, \
             patch.object(Denoise, "wavelet_fourier_horizontal") as mock_f_horiz, \
             patch.object(Denoise, "wavelet_denoise") as mock_wavelet_denoise:
            # Call the method and assert that it uses none of the horizontal 
            # or vertical lines removal.
            denoised_img = Denoise.wavelet_traced(input_img, option=4)
            denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
             # Assert that both the horizontal and the vertical line removal were
             # not called at all.
            mock_wavelet_denoise.assert_called_once_with(input_img)
            mock_f_vert.assert_not_called()
            mock_f_horiz.assert_not_called()


if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
