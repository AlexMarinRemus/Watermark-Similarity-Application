"""
This file contains the class that tests all sharpening methods.
Sharpening methods can be found in 'harmonization/sharpen.py'.
"""

import pytest
import numpy as np

import harmonization.sharpen as Sharpen

class TestSharpen:
    """
    Class that tests all sharpening methods.
    """

    @pytest.fixture(scope="session", autouse=True)
    def define_vars(self):
        """
        Defines global image variable used in all tests.
        """
        pytest.img_grayscale = np.array([[0,0,0,0,0],
                                        [50,50,50,50,50],
                                        [50,255,255,255,50],
                                        [50,50,50,50,50],
                                        [0,0,0,0,0]]).astype(np.uint8)

    def test_sharpening_strong(self, define_vars):
        """
        Checks if the sharpened image matches the expected image
        An exact expected image is used here because the sharpening
        filter can be easily calculated.
        """
        sharpened_img = Sharpen.filter_sharpening_strong(pytest.img_grayscale)
        expected_img = np.array([[0,0,0,0,0],
                                [0,0,0,0,0],
                                [0,255,255,255,0],
                                [0,0,0,0,0],
                                [0,0,0,0,0]]).astype(np.uint8)
        assert (sharpened_img == expected_img).all(), sharpened_img

    def test_sharpening_light(self, define_vars):
        """
        Checks if the sharpened image matches the expected image
        An exact expected image is used here because the sharpening
        filter can be easily calculated.
        """
        sharpened_img = Sharpen.filter_sharpening_light(pytest.img_grayscale)
        expected_img = np.array([[0,0,0,0,0],
                                [100,0,0,0,100],
                                [0,255,255,255,0],
                                [100,0,0,0,100],
                                [0,0,0,0,0]]).astype(np.uint8)
        assert (sharpened_img == expected_img).all(), sharpened_img

    def test_unsharp_gaussian(self, define_vars):
        """
        Checks if the sharpened image matches certain expectations
        1. The whitest parts stay white
        2. The darkest parts stay dark
        3. The faded edges around the darkest parts change
        """
        sharpened_img = Sharpen.unsharp_masking_gaussian(pytest.img_grayscale)
        assert (sharpened_img[0] == [0,0,0,0,0]).all()
        assert (sharpened_img[4] == [0,0,0,0,0]).all()

        assert (sharpened_img[2][1:4] == [255,255,255]).all()

        assert (sharpened_img[1] != [50,50,50,50,50]).any()
        assert (sharpened_img[3] != [50,50,50,50,50]).any()

    def test_unsharp_median(self, define_vars):
        """
        Asserts under the same assumptions as the unsharp gaussian test
        """
        sharpened_img = Sharpen.unsharp_masking_median(pytest.img_grayscale)
        assert (sharpened_img[0] == [0,0,0,0,0]).all()
        assert (sharpened_img[4] == [0,0,0,0,0]).all()

        assert (sharpened_img[2][1:4] == [255,255,255]).all()

        assert (sharpened_img[1] != [50,50,50,50,50]).any()
        assert (sharpened_img[3] != [50,50,50,50,50]).any()

    def test_unsharp_laplacian(self, define_vars):
        """
        Asserts under the same assumptions as the unsharp gaussian test
        """
        sharpened_img = Sharpen.unsharp_masking_laplacian(pytest.img_grayscale)
        assert (sharpened_img[0] == [0,0,0,0,0]).all()
        assert (sharpened_img[4] == [0,0,0,0,0]).all()

        assert (sharpened_img[2][1:4] == [255,255,255]).all()

        assert (sharpened_img[1] != [50,50,50,50,50]).any()
        assert (sharpened_img[3] != [50,50,50,50,50]).any()





if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
