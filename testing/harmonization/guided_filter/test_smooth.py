"""
This file contains the class that tests the box filter smoothing method.
Denoising methods can be found in 'harmonization/guided_filter/smooth.py'.
"""

import pytest
import numpy as np
import unittest

import harmonization.guided_filter.smooth as Smooth
class TestSmooth(unittest.TestCase):
    """
    Class that tests the box filter smoothing method.
    """
    @pytest.fixture(scope="session", autouse=True)
    def define_vars(self):
        """
        Defines global image variable used in all tests.
        """
        pytest.img_grayscale = np.array([[50, 50, 50, 50, 50],
                                        [50, 150, 150, 150, 50], 
                                        [50, 150, 255, 150, 50],
                                        [50, 150, 150, 150, 50],
                                        [50, 50, 50, 50, 50]]).astype(np.uint8)

    def test_box_filter_zero(self):
        """
        Test that the zero box filtering works as anticipated
        """
        filtered = Smooth.box_filter(pytest.img_grayscale, 2, border_type="zero").astype(np.uint8)
        print(type(filtered))
        print(type(pytest.img_grayscale))
        # Assert the image is filtered correctly
        assert (filtered < 95).all()
        assert not (filtered > 50).all()
    def test_box_filter_reflect(self):
        """
        Test that the reflect box filtering works as anticipated
        """
        filtered = Smooth.box_filter(pytest.img_grayscale, 2, border_type="reflect").astype(np.uint8)
        # Assert the image is filtered correctly
        assert (filtered < 95).all()
        assert (filtered > 70).all()
        assert (filtered[0] <= 90).all()
        
    def test_box_filter_reflect_101(self):
        """
        Test that the reflect 101 box filtering works as anticipated
        """
        filtered = Smooth.box_filter(pytest.img_grayscale, 2, border_type="reflect_101").astype(np.uint8)
        # Assert the image is filtered correctly
        assert (filtered >= 90).all()
        assert not (filtered[0] <= 90).all()
        
    def test_box_filter_edge(self):
        """
        Test that the edge box filtering works as anticipated
        """
        filtered = Smooth.box_filter(pytest.img_grayscale, 2, border_type="edge").astype(np.uint8)
        print(filtered)
        # Assert the image is filtered correctly
        assert (filtered < 100).all()
        assert (filtered > 50).all()
        assert not (filtered > 70).all()
        
    def test_box_filter_notimpl(self):
        """
        Test that the non-implemented filtering raises errors
        """
        with pytest.raises(NotImplementedError):
            Smooth.box_filter(pytest.img_grayscale, 2, border_type="else").astype(np.uint8)

if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
