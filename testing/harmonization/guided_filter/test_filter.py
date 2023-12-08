"""
This file contains the class that tests the guided filtering methods itself.
Denoising methods can be found in 'harmonization/guided_filter/filter.py'.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, mock_open, call, Mock
from unittest import TestCase
from numpy.testing import assert_array_equal
from skimage.util import img_as_float
from skimage.restoration import denoise_wavelet, estimate_sigma

import harmonization.guided_filter.filter as GF
class TestGF:
    """
    Class that tests the guided filtering.
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

    def test_filter(self):
        """
        Test that the generic filter works as anticipated
        """
        guided_filter = GF.GuidedFilter(pytest.img_grayscale, 2, 0.05)
        target = cv2.boxFilter(src=pytest.img_grayscale, ddepth=-1, ksize=(5, 5), anchor=(-1,-1), normalize= True, borderType=cv2.BORDER_REFLECT_101)
        
        # expected = np.array([[119, 119, 119, 119, 119],
        #                     [119, 108, 108, 108, 119],
        #                     [119, 108, 96, 108, 119],
        #                     [119, 108, 108, 108, 119],
        #                     [119, 119, 119, 119, 119]]).astype(np.uint8)
        padded = guided_filter.filter(target).astype(np.uint8)
        # Assert the image is filtered correctly
        assert min(padded.flatten()) == padded[2][2]
        # assert np.all(padded.flatten() == expected.flatten())


    # Test for the init method from GuidedFilter class.

    def test_init(self):
        """
        Test that the type of guided filter is correctly assigned when 
        initializing an instance of the GuidedFilter class.
        """

        # Assert that if the shape of the image is 2, then it will be an instance
        # of the GrayGuidedFilter class.
        grayscale_img = np.zeros((5,5)).astype(np.uint8)
        guided_filter = GF.GuidedFilter(grayscale_img, 2, 3)
        assert isinstance(guided_filter._Filter, GF.GrayGuidedFilter)
        # Assert that if the shape of the image is not 2, then it will be an instance
        # of the MultiDimGuidedFilter class.
        color_img = np.zeros((5 ,5, 5)).astype(np.uint8)
        guided_filter_2 = GF.GuidedFilter(color_img, 3, 2)
        assert isinstance(guided_filter_2._Filter, GF.MultiDimGuidedFilter)

    # Test for the filter method from GuidedFilter class.

    def test_filter_guided_filter(self):
        """
        Test that the filter method of GuidedFilter class calls the 
        filter function from MultiDimGuidedFilter in the right order
        and with the right attributes.
        """

        # Initialize input color image with shape 3.
        color_img = np.zeros((5, 5, 3), dtype=np.float32)

        # Mock the filter method from MultiDimGuidedFilter class and set its
        # return values.
        with patch.object(GF.MultiDimGuidedFilter, "filter", \
                          side_effect=[np.ones_like(color_img)[:,:,0], \
                                       2*np.ones_like(color_img)[:,:,1], \
                                       3*np.ones_like(color_img)[:,:,2]]) as mock_filter:

            # Call the filter method with the 3D color_img.
            g = GF.GuidedFilter(color_img, 2, 0.03)
            res = g.filter(color_img)
            # Assert that the mocked method has been called 3 times, once for each
            # color channel, and that the attributes used in the calls are the 
            # correct ones.
            assert mock_filter.call_count == 3
            filter_calls = mock_filter.call_args_list
            assert np.array_equal(filter_calls[0][0][0], color_img[:,:,0])
            assert np.array_equal(filter_calls[1][0][0], color_img[:,:,1])
            assert np.array_equal(filter_calls[2][0][0], color_img[:,:,2])
            
            # Assert that the result contains the filtered version of the color
            # channels in the right order, and that its shape is the same as the
            # one of color_img.
            assert np.array_equal(res[:,:,0], np.ones_like(color_img)[:,:,0])
            assert np.array_equal(res[:,:,1], 2*np.ones_like(color_img)[:,:,1])
            assert np.array_equal(res[:,:,2], 3*np.ones_like(color_img[:,:,2]))
            assert np.array_equal(res.shape, color_img.shape)

    # Tests for the filter method from MultiDimGuidedFilter class.
    
    def test_filter_MultiDimGuidedFilter(self):
        """
        Test that the part of the result that is not part of an edge in the image
        gets significantly changed to match the guidance image.
        """
        
        # Initialize the input image of shape (8, 8, 3).
        input_img = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], \
                               [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], \
                               [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], \
                               [255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [23, 23, 23], [30, 30, 30], \
                               [10, 10, 10], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [20, 20, 20], [25, 25, 25], \
                               [15, 15, 15], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [25, 25, 25], [30, 30, 30], \
                               [18, 18, 18], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], \
                               [255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], \
                               [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], 
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], \
                               [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]).astype(np.float32)

        # Initialize the guiding filter to be 2D and contains similar but less variable 
        # values as the input image.
        p = np.array([[25, 25, 25, 25, 25, 25, 25, 25, 25],
                      [25, 25, 25, 25, 25, 25, 25, 25, 25],
                      [25, 25, 255, 255, 255, 255, 255, 25, 25],
                      [25, 25, 255, 50, 50, 50, 255, 25, 25],
                      [25, 25, 255, 50, 50, 50, 255, 25, 25],
                      [25, 25, 255, 50, 50, 50, 255, 25, 25],
                      [25, 25, 255, 255, 255, 255, 255, 25, 25],
                      [25, 25, 25, 25, 25, 25, 25, 25, 25], 
                      [25, 25, 25, 25, 25, 25, 25, 25, 25]]).astype(np.float32)

        # Execute the filter method.
        g = GF.MultiDimGuidedFilter(input_img, 2, 0.3) 
        res = g.filter(p)
        # Assert that the values become closer to the values of the 
        # guided filter than of those of the input image. 
        assert all (abs(p[0] - res) < abs(res - input_img[0,:,0])) and \
               all (abs(p[0] - res) < abs(res - input_img[0,:,2])) and \
               all (abs(p[0] - res) < abs(res - input_img[0,:,1]))

    def test_filter_MultiDimGuidedFilter2(self):
        """
        Test that the part of the result that is part of an edge in the image
        does not change a lot.
        """
        
        # Initialize the input image.
        input_img = np.array([[[0, 0, 0], [0, 0, 0],  [255, 255, 255], [40, 40, 40], [41, 41, 41], \
                               [38, 38, 38], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [34, 34, 34], [42, 42, 42], \
                               [44, 44, 44], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [43, 43, 23], [30, 30, 30], \
                               [40, 40, 40], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [40, 40, 40], [45, 45, 45], \
                               [35, 35, 35], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [45, 45, 45], [30, 30, 30], \
                               [48, 48, 48], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], \
                               [255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], \
                               [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], \
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], \
                               [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]).astype(np.float32)

        # Initialize the guiding filter to be 2D and contains similar but less variable 
        # values as the input image.
        p = np.array([[25, 25, 255, 50, 50, 50, 255, 25, 25],
                      [25, 25, 255, 50, 50, 50, 255, 25, 25],
                      [25, 25, 255, 50, 50, 50, 255, 25, 25],
                      [25, 25, 255, 50, 50, 50, 255, 25, 25],
                      [25, 25, 255, 50, 50, 50, 255, 25, 25],
                      [25, 25,  255, 255, 255, 255, 255, 25, 25],
                      [25, 25, 25, 25, 25, 25, 25, 25, 25], 
                      [25, 25, 25, 25, 25, 25, 25, 25, 25]]).astype(np.float32)

        # Execute the filter method.
        g = GF.MultiDimGuidedFilter(input_img, 2, 0.3) 
        res = g.filter(p)
        # Assert that the values of the edge points are barely changed  while 
        # the other values are closer to the values of the guided filter 
        # than of those of the input image. 
        assert len(res) == len(p[0]) and len(res) == input_img.shape[1]
        for a in range(len(res)):
            if a == 2 or a == 6:
                assert abs(p[0][a] - res[a]) < 1 and \
                    abs(input_img[0,a,0] - res[a]) < 1 and \
                    abs(input_img[0,a,1] - res[a]) < 1 and \
                    abs(input_img[0,a,2] - res[a]) < 1
            else:
                assert abs(p[0][a] - res[a]) < abs(res[0] - input_img[0,a,0]) and \
                       abs(p[0][a] - res[a]) < abs(res[0] - input_img[0,a,1]) and \
                       abs(p[0][a] - res[a]) < abs(res[0] - input_img[0,a,2])

if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
