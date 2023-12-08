"""
This file contains the class that tests all harmonization methods.
Harmonization methods can be found in 'harmonization/harmonization.py'.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, mock_open, call, Mock
from unittest import TestCase
from numpy.testing import assert_array_equal
from skimage.util import img_as_float
from skimage.restoration import denoise_wavelet, estimate_sigma

from harmonization.harmonization import Harmonization
import harmonization.wavelet_denoising as Wavelet
import harmonization.contrast_enhancement as Contrast
import harmonization.binarize as Binarize
import harmonization.utils_untraced_harmonization as Utils_Untraced
import harmonization.utils_traced_harmonization as Utils_Traced

# TODO: Add more harmonization tests once the harmonization methods are finalized.
class TestHarmonization:
    """
    Class that tests harmonization methods
    """

    # Tests for init, getter and setter of the image attribute.
    
    def test_init(self):
        """
        Test that image attribute is correctly assigned when initializing
        an instance of the Harmonization class.
        """

        input_img = np.zeros((5,5)).astype(np.uint8)
        h = Harmonization(input_img)
        # Assert that the image attribute is the same one as the one passed
        # when initializing the class object. 
        assert (h.image == input_img).all()

    def test_get_image(self):
        """
        Test that get_image returns the correct image. 
        """

        input_img = np.zeros((5,5)).astype(np.uint8)
        h = Harmonization(input_img)
        # Assert that the get method returns the image used to initialize
        # the class object.
        assert (h.get_image() == input_img).all()

    def test_set_image(self):
        """
        Test that set_image correctly updates the image to a new one. 
        """

        input_img = np.zeros((5,5)).astype(np.uint8)
        h = Harmonization(input_img)
        result_img = np.zeros((3,3)).astype(np.uint8)
        
        # Assert that the input image is the same as the current image
        # attribute.
        assert (h.get_image() == input_img).all()

        h.set_image(result_img)
        # Assert that the image attribute for the object has changed.
        assert (h.get_image() == result_img).all()


    # Test for the preprocess_traced method.

    def test_preprocess_traced(self):
        """
        Test that all the preprocessing method calls and operations are done 
        in the right order and with the right input.
        """

        # Define the values that will be used.
        input_img = np.zeros((5,5)).astype(np.uint8)
        option = 1

        # Make mocks for all the functions that will be called in the execution of the 
        # preprocess_traced method. 
        with patch.object(Contrast, "ameliorate_contrast_on_margins") as mock_ameliorate_contrast, \
                 patch.object(Wavelet, "wavelet_traced") as mock_wavelet_traced, \
                 patch.object(Contrast, "contrast_stretch") as mock_contrast_stretch, \
                 patch("numpy.clip") as mock_np_clip, \
                 patch.object(Contrast, "remove_shadows") as mock_remove_shadows, \
                 TestCase.assertLogs(self) as captured:
            
            # Call the preprocessing method for the input image.
            h = Harmonization(input_img)
            preprocessed_img = h.preprocess_traced(option=option)
            remove_shadows_call_args = mock_remove_shadows.call_args

            # Assert that all the calls take place in the right order and with the right arguments.
            mock_ameliorate_contrast.assert_called_with(input_img)
            mock_wavelet_traced.assert_called_with(mock_ameliorate_contrast.return_value, option=option)
            mock_contrast_stretch.assert_called_with(mock_wavelet_traced.return_value)
            mock_np_clip.assert_called_with(mock_contrast_stretch.return_value, 0, 255)
           
            # Assert that the last method to be called has the right arguments.
            assert len(remove_shadows_call_args[0]) == 3
            assert remove_shadows_call_args.equals(mock_np_clip.return_value)
            assert (remove_shadows_call_args[0][1] == np.ones((8,8))).all()
            assert remove_shadows_call_args[0][2] == 33

            # Assert that the log message is as expected.
            assert len(captured.records) == 1
            assert captured.records[0].getMessage() == "Preprocessing traced image"

            # Assert that at the end of preprocessing, the returned value is the same as
            # the value returned by the last method that was called.
            assert np.array_equal(preprocessed_img, mock_remove_shadows.return_value)


    # Test for the preprocess_untraced method.

    def test_preprocess_untraced(self):
        """
        Test that the image is being inverted as a result of preprocessing for
        an untraced watermark image.
        """

        input_img = np.zeros((5,5)).astype(np.uint8)
        result_img = np.array([[255, 255, 255, 255, 255],
                               [255, 255, 255, 255, 255],
                               [255, 255, 255, 255, 255],
                               [255, 255, 255, 255, 255],
                               [255, 255, 255, 255, 255]]).astype(np.uint8)
        
        h = Harmonization(input_img)
        preprocessed_img = h.preprocess_untraced()

        # Assert that the image has been inverted as result of the preprocessing 
        # operation.
        assert len(preprocessed_img) == len(result_img)
        assert np.array_equal(preprocessed_img, result_img)

    
    # Test for the threshold_traced_light_noise method.

    def test_theshold_traced_light_noise(self):
        """
        Test that the thresholding method calls all the functions in the right
        order and with the right arguments, and that the return value of the 
        method is the same as the return value of the last operation that was performed
        on the given image.
        """

        # Initialize the values that will be used during the execution.
        input_img = np.zeros((5, 5)).astype(np.uint8)
        dilation_shape = (3, 3)
        window_size = 25
        k = 0.2

        # Make mocks for all the functions that will be called during the execution of the 
        # threshold_traced_light_noise method.
        with patch.object(Binarize, "sauvola_thresholding") as mock_sauvola_thresh, \
             patch("cv2.getStructuringElement") as mock_struct_elem, \
             patch("cv2.dilate") as mock_dilate, \
             TestCase.assertLogs(self) as captured:
            
            # Call the threshold method for the input image.
            h = Harmonization(input_img)
            thresholded_img = h.threshold_traced_light_noise(dilation_shape=dilation_shape, \
                                                             window_size=window_size, \
                                                            k=k)
            dilate_call_args = mock_dilate.call_args 

            # Assert that all mocks were called with the right arguments in the right order.
            mock_sauvola_thresh.assert_called_with(input_img, window_size=window_size, \
                                                   k=k, r=100)
            mock_struct_elem.assert_called_with(cv2.MORPH_ELLIPSE, dilation_shape)

            # Because np.uint8 was not mocked, apply it on the return values of the mocked
            # objects it is called with in the thresholding method.
            sauvola_thresh_uint = np.uint8(mock_sauvola_thresh.return_value)
            struct_elem_uint = np.uint8(mock_struct_elem.return_value)

            # Assert that the last method called has the right arguments.
            assert len(dilate_call_args) == 2
            assert dilate_call_args.equals((sauvola_thresh_uint, struct_elem_uint))

            # Assert that the log message is the expected one.
            assert len(captured.records) == 1
            assert captured.records[0].getMessage() == "Thresholding light noise"

            # Assert that the final return value is the same as the return value of the
            # last operation perfomed during the execution of the thresholding method.
            assert np.array_equal(thresholded_img, mock_dilate.return_value)


    # Test for the denoise_traced_heavy_noise method.

    def test_denoise_traced_heavy_noise(self):
        """
        Test that the denoising method calls all the functions in the right order and with
        the right arguments, and also that the value returned in the end is equal to the 
        value returned by the last method that was called during the execution.
        """

        # Set up parameters to be used.
        input_img = np.ones((5,5))
        denoise_sigma = 0.05
        gaussian_sigma = 2
        # img_as_float and denoise_wavelet are calculated by hand because they cannot
        # be mocked and checked for calls.
        img_float = img_as_float(input_img)
        img_denoised = denoise_wavelet(img_float, method="BayesShrink", mode="soft",
                                  rescale_sigma=True, sigma=denoise_sigma)*256

        # Mock the methods to be called during the execution.
        with patch("numpy.clip") as mock_np_clip, \
             patch("cv2.GaussianBlur") as mock_gauss_blur, \
             TestCase.assertLogs(self) as captured:
            
            # Call the method under test.
            h = Harmonization(input_img)
            result = h.denoise_traced_heavy_noise(denoise_sigma=denoise_sigma, \
                                                  gaussian_sigma=gaussian_sigma)
            # Save the call arguments for the mocks.
            np_clip_call_args = mock_np_clip.call_args
            gauss_blur_call_args = mock_gauss_blur.call_args

            # Assert that np_clip was called with the result from wavelet denoising.
            assert len(np_clip_call_args) == 2
            assert np.array_equal(np_clip_call_args[0][0], img_denoised)
            assert np_clip_call_args[0][1] == 0
            assert np_clip_call_args[0][2] == 255

            # Assert that GaussianBlur was called with the right arguments.
            assert len(gauss_blur_call_args) == 2
            assert np.array_equal(gauss_blur_call_args[0][0], input_img)
            assert gauss_blur_call_args[0][1] == (0,0)
            assert gauss_blur_call_args[0][2] == gaussian_sigma

            # Assert that the log message is as expected.
            assert len(captured.records) == 1
            assert captured.records[0].getMessage() == "Denoising heavy noise"

            # Assert that the value returned is the same as the one obtained after
            # the gaussian blur operation.
            assert np.array_equal(result, mock_gauss_blur.return_value)


    # Test for the threshold_traced_heavy_noise method.

    def test_threshold_traced_heavy_noise(self):
        """
        Test that the threshold method calls all the functions in the right order
        and with the right arguments, and also that the result returned after the 
        execution of this method is the same as the result of the last function being
        called.
        """

        # Set up the variables that will be used during the execution.
        input_img = np.array([[200, 200, 100, 200, 200],
                              [150, 50, 50, 50, 150],
                              [150, 50, 50, 50, 150],
                              [150, 50, 50, 50, 150],
                              [200, 200, 100, 200, 200]]).astype(np.uint8)
        # The expected value after the cv2 method for global thresholding.
        img_threshold = np.array([[0, 0, 255, 0, 0],
                                  [255, 255, 255, 255, 255],
                                  [255, 255, 255, 255, 255],
                                  [255, 255, 255, 255, 255],
                                  [0, 0, 255, 0, 0]]).astype(np.uint8)
        threshold_value = 190
        closing_shape = (6,6)
        dilation_shape = (3,3)

       # Mock the functions to be called during the execution, except for the 
       # global thresholding and np.ones, since these can be calculated directly and
       # verified.
        with patch("cv2.morphologyEx") as mock_morph_ex, \
             patch("cv2.getStructuringElement") as mock_struct_elem, \
             patch("cv2.dilate") as mock_dilate, \
             TestCase.assertLogs(self) as captured:
            
            # Call the method under test.
            h = Harmonization(input_img)
            result = h.threshold_traced_heavy_noise(threshold_value=threshold_value, \
                                                    closing_shape=closing_shape, \
                                                        dilation_shape=dilation_shape)
            # Initialize the kernel and get the call args for methods.
            kernel = np.ones(closing_shape).astype(np.uint8)
            morph_ex_call_args = mock_morph_ex.call_args
            dilate_call_args = mock_dilate.call_args

            # Assert that the closing operation was performed with the thresholded image
            # and the right kernel.
            assert len(morph_ex_call_args) == 2
            assert np.array_equal(morph_ex_call_args[0][0], img_threshold)
            assert morph_ex_call_args[0][1] == cv2.MORPH_CLOSE
            assert np.array_equal(morph_ex_call_args[0][2], kernel)

            # Assert that the structuring element was created with the proper arguments.
            mock_struct_elem.assert_called_with(cv2.MORPH_ELLIPSE, dilation_shape)

            # Assert that the dilation was performed with the right arguments.
            kernel_lines = np.uint8(mock_struct_elem.return_value)
            assert len(dilate_call_args) == 2
            assert dilate_call_args.equals((mock_morph_ex.return_value, kernel_lines))

            # Assert that the log message is as expected.
            assert len(captured.records) == 1
            assert captured.records[0].getMessage() == "Thresholding heavy noise"

            # Assert that the value returned in the end is the same as the returned value
            # of the dilation operation.
            assert np.array_equal(result, mock_dilate.return_value)


    # Test for the denoise_untraced method.

    def test_denoise_untraced(self):
        """
        Test that the denoise method calls the function from the untraced utils file 
        and returns its result.
        """

        # Set up variables to be used during the method execution.
        input_img = np.zeros((5,5)).astype(np.uint8)
        sigma_psd = 20

        # Mock the denoise_untraced method from the utils file.
        with patch.object(Utils_Untraced, "denoise_untraced") as mock_denoise_untraced:

            # Call the method under test and save the call arguments of the mock.
            h = Harmonization(input_img) 
            result = h.denoise_untraced(sigma_psd=sigma_psd) 
            denoise_untraced_call_args = mock_denoise_untraced.call_args

            # Assert that the denoise_untraced method is called with the right arguments.
            assert len(denoise_untraced_call_args) == 2
            assert denoise_untraced_call_args.equals((input_img, sigma_psd))

            # Assert that the result returned at the very end is the same as the one returned
            # by the denoise_untraced method.
            assert np.array_equal(result, mock_denoise_untraced.return_value)


    # Test for the threshold_untraced method.

    def test_threshold_untraced(self):
        """
        Test that the thresholding method calls the function from the utils file 
        and returns its result.
        """

        # Initialize the input image.
        input_img = np.zeros((5,5)).astype(np.uint8)

        # Mock the thresholding method from the utils file.
        with patch.object(Utils_Untraced, "threshold_untraced") as mock_threshold_untraced:

            # Call the method under test and get the call arguments for the mock.
            h = Harmonization(input_img) 
            result = h.threshold_untraced()
            threshold_untraced_call_args = mock_threshold_untraced.call_args 

            # Assert that the mock is executed with the correct default values.
            assert len(threshold_untraced_call_args) == 2
            assert threshold_untraced_call_args.equals((input_img, 25, 0.21, (3,3), 3))

            # Assert that the result of the threshold method is the same as the one from
            # threshold_untraced function from the utils file.
            assert np.array_equal(result, mock_threshold_untraced.return_value)


    # Tests for the post_process_traced method.

    def test_post_process_traced_iter_1(self):
        """
        Test that if iteration is 1, then the cluster_pixels method is called only once with the input image
        as the argument. Assert that the image shape is not decreased and that regions kept after the 
        filtering are enclosed within the coordinates returned by cluster_pixels. Also make
        sure that the rest of the portions in the image after post-processing contain only 0's. 
        """

        # Set up the variables to be used.
        raw_img = np.array([[200, 200, 200, 200, 200],
                              [100, 100, 100, 100, 100],
                              [50, 50, 50, 50, 50],
                              [100, 100, 100, 100, 100],
                              [200, 200, 200, 200, 200]]).astype(np.uint8)
        
        input_img = np.ones((5,5)).astype(np.uint8)
        iteration = 1
        wavelet_option = 1
        # Set values to be returned by cluster_pixels mock.
        clustered_img = np.ones((7,7)).astype(np.uint8) * 255
        min_x = min_y = 2
        max_x = max_y = 5

        # Mock cluster_pixels.
        with patch.object(Utils_Traced, "cluster_pixels") as mock_cluster_pixels, \
            TestCase.assertLogs(self) as captured:

            # Set the return value for the mock.
            mock_cluster_pixels.return_value = (clustered_img, (min_x, min_y, max_x, max_y))

            # Call the method to be tested.
            h = Harmonization(input_img)
            result = h.post_process_traced(iteration, raw_img, wavelet_option)

            # Assert that the mock was called with the input image.
            assert mock_cluster_pixels.call_args.equals((input_img))
            # The result will be the section from clustered_img bounded by the returned coordinates
            # and padded with 0's to maintain the image shape.
            post_processed_img = np.array([[0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 255, 255, 255, 0, 0],
                                           [0, 0, 255, 255, 255, 0, 0],
                                           [0, 0, 255, 255, 255, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
            
            # Assert that the log message is as expected.
            assert len(captured.records) == 1
            assert captured.records[0].getMessage() == "Post-processing traced image"
            
            # Assert that the returned result is as expected. The shape of the returned image is 
            # increased by 2 on each direction due to the padding with 0's.
            assert result.shape[0] == input_img.shape[0] + 2
            assert result.shape[1] == input_img.shape[1] + 2
            assert np.array_equal(result, post_processed_img)

    def test_post_process_traced_iter_2(self):
        """
        Test that if iteration is 2, then the methods are called in the right order with the right
        arguments, and the image shape after post processing is not reduced to bound the region 
        left after filtering. Also ensure that the rest of the values in the result are 0's, and 
        that both sets of coordinates resulted from cluster_pixels and harmonize_traced methods 
        contribute to maintaining the original position of the non-zero region kept after 
        post-processing. 
        """

        # Set up the variables to be used for calling the method post_process_traced.
        input_img = np.ones((7,7)).astype(np.uint8)
        raw_img = np.array([[255, 255, 255, 255, 255, 255, 255],
                            [200, 200, 200, 200, 200, 200, 200],
                            [100, 100, 100, 100, 100, 100, 100],
                            [50, 50, 50, 50, 50, 50, 50],
                            [100, 100, 100, 100, 100, 100, 100],
                            [200, 200, 200, 200, 200, 200, 200],
                            [255, 255, 255, 255, 255, 255, 255]]).astype(np.uint8)
        iteration = 2
        wavelet_option = 1

        # Initialize the return values for the mock of cluster_pixels.
        clustered_img = np.ones((9,9)).astype(np.uint8)
        min_x = min_y = 1
        max_x = max_y = 7

        # Initialize the padded raw image which will be used to call harmonize_traced, as well 
        # as the return values for the mock of this method.
        raw_img_padded = cv2.copyMakeBorder(raw_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        harmonize_kept = np.array([[34, 35, 36, 37, 38],
                                   [100, 100, 100, 100, 100],
                                   [10, 11, 12, 13, 14],
                                   [45, 46, 47, 48, 49],
                                   [100, 100, 100, 100, 100]]).astype(np.uint8)
        min_x2 = min_y2 = 2
        max_x2 = max_y2 = 5

        # Mock the two methods that are called by post_process_traced. 
        with patch.object(Utils_Traced, "cluster_pixels") as mock_cluster_pixels, \
             patch.object(Utils_Traced, "harmonize_traced") as mock_harmonize_traced, \
             TestCase.assertLogs(self) as captured:
            
            # Set the return values for both mocks
            mock_cluster_pixels.return_value = (clustered_img, (min_x, min_y, max_x, max_y))
            mock_harmonize_traced.return_value = (harmonize_kept, (min_x2, min_y2, max_x2, max_y2))
            
            # Call the method under testing.
            h = Harmonization(input_img)
            result = h.post_process_traced(iteration, raw_img, wavelet_option)
            harmonize_traced_call_args = mock_harmonize_traced.call_args
            # Assert that the mock for cluster_pixels is called with the input image.
            assert mock_cluster_pixels.call_args.equals((input_img))

            # Assert that the mock for harmonize_traced is called with the correct arguments.
            raw_img_kept = raw_img_padded[min_y:max_y, min_x:max_x]
            assert len(harmonize_traced_call_args) == 2
            assert np.array_equal(harmonize_traced_call_args[0][0], raw_img_kept)
            assert np.array_equal(harmonize_traced_call_args[0][1], raw_img_padded)
            assert harmonize_traced_call_args[1]["wavelet_option"] == wavelet_option
            
            # Initialize the expected result as an array of zeros with increased shape, due to 
            # padding with zeros.
            post_processed_img = np.zeros((input_img.shape[0] + 2, input_img.shape[1] + 2))
            # Only the portion bounded by the set of coordinates from the harmonize_traced method
            # that is inside the portion bounded by the set of coordinates from the cluster_pixels
            # method will have non-zero values.
            post_processed_img[min_y2 + min_y:max_y2 + min_y, min_x2 + min_x:max_x2 + min_x] = \
                                    harmonize_kept[min_y2:max_y2, min_x2:max_x2]
            # Assert that the returned result is the expected one.
            assert result.shape[0] == input_img.shape[0] + 2
            assert result.shape[1] == input_img.shape[1] + 2
            assert np.array_equal(post_processed_img, result)
            
            # Assert that the log message is as expected.
            assert len(captured.records) == 1
            assert captured.records[0].getMessage() == "Post-processing traced image"


    # Test for the post_process_untraced method.

    def test_post_process_untraced(self):
        """
        Test that post process calls the connected_component_analysis method with the
        right arguments and returns its result.
        """

        # Initialize the input image.
        input_img = np.zeros((5,5)).astype(np.uint8)

        # Mock the connected_component_analysis method.
        with patch.object(Utils_Untraced, "connected_component_analysis") as mock_connected_component:

            # Call the method being tested.
            h = Harmonization(input_img)
            result = h.post_process_untraced()

            # Assert that the mock is called with the right arguments and that the result
            # of post processing is the expected one with the right type.
            assert mock_connected_component.call_args.equals((input_img, 200))
            assert np.array_equal(result, np.uint8(mock_connected_component.return_value))


    # Tests for the harmonize method.

    def test_harmonize_traced_heavy(self):
        """
        Test that if the image is traced and heavily noised then it will be harmonized using the
        heavy noise methods in the right order with the right arguments.
        """
        
        # Initialize input image with large differences between neighboring values.
        input_img = np.array([[25, 103, 204, 5, 58],
                              [180, 13, 100, 55, 200],
                              [2, 200, 4, 255, 8],
                              [184, 5, 220, 8, 165],
                              [33, 200, 1, 240, 0]]).astype(np.uint8) 
        
        # Mock all methods that may be called from the harmonization file.
        with patch.object(Harmonization, "preprocess_traced") as mock_preprocess, \
             patch.object(Harmonization, "denoise_traced_heavy_noise") as mock_denoise, \
             patch.object(Harmonization, "threshold_traced_heavy_noise") as mock_threshold_heavy, \
             patch.object(Harmonization, "post_process_traced") as mock_post_process, \
             patch.object(Harmonization, "threshold_traced_light_noise") as mock_threshold_light, \
             TestCase.assertLogs(self) as captured:
            
            # Call the method to be tested.
            h = Harmonization(input_img)
            result = h.harmonize(is_traced=True)
            
            # Assert that preprocessing is called as usual.
            mock_preprocess.assert_called_with(option=1)
            # Assert that the image has a large value for estimate_sigma, so it will be 
            # processed as heavily noisy.
            assert estimate_sigma(input_img, average_sigmas=True) > 1
            # Assert that light noise thresholding is not called.
            mock_threshold_light.assert_not_called()
            # Assert that the heavy noise denoise and threshold methods are called.
            mock_denoise.assert_called_once()
            mock_threshold_heavy.assert_called_once()
            # Assert that post processing method is called with the right arguments and that
            # the return value is the expected one.
            assert mock_post_process.call_args.equals((2, input_img, 1))
            assert np.array_equal(result, mock_post_process.return_value)

            # Assert that the log messages are as expected.
            assert len(captured.records) == 2
            assert captured.records[0].getMessage() == "Harmonizing image"
            assert captured.records[1].getMessage() == "Image is traced"

    def test_harmonize_traced_light(self):
        """
        Test that if the image is not noisy then the methods for harmonizing light noise images 
        will be called with the right arguments.
        """

        # Initialize the image to harmonize.
        input_img = np.ones((5,5)).astype(np.uint8) 

        # Mock all methods from harmonization file that may be called.
        with patch.object(Harmonization, "preprocess_traced") as mock_preprocess, \
             patch.object(Harmonization, "denoise_traced_heavy_noise") as mock_denoise, \
             patch.object(Harmonization, "threshold_traced_heavy_noise") as mock_threshold_heavy, \
             patch.object(Harmonization, "threshold_traced_light_noise") as mock_threshold_light, \
             patch.object(Harmonization, "post_process_traced") as mock_post_process, \
             TestCase.assertLogs(self) as captured:
            
            # Call the method being tested.
            h = Harmonization(input_img)
            result = h.harmonize(is_traced=True)

            # Assert that preprocessing mock is called with the correct argument.
            mock_preprocess.assert_called_with(option=1)
            # Assert that estimate_sigma is less than 1, so the light thresholding
            # method will be used.
            assert estimate_sigma(input_img, average_sigmas=True) < 1
            mock_threshold_light.assert_called_once()
            # Assert that the heavy noise methods are not called.
            mock_denoise.assert_not_called()
            mock_threshold_heavy.assert_not_called()
            # Assert that post processing is done with the right arguments and that the
            # result is as expected.
            assert mock_post_process.call_args.equals((2, input_img, 1))
            assert np.array_equal(result, mock_post_process.return_value)

            # Assert that the log messages are as expected.
            assert len(captured.records) == 2
            assert captured.records[0].getMessage() == "Harmonizing image"
            assert captured.records[1].getMessage() == "Image is traced"
        
    def test_harmonize_untraced(self):
        """
        Test that, for an untraced image, the harmonize method calls all functions in the 
        correct order with the correct arguments. 
        """

        # Initialize the input image.
        input_img = np.ones((5,5)).astype(np.uint8)

        # Mock the methods that will be called during execution of the harmonize method.
        with patch.object(Harmonization, "preprocess_untraced") as mock_preprocess, \
             patch.object(Utils_Untraced, "denoise_untraced") as mock_denoise, \
             patch.object(Utils_Untraced, "threshold_untraced") as mock_threshold, \
             patch.object(Harmonization, "post_process_untraced") as mock_post_process, \
             TestCase.assertLogs(self) as captured:
            
            # Call the method being tested.
            h = Harmonization(input_img) 
            result = h.harmonize(is_traced=False)

            # Assert that all methods are called in the right order with the right 
            # arguments.
            mock_preprocess.assert_called_once()
            mock_denoise.assert_called_with(mock_preprocess.return_value, 20)
            mock_threshold.assert_called_with(mock_denoise.return_value, 25, 0.21, (3, 3), 3)
            mock_post_process.assert_called_once()
            # Assert that the result is as expected.
            assert np.array_equal(result, mock_post_process.return_value)

            # Assert that the log messages are the expected ones.
            assert len(captured.records) == 2
            assert captured.records[0].getMessage() == "Harmonizing image"
            assert captured.records[1].getMessage() == "Image is not traced"



if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
