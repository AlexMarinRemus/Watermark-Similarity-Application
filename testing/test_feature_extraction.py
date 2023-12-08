import sys
sys.path.append("c:\\Users\\skho2\\Documents\\GitHub\\School\\cs-for-the-humanities-watermarks-18b")
import numpy as np
import pytest
import cv2
from unittest.mock import patch, mock_open, call, Mock
from unittest import TestCase
from sklearn.decomposition import PCA
import mahotas.features
import skimage

from feature_extraction.strategies import DetectionStrategy, ExtractionStrategy
from feature_extraction.feature_extraction import FeatureExtraction


class TestFeatureExtraction:

    # Test for the extract_features method from feature_extraction/feature_extraction.py
    # file.

    def test_extract_features(self):
        """
        Test that extract_features method calls all functions in the right order and 
        with the right arguments.
        """

        # Initialize the image to be used during execution.
        input_image = np.zeros((5,5)).astype(np.uint8)

        # Mock all the methods and classes that are called during the executio of the 
        # extract_features method.
        with patch.object(DetectionStrategy, "SIFT") as mock_detection_sift, \
             patch.object(ExtractionStrategy, "SIFT") as mock_extraction_sift, \
             patch.object(ExtractionStrategy, "Hu_moments") as mock_hu_moments, \
             patch.object(ExtractionStrategy, "Zernike_moments") as mock_zernike_moments, \
             patch("cv2.SIFT_create") as mock_sift_create, \
             TestCase.assertLogs(self) as captured:
            
            # Call the method being tested with the input image as the argument.
            f = FeatureExtraction()
            result = f.extract_features(input_image) 

            # Assert that the ExtractionStrategy and DetectionStrategy files are called 
            # correctly and with the right attributes. 
            mock_sift_create.assert_called_once()
            mock_detection_sift.assert_called_once_with(mock_sift_create.return_value)
            mock_extraction_sift.assert_called_once_with(mock_sift_create.return_value) 
            mock_hu_moments.assert_called_once()
            mock_zernike_moments.assert_called_once()

            # The mocks are classes that have methods, so the returned value from the 
            # method of the mocked class is saved and will be returned as the result later.
            mock_keypoints = mock_detection_sift.return_value.detect.return_value 
            mock_descriptors = mock_extraction_sift.return_value.extract.return_value
            mock_hu_return = mock_hu_moments.return_value.extract.return_value
            mock_zernike_return = mock_zernike_moments.return_value.extract.return_value

            # Assert that the methods of the mocked classes are called correctly with the
            # right attributes.
            mock_detection_sift.return_value.detect.assert_called_once_with(input_image)
            mock_extraction_sift.return_value.extract.assert_called_once_with(input_image, \
                                                                               mock_keypoints)
            mock_hu_moments.return_value.extract.assert_called_once_with(input_image, mock_keypoints)
            mock_zernike_moments.return_value.extract.assert_called_once_with(input_image)

            # Assert that the result contains all the necessary elements.
            assert len(result) == 3
            assert result[0] == mock_descriptors
            assert result[1] == mock_hu_return
            assert result[2] == mock_zernike_return

            # Assert that the log message is as expected.
            assert len(captured.records) == 1
            assert captured.records[0].getMessage() == "Detecting and extracting features..."


    # Tests for DetectionStrategy.py file.

    def test_detection_harris(self):
        """
        Test the Harris corner detection algorithm
        """
        image = np.zeros((400, 400), dtype=np.uint8)

        image[100:300, 100:300] = 255

        harris = DetectionStrategy.Harris()
        keypoints = harris.detect(image)

        def close_to_corner(pt):
            return (99 <= pt[0] <= 101 or 299 <= pt[0] <= 301)\
            and (99 <= pt[1] <= 101 or 299 <= pt[1] <= 301)

        assert all(close_to_corner(kp.pt) for kp in keypoints)

    def test_detection_SIFT(self, mocker):
        """
        Test the SIFT algorithm by mocking a SIFT object
        and checking that the detect method is called
        """
        image = np.zeros((1, 1), np.uint8)
        sift_mock = mocker.Mock()

        d = DetectionStrategy.SIFT(sift_mock)
        d.detect(image)

        sift_mock.detect.assert_called_once_with(image, None)

    def test_extraction_SIFT(self, mocker):
        """
        Test the SIFT algorithm by mocking a SIFT object
        and checking that the compute method is called
        """
        image = np.zeros((1, 1), np.uint8)
        keypoints = [cv2.KeyPoint(0, 0, 1)]
        sift_mock = mocker.Mock()
        sift_mock.compute.return_value = (None, None) # extract() returns the second value of a tuple

        e = ExtractionStrategy.SIFT(sift_mock)
        e.extract(image, keypoints)

        sift_mock.compute.assert_called_once_with(image, keypoints)

    def test_extraction_LBP_image(self):
        """
        Test the LBP algorithm. Each binary string should be interpreted as
        8 surrounding pixels that are brighter (or equal) or darker than the center pixel.
        The order is clockwise, starting from the bottom right corner.
        Here are some examples with corresponding binary strings:
         Corner      Edge        Flat
         0 0 0       0 1 1       1 1 1
         0 - 1       0 - 1       1 - 1
         0 1 1       0 1 1       1 1 1
        1100001     11000111    11111111
        """
        image = np.zeros((64, 64), np.uint8)

        image[16:48, 16:48] = 255

        keypoints = [cv2.KeyPoint(31, 31, 1)]

        e = ExtractionStrategy.LBP_image()
        result_image = np.reshape(e.extract(image, keypoints)[0], (64, 64))

        assert result_image.shape == (64, 64)
        # Corners
        assert result_image[16, 16] == 193 #11000001
        assert result_image[47, 47] == 28 #00011100
        assert result_image[16, 47] == 112 #01110000
        assert result_image[47, 16] == 7 #00000111

        # Edges
        assert all(result_image[17:47, 16] == 199) #11000111
        assert all(result_image[17:47, 47] == 124) #01111100
        assert all(result_image[16, 17:47] == 241) #11110001
        assert all(result_image[47, 17:47] == 31) #00011111

        # Center
        assert (result_image[17:47, 17:47] == 255).all() #11111111

    def test_extraction_LBP_histogram(self):
        """
        Test the LBP histogram algorithm. Should be the same as the LBP_image, but as a histogram.
        """
        image = np.zeros((64, 64), np.uint8)

        image[16:48, 16:48] = 255

        keypoints = [cv2.KeyPoint(31, 31, 1)]

        e = ExtractionStrategy.LBP_histogram()
        result_histogram = e.extract(image, keypoints)[0]

        assert result_histogram.shape == (256,)
        # Corners
        assert result_histogram[193] == 1
        assert result_histogram[28] == 1
        assert result_histogram[112] == 1
        assert result_histogram[7] == 1

        # Edges
        assert result_histogram[199] == 30
        assert result_histogram[124] == 30
        assert result_histogram[241] == 30
        assert result_histogram[31] == 30

        # Center
        assert result_histogram[255] == 3972 # 64*64 - 4*30 - 4*1

    def test_extraction_LBP_histogram_none(self):
        """
        Test the LBP histogram method when there are no keypoints.
        """

        # Initialize the input image
        input_image = np.zeros((5,5)).astype(np.uint8)
        # Mock the local_binary_pattern method and set its return value.
        with patch.object(skimage.feature, "local_binary_pattern") as mock_loc_bin_pat:
            mock_loc_bin_pat.return_value = [[1,2,3],[2,3,4]]
            # Call the extract method with the input image and with None as the keypoints
            # argument.
            l = ExtractionStrategy.LBP_histogram()
            res = l.extract(input_image, keypoints=None)
            # Assert that the call to the mocked method was performed with the right 
            # attributes
            mock_loc_bin_pat.assert_called_once_with(input_image, 8, 1, method=l.method)
            # Construct the expected result and assert that it is equal to the one returned
            # from the extract method.
            expected_res = np.zeros(256)
            expected_res[1] = 1
            expected_res[2] = 2
            expected_res[3] = 2
            expected_res[4] = 1
            assert np.array_equal(expected_res, res[0])


    # Tests for the gabor_images method.

    def test_gabor_images_energy_false(self):
        """
        Test that if energy is evaluated o false, then only the gabor_kernel and the
        filter_2D operations will be performed 8x5=40 times and then all results will
        be attached to the returned list.
        """
        
        # Initialize the input image.
        input_img = np.ones((5,5)).astype(np.uint8)

        # Mock the operations that are being called during the execution.
        with patch("skimage.filters.gabor_kernel") as mock_gabor_kernel, \
             patch("cv2.filter2D") as mock_filter_2D, \
             patch("numpy.power") as mock_np_power, \
             patch("cv2.GaussianBlur") as mock_gauss_blur:
            
            # Call the method being tested with attribute energy=False.
            h = ExtractionStrategy._Helper()
            res = h.gabor_images(input_img, energy=False)

            # Assert that there were 40 calls to the gabor_kernel and filter_2D
            # methods and none to the methods that require energy to be evaluated to True.
            assert mock_gabor_kernel.call_count == 40
            assert mock_filter_2D.call_count == 40
            assert mock_np_power.call_count == 0
            assert mock_gauss_blur.call_count == 0
            # Assert that the returned list has the 40 images obtained after the 40 iterations.
            assert len(res) == 40

    def test_gabor_images_energy_true(self):
        """
        Test that if energy is True, then np_power and GaussianBlur are also called
        40 times with the other methods.
        """

        # Initialize the input image.
        input_img = np.ones((5,5)).astype(np.uint8)

        # Mock the operations that are being called during the execution.
        with patch("skimage.filters.gabor_kernel") as mock_gabor_kernel, \
             patch("cv2.filter2D") as mock_filter_2D, \
             patch("numpy.power") as mock_np_power, \
             patch("cv2.GaussianBlur") as mock_gauss_blur:
            
            # Call the method being tested with attribute energy=True.
            h = ExtractionStrategy._Helper()
            res = h.gabor_images(input_img, energy=True)

            # Assert that there were 40 calls for all the methods.
            assert mock_gabor_kernel.call_count == 40
            assert mock_filter_2D.call_count == 40
            assert mock_np_power.call_count == 40
            assert mock_gauss_blur.call_count == 40
            # Assert that the returned list has the 40 images obtained after the 40 iterations.
            assert len(res) == 40

    # Test for the train_PCA method.

    def test_train_pca(self):
        """
        Test that the method returns an instance of a PCA.
        """

        # Initialize the variables to be used.
        img = np.ones((5,5)).astype(np.uint8)
        img2 = np.array([[1,3,4,5,6],
                         [22, 33, 44, 55, 66],
                         [100, 102, 100, 102, 100],
                         [12, 13, 14, 15, 15],
                         [33,33,33,33,33]]).astype(np.uint8)
        imgs = [img, img2]

        # Mock methods.
        with patch.object(PCA, "fit") as mock_pca, \
             patch.object(ExtractionStrategy._Helper, "gabor_images") as mock_gabor:
            
            # Call train_PCA method with the list of images.
            h = ExtractionStrategy._Helper()
            res = h.train_PCA(imgs)
            # Assert that there are calls to the mock of gabor_images for both
            # images in the list.
            gabor_calls = mock_gabor.call_args
            assert gabor_calls.equals((img, img2))
            # Assert that the result is an instance of a PCA.
            assert isinstance(res, PCA)

    # Test for the extract method from Gabor class.

    def test_extract(self):
        """
        Test that the method performs all the calls to other functions
        properly and that the returned value is as expected. 
        """

        # Initialize the image to be passed the extract method and the 
        # return value that will be assigned for the mock of the gabor_images
        # function.
        input_img = np.zeros((5,5)).astype(np.uint8)
        gabor_returned = [[1,1,1], [2,2,2], [3,3,3]]

        # Mock gabor_images method.
        with patch.object(ExtractionStrategy._Helper, "gabor_images") as mock_gabor:
            # Set the return value and call the extract method.
            mock_gabor.return_value = gabor_returned
            g = ExtractionStrategy.Gabor()
            res = g.extract(input_img)
            # Assert that the call to the mock of gabor_images was done with 
            # the right attributes and that the expected result is correct.
            mock_gabor.assert_called_once_with(input_img, energy=True)
            feature_vect = [[1, 0], [2, 0], [3, 0]]
            assert np.array_equal(res, np.array(feature_vect).flatten())

    # Test for the init method of Gabor_DCT class.

    def test_init_gabor(self):
        """
        Test that pca_model attribute is correctly assigned when initializing
        an instance of the Gabor_DCT class.
        """

        # Initialize the list to be passed to init.
        images = [np.zeros((5,5)).astype(np.uint8)]
        # Mock the train_PCA method.
        with patch.object(ExtractionStrategy._Helper, "train_PCA") as mock_pca:
            # Create the instance of the class and make sure the mock is called 
            # with the right attribute.
            res = ExtractionStrategy.Gabor_DCT(images)
            mock_pca.assert_called_with(images)
            # Assert that the pca_model attribute is the same one as the one passed
            # when initializing the class object.
            assert np.array_equal(res.pca_model, mock_pca.return_value)

    # Test for the extract method of Gabor_DCT class.

    def test_extract_gabor_dct(self):
        """
        Test that the extract method calls all the other functions in the right order
        and with the right attributes.
        """

        # Initialize the input image. 
        input_img = np.array([[1,3,4,5,6],
                         [22, 33, 44, 55, 66],
                         [100, 102, 100, 102, 100],
                         [12, 13, 14, 15, 15],
                         [33,33,33,33,33]]).astype(np.uint8)
        images = [100*np.random.rand(10,10), 100*np.random.rand(10,10)]
        # images = [2*input_img, 3*input_img]
        
        # Mock the methods that will be aclled.
        with patch.object(ExtractionStrategy._Helper, "gabor_images") as mock_gabor_imgs, \
             patch("scipy.fftpack.dct") as mock_fft_dct, \
             patch.object(PCA, "transform") as mock_transform:
            # Set return values for the mocks.
            mock_gabor_imgs.return_value = [100*np.random.rand(10,10), \
                                            100*np.random.rand(10,10), \
                                            100*np.random.rand(10,10)]
            mock_fft_dct.return_value = 100*np.random.rand(10,10)

            # Call the extract method with the input image.
            g = ExtractionStrategy.Gabor_DCT(images)
            res = g.extract(input_img)
            
            # Assert that the mocked methods are called in the right order with the
            # corresponding attributes and values.
            mock_gabor_imgs.assert_called_with(input_img, energy=True)
            fft_dct_calls = mock_fft_dct.call_args
            assert fft_dct_calls.equals((mock_gabor_imgs.return_value))
            transform_calls = mock_transform.call_args
            calls_transf = [call(mock_gabor_imgs.return_value[0]), call(mock_gabor_imgs.return_value[1])] 
            assert len(transform_calls) == 2
            transform_calls.assert_has_calls(calls_transf)
            # Assert that the result is the expected one.
            assert np.array_equal(res, np.array(mock_transform.return_value).flatten())


    # Test for the extract method from the Hu_moments class.
    
    def test_extract_hu(self):
        """
        Test for the extract method and its return value.
        """
        
        # Initialize the input image.
        input_img = np.zeros((5,5)).astype(np.uint8)
        # Mock the hu moments cv2 function.
        with patch('cv2.HuMoments') as mock_hu:
            # Assign a return value for the mocked method and call the extract
            # function with the input image.
            mock_hu.return_value = np.array([100, 1000, 10, 0, 10, 0, 0])
            h = ExtractionStrategy.Hu_moments()
            res = h.extract(input_img, keypoints=None)
            # Assert that the return value is the expected one.
            assert np.array_equal(res, [-2, -3, -1, 0, -1, 0, 0])

    # Tests for the extract method from the Zernike_moments class.

    def test_extract_zernike_none(self):
        """
        Test that if there are no zernike_moments found, the method returns
        an empty list with the original shape of the input image.
        """

        # Initialize the input image.
        input_img = np.zeros((5,5)).astype(np.uint8)
        # Mock the zernike_moments method and set its return value to None.
        with patch.object(mahotas.features, "zernike_moments") as mock_zernike:
            mock_zernike.return_value = None 
            # Call the extract method with the input image and assert that the 
            # result after execution is an empty list with the shape of the input
            # image.
            z = ExtractionStrategy.Zernike_moments()
            res = z.extract(input_img, keypoints=None)
            assert np.array_equal(res, np.zeros(input_img.shape))

    def test_extract_zernike(self):
        """
        Test that if there are zernike moments identified, they are processed
        properly and the result is returned as expected.
        """

        # Initialize the input image.
        input_img = np.zeros((5,5)).astype(np.uint8)
        # Mock the zernike_moments method and set its return value.
        with patch.object(mahotas.features, "zernike_moments") as mock_zernike:
            mock_zernike.return_value = np.array([10, 100, 1000, 0, 0, 10])
            # Call the extract method with the input image.
            z = ExtractionStrategy.Zernike_moments()
            res = z.extract(input_img, keypoints=None)
            # Assert that the result is as expected.
            assert np.array_equal(res, [-1, -2, -3, 0, 0, -1])




if __name__ == '__main__':

    pytest.main()  # Run this file with pytest
