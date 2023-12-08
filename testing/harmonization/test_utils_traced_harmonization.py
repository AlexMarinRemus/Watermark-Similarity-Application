"""
This file contains the class that tests all util harmonization methods for traced watermarks.
Harmonization methods can be found in 'harmonization/utils_traced_harmonization.py'.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, mock_open, call, Mock
from unittest import TestCase
from skimage.restoration import estimate_sigma

import harmonization.utils_traced_harmonization as Utils_Traced
from harmonization.harmonization import Harmonization
import harmonization.wavelet_denoising as Wavelet
import harmonization.contrast_enhancement as Contrast
import harmonization.binarize as Binarize
import harmonization.utils_untraced_harmonization as Utils_Untraced
from harmonization.harmonization import Harmonization

class TestUtilsTracedHarmonization:
    """
    Class that tests harmonization methods
    """

    # Cluster tests

    def test_cluster_pixels_no_clusters(self):
        """
        Tests that an image with no clusters returns
        the image shape.
        """

        input_img = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]).astype(np.uint8)
        cluster_img, (min_x, min_y, max_x, max_y) = Utils_Traced.cluster_pixels(input_img)
        result_img = np.zeros((5,5)).astype(np.uint8)
        # Assert that when no clusters exist, the coordinates
        # default to the size of the image.
        assert min_x == 0 and min_y == 0
        assert max_x == 5 and max_y == 5
        # Assert that the image after clustering is as expected.
        assert np.array_equal(cluster_img, result_img)

    def test_cluster_pixels_one_cluster(self):
        """
        Tests clustering pixels on a basic image with
        one cluster with a size of one pixel.
        """

        input_img = np.array([[0, 0, 0],
                              [0, 255, 0],
                              [0, 0, 0]]).astype(np.uint8)
        cluster_img, (min_x, min_y, max_x, max_y) = Utils_Traced.cluster_pixels(input_img)
        result_img = np.zeros((5, 5)).astype(np.uint8)
        result_img[2, 2] = 255
        # Note: the maximums are exclusive, so the following coordinates
        # only capture the center pixel.
        assert min_x == 2 and min_y == 2
        assert max_x == 3 and max_y == 3
        # Assert that the image after clustering is as expected.
        assert np.array_equal(cluster_img, result_img)

    def test_cluster_pixels_clusters_with_filtering_outlier(self):
        """
        Tests that when an image has a cluster with its centroid being
        an outlier, when filter_outliers is True, it is filtered out
        by the cluster pixel function.
        """

        input_img = np.array([[255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]]).astype(np.uint8)
        cluster_img, (min_x, min_y, max_x, max_y) = Utils_Traced.cluster_pixels(input_img, \
                                                                                filter_outliers=True)
        result_img = np.zeros((13, 13)).astype(np.uint8)
        result_img[1:12, 1:12] = input_img
        result_img[11, 11] = 0
        # Assert that only the clusters on the top left of the image
        # are captured.
        assert min_x == 1 and min_y == 1
        assert max_x == 6 and max_y == 6
        # Assert that the image after clustering is as expected.
        assert np.array_equal(cluster_img, result_img)

    def test_cluster_pixels_clusters_without_filtering_outlier(self):
        """
        Tests that when an image has a cluster with its centroid being
        an outlier, when filter_outliers is False, it is not filtered out
        by the cluster pixel function.
        """

        input_img = np.array([[255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0],
                            [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]]).astype(np.uint8)
        cluster_img, (min_x, min_y, max_x, max_y) = Utils_Traced.cluster_pixels(input_img, \
                                                                   filter_outliers=False)
        result_img = np.zeros((13, 13)).astype(np.uint8)
        result_img[1:12, 1:12] = input_img
        # Assert that all clusters are captured including the outlier on the
        # bottom right.
        assert min_x == 1 and min_y == 1
        assert max_x == 12 and max_y == 12
        # Assert that the image after clustering is as expected.
        assert np.array_equal(cluster_img, result_img)

    def test_cluster_pixels_clusters_without_filtering_outlier_two_clusters(self):
        """
        Tests that when an image has two clusters, even if one may seem like an outlier,
        it is not filtered out because of the low number of clusters.
        """

        input_img = np.array([[255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]]).astype(np.uint8)
        cluster_img, (min_x, min_y, max_x, max_y) = Utils_Traced.cluster_pixels(input_img, \
                                                                                filter_outliers=True)
        result_img = np.zeros((13, 13)).astype(np.uint8)
        result_img[1:12, 1:12] = input_img
        # Assert that since there are only two clusters no outliers can
        # be calculated.
        assert min_x == 1 and min_y == 1
        assert max_x == 12 and max_y == 12
        # Assert that the image after clustering is as expected.
        assert np.array_equal(cluster_img, result_img)

    def test_cluster_pixels_removing_too_high_clusters(self):
        """
        Tests that when a cluster is too high and not wide enough,
        according to the height_proportion parameter, it is filtered out.
        """

        input_img = np.array([[255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255]]).astype(np.uint8)
        cluster_img, (min_x, min_y, max_x, max_y) = Utils_Traced.cluster_pixels(input_img, \
                                                                                height_proportion=8)
        result_img = np.zeros((13, 13)).astype(np.uint8)
        result_img[10:12, 10:12] = 255
        # Assert that onlyt he cluster on the bottom right is kept.
        assert min_x == 10 and min_y == 10
        assert max_x == 12 and max_y == 12
        # Assert that the image after clustering is as expected.
        assert np.array_equal(cluster_img, result_img)

    def test_cluster_pixels_removing_too_wide_clusters(self):
        """
        Tests that when a cluster is too wide and not high enough,
        according to the width_proportion parameter, it is filtered out.
        """

        input_img = np.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255]]).astype(np.uint8)
        cluster_img, (min_x, min_y, max_x, max_y) = Utils_Traced.cluster_pixels(input_img, \
                                                                                width_proportion=8)
        result_img = np.zeros((13, 13)).astype(np.uint8)
        result_img[10:12, 10:12] = 255
        # Assert that only the cluster on the bottom right is kept.
        assert min_x == 10 and min_y == 10
        assert max_x == 12 and max_y == 12
        # Assert that the image after clustering is as expected.
        assert np.array_equal(cluster_img, result_img)


    def test_cluster_pixels_remove_small_clusters(self):
        """
        Tests that clusters that are too small (below a certain
        fraction of the image's area) then they are filtered out.
        """
        input_img = np.array([[255, 255, 0, 0, 0],
                            [255, 255, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 255],
                            ]).astype(np.uint8)
        cluster_img, (min_x, min_y, max_x, max_y) = Utils_Traced.cluster_pixels(input_img, \
                                                                   min_bbox_percent=0.05)
        result_img = np.zeros((7, 7)).astype(np.uint8)
        result_img[1:3, 1:3] = 255
        # Assert that only the cluster on the top left is kept.
        assert min_x == 1 and min_y == 1
        assert max_x == 3 and max_y == 3 
        # Assert that the image after clustering is as expected.
        assert np.array_equal(cluster_img, result_img)


    # Tests for the make_contours method.

    def test_make_contours_no_regions(self):
        """
        Test that when there are not regions identified in the image, 
        an empty list of contours is returned
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        contours = Utils_Traced.make_contours(input_img, [])
        # Assert that no contour is returned.
        assert len(contours) == 0 


    def test_make_contours_one_region(self):
        """
        Tests that when there is only one region identified in the image, its
        contour is returned
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        region = DummyRegionObject(bbox = (0, 0, 10, 10))
        contours = Utils_Traced.make_contours(input_img, [region]) 
        result_contour = np.array([[[0, 0]], [[0, 11]], [[11, 11]], [[11, 0]]], dtype=np.int32)
        result = (result_contour,)
        # Assert that the contour of region is returned.
        assert len(contours) == len(result)
        assert np.array_equal(contours, result)
        

    def test_make_contours_more_regions(self):
        """
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        region_1 = DummyRegionObject(bbox=(5, 5, 10, 10))
        region_2 = DummyRegionObject(bbox=(4, 4, 10, 10))
        region_3 = DummyRegionObject(bbox=(17, 17, 18, 18))
        contours = Utils_Traced.make_contours(input_img, [region_1, region_2, region_3])
        result_contour_1 = np.array([[[16, 16]], [[16, 19]], [[19, 19]], [[19, 16]]], dtype=np.int32)
        result_contour_2 = np.array([[[3, 3]], [[3, 11]], [[11, 11]], [[11, 3]]], dtype=np.int32)
        result = (result_contour_1, result_contour_2,)
        # Assert that two contours are returned, one for region_3 and one for the 
        # overlap between region_1 and region_2.
        assert len(contours) == len(result) 
        assert np.array_equal(contours, result)


    # Tests for the filter_close_to_borders method.
    
    def test_filter_ctb_no_contours(self):
        """
        Test that the method returns an empty list given an empty
        list of contours.
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        contours = Utils_Traced.filter_close_to_borders(input_img, [], 1, 1)
        # Assert that if no contour was found in the image, the method
        # returns an empty string.
        assert len(contours) == 0


    def test_filter_ctb_one_contour(self):
        """
        Test that if there is only one contour passed as argument, 
        the method returns an empty list.
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        contour = np.array([[[3, 3]], [[3, 11]], [[11, 11]], [[11, 3]]], dtype=np.int32)
        contours = (contour,)
        filtered_contours = Utils_Traced.filter_close_to_borders(input_img, contours, 2, 3)
        # Assert that the list of filtered contours is empty.
        assert len(filtered_contours) == 0


    def test_filter_ctb_more_contours_1(self):
        """
        Test for image where width is not significantly larger than height.
        Test that if there are more contours in the image, the contours that are 
        within the set bounds (and are not the largest contour in the image) are kept.
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        largest_contour = np.array([[[10, 10]], [[10, 18]], [[18, 18]], [[18, 10]]], dtype=np.int32)
        contour_2 = np.array([[[2, 2]], [[2, 5]], [[5, 5]], [[5, 2]]], dtype=np.int32)
        contour_3 = np.array([[[1, 1]], [[1, 8]], [[8, 8]], [[8, 1]]], dtype=np.int32)
        contours = (largest_contour, contour_2, contour_3,) 
        filtered_contours = Utils_Traced.filter_close_to_borders(input_img, contours, 2, 1)
        result_contours = (contour_2,)
        # Assert that only the second contour in the image is kept.
        assert len(filtered_contours) == len(result_contours)
        assert np.array_equal(filtered_contours, result_contours)

    
    def test_filter_ctb_more_contours_2(self):
        """
        Test for image where width is significantly larger than height.
        Test that if there are more contours in the image, the contours that are 
        within the set bounds (and are not the largest contour in the image) are kept.
        """
        input_img = np.zeros((5, 15), dtype=np.uint8)
        largest_contour = np.array([[[0, 0]], [[0, 3]], [[4, 3]], [[4, 0]]], dtype=np.int32)
        contour_1 = np.array([[[12, 3]], [[12, 4]], [[14, 3]], [[14, 4]]], dtype=np.int32)
        contour_2 = np.array([[[6, 1]], [[6, 3]], [[10, 3]], [[10, 1]]], dtype=np.int32)
        contours = (contour_1, contour_2, largest_contour,)
        filtered_contours = Utils_Traced.filter_close_to_borders(input_img, contours, 1, 3)
        result_contours = (contour_2,)
        # Assert that only the second contour in the image is kept.
        assert len(filtered_contours) == len(result_contours)
        assert np.array_equal(filtered_contours, result_contours)


    # Tests for the filter_overlap method.
    
    def test_filter_overlap_empty(self):
        """
        Test that if no other contours other than the largest one
        there is only one contour in the image, 
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        largest_contour = np.array([[[10, 10]], [[10, 18]], [[18, 18]], [[18, 10]]], dtype=np.int32)
        filtered_contours = Utils_Traced.filter_overlap(input_img, [largest_contour], largest_contour, 80)
        # Assert that no contour is returned.
        assert len(filtered_contours) == 0

    
    def test_filter_overlap_more_contours(self):
        """
        Test that contours having low overlap percentage with the largest 
        contour are filtered out.
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        largest_contour = np.array([[[10, 10]], [[10, 18]], [[18, 18]], [[18, 10]]], dtype=np.int32)
        contour_overlapped = np.array([[[9, 9]], [[9, 16]], [[16, 16]], [[16, 9]]], dtype=np.int32)
        contour_not_overlapped = np.array([[[5, 5]], [[5, 15]], [[15, 15]], [[15, 5]]], dtype=np.int32)
        contour_not_intersected = np.array([[[2, 2]], [[2, 4]], [[4, 4]], [[4, 2]]], dtype=np.int32)
        contours = [largest_contour, contour_overlapped, contour_not_overlapped, contour_not_intersected]
        filtered_contours = Utils_Traced.filter_overlap(input_img, contours, largest_contour, 75)
        result = [contour_overlapped]
        # Assert that only the contour that overlaps the largest contour more than 75% is kept.
        assert len(filtered_contours) == len(result) 
        assert np.array_equal(filtered_contours, result)


    # Tests for the filter_by_area method.

    def test_filter_by_area_empty(self):
        """
        Test that if there are no contours in the image, an empty
        list is returned.
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        contours = []
        filtered_contours = Utils_Traced.filter_by_area(input_img, contours) 
        # Assert that no contour is returned.
        assert len(filtered_contours) == 0


    def test_filter_by_area_one_contour(self):
        """
        Test that if there is only one contour in the image, it will be returned
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        largest_contour = np.array([[[10, 10]], [[10, 18]], [[18, 18]], [[18, 10]]], dtype=np.int32)
        filtered_contours = Utils_Traced.filter_by_area(input_img, [largest_contour])
        # Assert that the contour is returned.
        assert len(filtered_contours) == 1
        assert np.array_equal(filtered_contours, [largest_contour])


    def test_filter_by_area_more_contours_1(self):
        """
        Test for image where width is not significantly larger than height.
        Test that contours that are too small are filtered out.
        """
        input_img = np.zeros((20, 20), dtype=np.uint8)
        largest_contour = np.array([[[10, 10]], [[10, 18]], [[18, 18]], [[18, 10]]], dtype=np.int32)
        contour_1 = np.array([[[1, 1]], [[1, 2]], [[2, 2]], [[2, 1]]], dtype=np.int32)
        contour_2 = np.array([[[3, 3]], [[3, 6]], [[6, 6]], [[6, 3]]], dtype=np.int32)
        contour_3 = np.array([[[7, 7]], [[7, 10]], [[10, 10]], [[10, 7]]], dtype=np.int32)
        contours = [largest_contour, contour_1, contour_2, contour_3]
        filtered_contours = Utils_Traced.filter_by_area(input_img, contours, is_restrictive=True)
        result = [largest_contour, contour_2, contour_3]
        # Assert that the first contour (which was too small) was filtered out.
        assert len(filtered_contours) == len(result) 
        assert np.array_equal(filtered_contours, result)


    def test_filter_by_area_more_contours_2(self):
        """
        Test for image where width is significantly larger than height.
        Test that contours that are too small and far from the middle are filtered out.
        """
        input_img = np.zeros((10, 30), dtype=np.uint8)
        largest_contour = np.array([[[0, 0]], [[0, 5]], [[10, 5]], [[10, 0]]], dtype=np.int32)
        contour_small_close_to_middle = np.array([[[12, 3]], [[12, 4]], [[13, 4]], [[13, 3]]],\
                                                dtype=np.int32)
        contour_small_far_from_middle = np.array([[[6, 1]], [[6, 3]], [[10, 3]], [[10, 1]]],\
                                                dtype=np.int32)
        contour_big_close_to_middle = np.array([[[18, 1]], [[18, 5]], [[23, 5]], [[23, 1]]],\
                                                dtype=np.int32)
        contour_big_far_from_middle = np.array([[[0, 0]], [[0, 4]], [[8, 4]], [[8, 0]]],\
                                               dtype=np.int32)
        contours = [largest_contour, contour_small_close_to_middle, \
                    contour_small_far_from_middle, contour_big_close_to_middle, \
                    contour_big_far_from_middle]        
        filtered_contours = Utils_Traced.filter_by_area(input_img, contours,is_restrictive=True) 
        result = [largest_contour, contour_big_close_to_middle, contour_big_far_from_middle]
        # Assert that only the contours that are too small and too far from the middle are
        # filtered out.
        assert len(filtered_contours) == len(result) 
        assert np.array_equal(filtered_contours, result)


    def test_filter_by_area_more_contours_or(self):
        """
        Tests that contour with area 0 is eliminated for an image with width much
        larger than height and is_restrictive=False.
        """
        input_img = np.zeros((30, 70), dtype=np.uint8) 
        largest_contour = np.array([[[7, 7]], [[7, 13]], [[13, 13]], [[13, 7]]], dtype=np.int32) 
        contour_small = np.array([[[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]]], dtype=np.int32) 
        contour_1 = np.array([[[15, 15]], [[15, 17]], [[17, 17]], [[17, 15]]], dtype=np.int32) 
        contours = [largest_contour, contour_small, contour_1]
        filtered_contours = Utils_Traced.filter_by_area(input_img, contours, is_restrictive=False) 
        result = [largest_contour, contour_1] 
        assert len(filtered_contours) == len(result) 
        assert np.array_equal(filtered_contours, result) 

    
    # Tests for the compute_kept_regions method.

    def test_compute_kept_regions_empty(self):
        """
        Tests that if no contours are given, then the array returned is
        empty.
        """
        input_img = np.zeros((20,20), dtype=np.uint8)
        input_img[0:1, 0:1] = 255
        input_img[19,19] = 255
        contours = []
        kept_regions = Utils_Traced.compute_kept_regions(input_img, contours, 10, 25)
        # Assert that no region is kept.
        assert len(kept_regions) == 0

    
    def test_compute_kept_regions_one_region(self):
        """
        Test that if one contour is given, it will be kept.
        """
        input_img = np.zeros((10,30), dtype=np.uint8)
        contour = np.array([[[5, 5]], [[5, 10]], [[10, 10]], [[10, 5]]], dtype=np.int32) 
        kept_regions = Utils_Traced.compute_kept_regions(input_img, [contour], 10, 25) 
        # Assert that the contour is kept.
        assert len(kept_regions) == 1
        assert np.array_equal(kept_regions, [contour]) 


    def test_compute_kept_regions_more_regions(self):
        """
        Test that the regions that are too small and too far from the middle are not kept 
        when there are somehow many contours.
        """
        input_img = np.zeros((20, 20), dtype=np.uint8) 
        largest_contour = np.array([[[7, 7]], [[7, 13]], [[13, 13]], [[13, 7]]], dtype=np.int32)
        contour_overlapped = np.array([[[6, 6]], [[6, 12]], [[12, 12]], [[12, 6]]], dtype=np.int32) 
        contour_big = np.array([[[3, 3]], [[3, 8]], [[8, 8]], [[8, 3]]], dtype=np.int32) 
        contour_small = np.array([[[18, 18]], [[18, 19]], [[19, 19]], [[19, 18]]], dtype=np.int32) 
        contour_middle = np.array([[[9, 9]], [[9, 11]], [[11, 11]], [[11, 9]]], dtype=np.int32) 
        contours = [largest_contour, contour_overlapped, contour_big, contour_small, contour_middle]
        kept_regions = Utils_Traced.compute_kept_regions(input_img, contours, 2, 6)
        list_regions = []
        for region in kept_regions:
            if region.tolist() not in list_regions:
                list_regions.append(region.tolist())
        result = [largest_contour, contour_overlapped, contour_big, contour_middle]
        # Assert that small contours far from the middle are removed.
        assert len(list_regions) == len(result) 
        assert np.array_equal(list_regions, result) 


    def test_compute_kept_regions_many_regions(self):
        """
        Test that if there are many regions, all of the contours will be kept.
        """
        input_img = np.zeros((20, 20), dtype=np.uint8) 
        largest_contour = np.array([[[7, 7]], [[7, 13]], [[13, 13]], [[13, 7]]], dtype=np.int32)
        contour_overlapped = np.array([[[6, 6]], [[6, 12]], [[12, 12]], [[12, 6]]], dtype=np.int32) 
        contour_big = np.array([[[3, 3]], [[3, 8]], [[8, 8]], [[8, 3]]], dtype=np.int32) 
        contour_small = np.array([[[18, 18]], [[18, 19]], [[19, 19]], [[19, 18]]], dtype=np.int32) 
        contour_middle = np.array([[[9, 9]], [[9, 11]], [[11, 11]], [[11, 9]]], dtype=np.int32) 
        contours = [largest_contour, contour_overlapped, contour_big, contour_small, contour_middle]
        kept_regions = Utils_Traced.compute_kept_regions(input_img, contours, 2, 4)
        list_regions = []
        for region in kept_regions:
            if region.tolist() not in list_regions:
                list_regions.append(region.tolist())
        result = [largest_contour, contour_overlapped, contour_big, contour_small, contour_middle]
        # Assert that all the original contours are kept.
        assert len(list_regions) == len(result) 
        assert np.array_equal(list_regions, result)

    # Tests for the harmonize_traced method.

    def test_harmonize_traced_light_noise(self):
        """
        Test that all the harmonize_traced method calls and operations are done 
        in the right order and with the right input if there is light noise.
        """

        # Define the values that will be used.
        input_img = np.zeros((5,5)).astype(np.uint8)
        raw_img = np.ones((5,5)).astype(np.uint8)
        option = 1
        # Define the values to be returned by mocks.
        clustered_img = np.ones((7,7)).astype(np.uint8) * 255
        min_x = min_y = 2
        max_x = max_y = 5
        shadow_img = np.ones((5,5)).astype(np.uint8) * 4

        # Make mocks for all the functions that will be called in the execution of the 
        # harmonize_traced method. 
        with patch.object(Wavelet, "wavelet_traced") as mock_wavelet_traced, \
                patch.object(Contrast, "contrast_stretch") as mock_contrast_stretch, \
                patch("numpy.clip") as mock_np_clip, \
                patch.object(Contrast, "remove_shadows") as mock_remove_shadows, \
                patch.object(Harmonization, "denoise_traced_heavy_noise") as mock_denoise, \
                patch.object(Harmonization, "threshold_traced_heavy_noise") \
                      as mock_thresh_heavy, \
                patch.object(Harmonization, "threshold_traced_light_noise")  \
                      as mock_thresh_light, \
                patch.object(Utils_Traced, "cluster_pixels") as mock_cluster_pixels:
            
            mock_cluster_pixels.return_value = (clustered_img, (min_x, min_y, max_x, max_y)) 
            mock_remove_shadows.return_value =  shadow_img         
            res = Utils_Traced.harmonize_traced(input_img, raw_img, option)
            remove_shadows_call_args = mock_remove_shadows.call_args


            # Assert that all the calls take place in the right order and with the right arguments.
            mock_wavelet_traced.assert_called_with(input_img, option=option)
            mock_contrast_stretch.assert_called_with(mock_wavelet_traced.return_value)
            mock_np_clip.assert_called_with(mock_contrast_stretch.return_value, 0, 255)
            assert len(remove_shadows_call_args[0]) == 3
            assert remove_shadows_call_args.equals(mock_np_clip.return_value)
            assert (remove_shadows_call_args[0][1] == np.ones((8,8))).all()
            assert remove_shadows_call_args[0][2] == 33
            h = Harmonization(image=mock_remove_shadows.return_value)
            # Assert that the value for estimate_sigma is low, so the image will be processed
            # as not heavily noisy.
            assert estimate_sigma(raw_img) < 1
            mock_thresh_heavy.assert_not_called()
            mock_denoise.assert_not_called()
            mock_thresh_light.assert_called_once()
            mock_cluster_pixels.assert_called_with(mock_thresh_light.return_value, \
                                                    is_restrictive=False)
            # Assert that the returned values are the ones obtained from cluster_pixels.
            assert np.array_equal(res[0], clustered_img) 
            assert np.array_equal(res[1], (min_x, min_y, max_x, max_y))


    def test_harmonize_traced_heavy_noise(self):
        """
        Test that all the harmonize_traced method calls and operations are done 
        in the right order and with the right input if there is heavy noise.
        """

        # Define the values that will be used.
        input_img = np.zeros((5,5)).astype(np.uint8)
        raw_img = np.array([[25, 103, 204, 5, 58],
                              [180, 13, 100, 55, 200],
                              [2, 200, 4, 255, 8],
                              [184, 5, 220, 8, 165],
                              [33, 200, 1, 240, 0]]).astype(np.uint8)
        option = 1
        # Define the values to be returned by mocks.
        clustered_img = np.ones((7,7)).astype(np.uint8) * 255
        min_x = min_y = 2
        max_x = max_y = 5
        shadow_img = np.ones((5,5)).astype(np.uint8)

        # Make mocks for all the functions that will be called in the execution of the 
        # harmonize_traced method. 
        with patch.object(Wavelet, "wavelet_traced") as mock_wavelet_traced, \
                patch.object(Contrast, "contrast_stretch") as mock_contrast_stretch, \
                patch("numpy.clip") as mock_np_clip, \
                patch.object(Contrast, "remove_shadows") as mock_remove_shadows, \
                patch.object(Harmonization, "denoise_traced_heavy_noise") as mock_denoise, \
                patch.object(Harmonization, "threshold_traced_heavy_noise") \
                      as mock_thresh_heavy, \
                patch.object(Harmonization, "threshold_traced_light_noise")  \
                      as mock_thresh_light, \
                patch.object(Utils_Traced, "cluster_pixels") as mock_cluster_pixels:
            
            mock_cluster_pixels.return_value = (clustered_img, (min_x, min_y, max_x, max_y)) 
            mock_remove_shadows.return_value =  shadow_img         
            res = Utils_Traced.harmonize_traced(input_img, raw_img, option)
            remove_shadows_call_args = mock_remove_shadows.call_args

            # Assert that all the calls take place in the right order and with the right arguments.
            mock_wavelet_traced.assert_called_with(input_img, option=option)
            mock_contrast_stretch.assert_called_with(mock_wavelet_traced.return_value)
            mock_np_clip.assert_called_with(mock_contrast_stretch.return_value, 0, 255)
            assert len(remove_shadows_call_args[0]) == 3
            assert remove_shadows_call_args.equals(mock_np_clip.return_value)
            assert (remove_shadows_call_args[0][1] == np.ones((8,8))).all()
            assert remove_shadows_call_args[0][2] == 33
            h = Harmonization(image=mock_remove_shadows.return_value)
            # Assert that the value for estimate_sigma is high, so the image will be processed
            # as heavily noisy.
            assert estimate_sigma(raw_img) > 1
            mock_denoise.assert_called_once()
            mock_thresh_heavy.assert_called_once()
            mock_thresh_light.assert_not_called()
            mock_cluster_pixels.assert_called_with(mock_thresh_heavy.return_value, \
                                                    is_restrictive=False)
            # Assert that the returned values are the ones obtained from cluster_pixels.
            assert np.array_equal(res[0], clustered_img) 
            assert np.array_equal(res[1], (min_x, min_y, max_x, max_y))





# Dummy region object used for testing
class DummyRegionObject:
    def __init__(self, bbox):
        self.bbox = bbox


if __name__ == '__main__':
    pytest.main()  # Run this file with pytest