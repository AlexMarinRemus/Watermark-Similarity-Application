"""
This file contains the class that tests all compare methods.
Compare methods can be found in 'similarity_comparison/compare.py'.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import patch, mock_open, call, Mock
from unittest import TestCase

from similarity_comparison.compare import Compare
from similarity_comparison.similarity import Similarity


class TestCompare:
    """
    Class that the tests the Compare class.
    """

    # Tests for the init of the db attribute.
    
    def test_init(self):
        """
        Test that db attribute is correctly assigned when initializing an 
        instance of the Compare class.
        """

        db = {"image_path": ["images/image1.png"], "features": [[0,1,2]]}
        c = Compare(db)
        # Assert that the db attribute is the same one as the one passed
        # when initializing the class object. 
        assert c.db == db


    # Tests for the compare method.

    def test_compare_one_elem(self):
        """
        Test that if the database has one element the similarity methods run properly and the
        returned ranked list only has one element with score 0.
        """

        # Initialize the variables to be used.
        db = [("images/image1.png", [[2],[3],[4]])]
        f1 = [[1],[4],[10]]

        # Mock the methods to be called during the execution of the compare method.
        with patch.object(Similarity, "match_features", side_effect=[40]) as mock_match_features, \
             patch.object(Similarity, "manhattan_distance", side_effect=[0.5, 0.25]) as mock_manhattan_dist, \
             TestCase.assertLogs(self) as captured:
            
            # Call the method being tested.
            c = Compare(db)
            result = c.compare(f1) 

            # Assert that the calls to the method were done with the right arguments.
            mock_match_features.assert_called_once_with([1],[2],0.75)
            calls_manhattan_dist = [call([4], [3]), call([10], [4])]
            mock_manhattan_dist.assert_has_calls(calls_manhattan_dist)

            # Assert that the returned value is the expected one.
            rank_list = [['images/image1.png', 40, 0.5, 0.25, 0.0]]
            assert result == rank_list

            # Assert that the log message is properly created.
            assert len(captured.records) == 1
            assert captured.records[0].getMessage() == "Comparing the input image with 1 images in the database"

    def test_compare_two_elems(self):
        """
        Test that if there are two elements in the database, they will be ordered in decreasing order 
        of their scores and the methods will be called in the right order with the right arguments.
        """

        # Initialize the database and the feature vector used for comparison.
        db = [("images/image1.png", [[2],[3],[4]]), ("images/image2.png", [[10], [15], [22]])]
        f1 = [[2],[4],[5]]

        # Mock the methods that will be called during the execution of the compare method.
        with patch.object(Similarity, "match_features", side_effect=[9, 81]) as mock_match_features, \
             patch.object(Similarity, "manhattan_distance", side_effect=[0.3, 1.5, 0.9, 3]) as mock_manhattan_dist, \
             TestCase.assertLogs(self) as captured:
            
            # Call the method being tested.
            c = Compare(db) 
            result = c.compare(f1)
            
            # Assert that the calls to the similarity methods are done with the right arguments in the
            # right order.
            calls_match_features = [call([2], [2], 0.75), call([2], [10], 0.75)]
            mock_match_features.assert_has_calls(calls_match_features)
            calls_manhattan_dist = [call([4],[3]), call([5],[4])]
            mock_manhattan_dist.assert_has_calls(calls_manhattan_dist)

            # Assert that the first element in the list of results contains the first set of side_effect values 
            # from the mock object.
            assert 'images/image1.png' in result[0]
            assert 9 in result[0]
            assert 0.3 in result[0]
            assert 1.5 in result[0]
            # Assert that the second element contains the distances from the second call of the mock objects.
            assert 'images/image2.png' in result[1]
            assert 81 in result[1]
            assert 0.9 in result[1]
            assert 3 in result[1]
            # Assert that the results are ordered in the list according to their score in decreasing order.
            assert result[0][4] > result[1][4]

            # Assert that the log message is as expected.
            assert len(captured.records) == 1
            assert captured.records[0].getMessage() == "Comparing the input image with 2 images in the database"

    def test_compare_none_f2(self):
        """
        Test that the logger gives warnings if the feature vector from the database is None or is an empty list.
        """
        
        # Initialize the dataset and the feature vector to do the comparison with.
        db = [("images/image1.png", [[2],[3],[4]]), ("images/image2.png", [None, [15], [22]]), \
              ("images/image3.png", [[], [1], [13]])]
        f1 = [[2],[4],[5]]

        # Mock the methods that will be called during the execution.
        with patch.object(Similarity, "match_features", side_effect=[9]) as mock_match_features, \
             patch.object(Similarity, "manhattan_distance", side_effect=[0.3, 1.5]) as mock_manhattan_dist, \
             TestCase.assertLogs(self) as captured:
            
            # Call the method being tested with its argument.
            c = Compare(db) 
            result = c.compare(f1) 

            # Assert that the mocks were called with the right arguments.
            mock_match_features.assert_called_once_with([2],[2],0.75)
            calls_manhattan_dist = [call([4], [3]), call([5], [4])]
            mock_manhattan_dist.assert_has_calls(calls_manhattan_dist)

            # Assert that only the first element of the database exists in the result list, as the other
            # two are invalid.
            assert len(result) == 1
            assert result == [['images/image1.png', 9, 0.3, 1.5, 0.0]]

            # Assert that the logger registers the right messages and warnings for the given elements.
            assert len(captured.records) == 3
            assert captured.records[0].getMessage() == "Comparing the input image with 3 images in the database"
            assert "No features were extracted from images/image2.png," in captured.records[1].getMessage()
            assert "this typically means the feature extraction method couldn't find anything." \
                in captured.records[1].getMessage()
            assert "No features were extracted from images/image3.png," in captured.records[2].getMessage()
            assert "this typically means the feature extraction method couldn't find anything." \
                in captured.records[2].getMessage()


if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
