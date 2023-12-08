"""
Test similarity methods
"""
import pytest
import math
import numpy as np
import networkx as nx
import cv2
from unittest.mock import patch, mock_open, call, Mock
from unittest import TestCase
from similarity_comparison.similarity import Similarity


class TestSimilarity:
    """
    Test the similarity class
    """

    def test_cosine_similarity(self):
        """
        Test the cosine similarity method
        """
        vec1 = [1, 2, 3]
        vec2 = [4, 5, 6]
        sim = Similarity()
        expected_result = 0.97463184
        result = sim.cosine_similarity(vec1, vec2)
        pytest.approx(result, expected_result)

    def test_cosine_similarity_zero(self):
        """
        Test cosine similarity method with one 0 vector
        """
        vec1 = [1, 2, 3]
        vec2 = [0, 0, 0]
        sim = Similarity()
        expected_result = 0
        result = sim.cosine_similarity(vec1, vec2)
        pytest.approx(result, expected_result)

    def test_euclidean_similarity(self):
        """
        Test the euclidean similarity method
        """
        vec1 = [1, 2, 3]
        vec2 = [4, 5, 6]
        sim = Similarity()
        expected_result = 0.97463184
        result = sim.euclidean_similarity(vec1, vec2)
        pytest.approx(result, expected_result)

    def test_euclidean_similarity_zero(self):
        """
        Test the euclidean similarity method with one 0 vector
        """
        vec1 = [1, 2, 3]
        vec2 = [0, 0, 0]
        sim = Similarity()
        expected_result = 0
        result = sim.euclidean_similarity(vec1, vec2)
        pytest.approx(result, expected_result)

    def test_manhattan_distance(self):
        """
        Test Manhattan distance method
        """
        vec1 = [1, 2, 3]
        vec2 = [4, 5, 6]
        sim = Similarity()
        expected_result = 9
        result = sim.manhattan_distance(vec1, vec2)
        pytest.approx(result, expected_result)

    def test_manhattan_distance_zero(self):
        """
        Test Manhattan distance method with 2 zero vectors
        """
        vec1 = [0, 0, 0]
        vec2 = [0, 0, 0]
        sim = Similarity()
        expected_result = 0
        result = sim.manhattan_distance(vec1, vec2)
        pytest.approx(result, expected_result)

    def test_manhattan_distance_negative(self):
        """
        Test Manhattan distance method
        """
        vec1 = [1, 2, 3]
        vec2 = [-4, -5, -6]
        sim = Similarity()
        expected_result = 18
        result = sim.manhattan_distance(vec1, vec2)
        pytest.approx(result, expected_result)

    def test_compare_histograms(self):
        """
        Test the compare histograms method.
        """

        vec1 = np.array([0.1, 0.2, 0.3]).astype(np.float32)
        vec2 = np.array([0.4, 0.5, 0.6]).astype(np.float32)
        sim = Similarity() 
        result = sim.compare_histograms(vec1, vec2) 
        pytest.approx(result, 1) 

    def test_graph_matching_kernel_similarity(self):
        """
        Test the graph_matching_kernel_similarity method.
        """

        # Initialize the 2 vectors to be used.
        vec1 = [[1,1],[2,3]]
        vec2 = [[4,4],[5,6]]
        
        # Mock the shortest_path_kernel method.
        with patch.object(Similarity, "shortest_path_kernel") as mock_shortest_path:

            # Declare the value of the return value of the mock.
            mock_shortest_path.return_value = [[2,6],[3,8]]

            # Call method being tested with the 2 vectors.
            sim = Similarity()
            result = sim.graph_matching_kernel_similarity(vec1, vec2)
            
            # Assert that the mock is called with the right arguments.
            shortest_path_args = mock_shortest_path.call_args   
            assert shortest_path_args.equals((nx.from_numpy_array(np.array(vec1)), \
                                              nx.from_numpy_array(np.array(vec2))))
            
            # Assert that the expected result is the same as the actual one
            # by the formula in the method, 6/(2*8) = 1.5
            expected_res = 1.5
            assert result == expected_res

    def test_match_features_more_matches(self):
        """
        Test that the method computes correctly the number of good matches for 
        a list of matches with more elements.
        """

        # Initialize the vectors.
        vec1 = [1,2,3]
        vec2 = [4,5,6]

        # Make dummy objects to control the distances and the number of returned
        # good matches.
        match_11 = DummyMatchObject(distance=2)
        match_12 = DummyMatchObject(distance=3)
        match_21 = DummyMatchObject(distance=0.5)
        match_22 = DummyMatchObject(distance=2)
        match_31 = DummyMatchObject(distance=0.75)
        match_32 = DummyMatchObject(distance=1)

        # Make the list of dummy matches to be used by the knnMatch.
        matches = [(match_11, match_12), (match_21, match_22), (match_31, match_32)]

        # Mock the BFMatcher.
        with patch("cv2.BFMatcher") as mock_bf_matcher:

            # Set the return value of the knnMatch to be the list of dummy match objects
            # and call the method with the vectors.
            mock_bf_matcher.return_value.knnMatch.return_value = matches 
            sim = Similarity()
            res = sim.match_features(vec1, vec2, ratio=0.75)

            # Assert that the mock and the knnMatch are called with the right arguments.
            mock_bf_matcher.assert_called_once()
            knnMatch_call_args = mock_bf_matcher.return_value.knnMatch.call_args
            assert knnMatch_call_args.equals(
                (np.float32(vec1), np.float32(vec2), 2))
            
            # Assert that the results is as expected.
            # Here the first 2 pairs of dummy match objects fulfill the condition with the
            # ratio, while the 3rd pair does not, so the result is 2 good matches.
            assert res == 2

    def test_match_features_no_good_matches(self):
        """
        Test that the number of good matches is zero if no match is found or if the
        first set of matches contains more or less than 2 elements.
        """

        # Initialize the vectors as empty lists.
        vec1 = []
        vec2 = []

        # Call the method with the vectors and assert that the number of good matches is 0
        # because no match was found.
        sim = Similarity()
        res = sim.match_features(vec1, vec2, ratio=0.75)
        assert res == 0

        # Create matches list with first element having length 3.
        matches = [(DummyMatchObject(distance=3), DummyMatchObject(distance=3), DummyMatchObject(distance=4))]

        # Mock the BFMatcher.
        with patch("cv2.BFMatcher") as mock_bf_matcher:
            # Set the return of knnMatch to be the list of dummy matches and call match_features
            # method with two vectors.
            mock_bf_matcher.return_value.knnMatch.return_value = matches 
            sim = Similarity()
            res = sim.match_features([1,2], [3,4], ratio=0.75)

            # Assert that all methods are called with the right arguments.
            mock_bf_matcher.assert_called_once()
            knnMatch_call_args = mock_bf_matcher.return_value.knnMatch.call_args
            assert knnMatch_call_args.equals(
                (np.float32(vec1), np.float32(vec2), 2))
            
            # Assert that no good match was found because matches[0] = 3.
            assert res == 0

    def test_shortest_path_kernel(self):
        """
        Test that shortest_path_kernel performs all operations in the right
        order and the result is of the right shape.
        """

        # Initialize the vectors and graphs to use.
        vec1 = [[1,2], [2,3]]
        vec2 = [[1,4], [4,2]]
        G1 = nx.from_numpy_array(np.array(vec1))
        G2 = nx.from_numpy_array(np.array(vec2))
        graphs = [G1, G2]
        
        # Mock the methods being called.
        with patch("networkx.average_shortest_path_length") as mock_shortest_path, \
            patch("networkx.disjoint_union") as mock_disjoint_union: 
            # Call the method being tested.
            mock_disjoint_union.return_value = nx.disjoint_union(G1, G2)
            sim = Similarity()
            res = sim.shortest_path_kernel(graphs)
            # Assert that the result has the right shape.
            assert res.shape[0] == 2
            assert res.shape[1] == 2



# Dummy match object used for testing.
class DummyMatchObject:
    def __init__(self, distance):
        self.distance = distance


if __name__ == "__main__":
    pytest.main()
