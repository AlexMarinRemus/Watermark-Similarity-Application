"""
Different similarity methods
"""
import math
import numpy as np
import networkx as nx
import cv2


class Similarity:
    """
    Similarity methods class
    """

    def cosine_similarity(self, vec1, vec2):
        """
        Method that calculates the cosine similarity

        Args:
            vec1: The first vector
            vec2: The second vector

        Returns:
            The cosine similarity between the two vectors
        """
        dot_product = sum(vec1[i] * vec2[i]
                          for i in range(len(vec1)))
        magnitude_vec1 = math.sqrt(sum(x**2 for x in vec1))
        magnitude_vec2 = math.sqrt(sum(x**2 for x in vec2))
        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0
        return dot_product / (magnitude_vec1 * magnitude_vec2)

    def euclidean_similarity(self, vec1, vec2):
        """
        Method that calculates the euclidean similarity

        Args:
            vec1: The first vector
            vec2: The second vector

        Returns:
            The euclidean similarity between the two vectors
        """
        squared_difference = sum(
            (vec1[i] - vec2[i])**2 for i in range(len(vec1)))
        return 1 / (1 + math.sqrt(squared_difference))

    def manhattan_distance(self, vec1, vec2):
        """
        Method that calculates the manhattan distance

        Args:
            vec1: The first vector
            vec2: The second vector

        Returns:
            The manhattan distance between the two vectors
        """
        return sum(abs(el1 - el2) for el1, el2 in zip(vec1, vec2))

    def compare_histograms(self, vec1, vec2):
        """
        Method that calculates the histogram comparison

        Args:
            vec1: The first vector
            vec2: The second vector

        Returns:
            The histogram comparison between the two vectors
        """
        return cv2.compareHist(vec1, vec2, cv2.HISTCMP_CORREL)

    def graph_matching_kernel_similarity(self, vec1, vec2):
        """
        Method the calculated the similarity between the arrays using a graph based method.

        Args:
            vec1: The first vector
            vec2: The second vector

        Returns:
            The similarity between the two vectors
        """
        # Convert the input to a graph
        GraphA = nx.from_numpy_array(np.array(vec1))
        GraphB = nx.from_numpy_array(np.array(vec2))

        # Compute the shortest path between the 2 graphs
        spk = self.shortest_path_kernel([GraphA, GraphB])

        # Compute the shortest path based on the shortest path
        sim = spk[0][1] / np.sqrt(spk[0][0] * spk[1][1])

        return sim

    def match_features(self, vec1, vec2, ratio=0.75):
        """
        Matches two sets of local features and returns the number of good matches

        Args:
            vec1: The first vector
            vec2: The second vector
            ratio: The ratio to use for the ratio test

        Returns:
            The number of good matches, according to the ratio test
        """
        bf = cv2.BFMatcher()

        feature1 = np.float32(vec1)
        feature2 = np.float32(vec2)

        matches = bf.knnMatch(feature1, feature2, k=2)
        # a good match is one that satisfy the condition "first_match.distance < ratio * second_match.distance"
        good_matches = []
        for first_match, second_match in matches if len(matches) > 0 and len(matches[0]) == 2 else []:
            if first_match.distance < ratio * second_match.distance:
                good_matches.append(first_match)

        return len(good_matches)

    def shortest_path_kernel(self, graphs):
        """
        Method that finds the shortest path in the kernel between the graphs

        Args:
            graphs: The graphs to find the shortest path between

        Returns:
            The shortest path between the graphs
        """
        n = len(graphs)
        K = np.zeros((n, n))

        # Compute the shortest path kernel for each pair of graphs
        for i in range(n):
            for j in range(i, n):
                # Compute the sum of the average shortest path length of both graphs minus the
                # average shortest path length of the disjoint union of the two graphs
                K[i][j] = K[j][i] = nx.average_shortest_path_length(graphs[i]) \
                    + nx.average_shortest_path_length(graphs[j]) \
                    - 2 * \
                    nx.average_shortest_path_length(
                        nx.disjoint_union(graphs[i], graphs[j]))

            return K
