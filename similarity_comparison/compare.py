"""
Compares the input watermark to the database
"""
import logging
import numpy as np

from similarity_comparison.similarity import Similarity

logger = logging.getLogger(__name__)


class Compare:
    """
    Comparison class
    """

    def __init__(self, db):
        self.db = db

    def compare(self, f1):
        """Compare the image with the database

        Args:
            features: a numpy array of features to be compared with the database

        Returns:
            A ranking in the form of a list of lists of the form
            [image_path, good_sift_matches, hu_moments_distance, zernike_moments_distance, score],
            sorted on: (percentage of good sift matches * complement of percentage of zernike distance
            * complement of percentage of hu distance)
        """
        logger.info(
            "Comparing the input image with %d images in the database", len(self.db))

        similarity = Similarity()
        result = []
        for (filepath, f2) in self.db:
            if f2[0] is not None and len(f2[0]) != 0:
                good_matches = similarity.match_features(f1[0], f2[0], 0.75)
                hu_moments_distance = similarity.manhattan_distance(
                    f1[1], f2[1])
                zernike_moments_distance = similarity.manhattan_distance(
                    f1[2], f2[2])

                result.append([filepath, good_matches, hu_moments_distance,
                              zernike_moments_distance, 0])
            else:
                logger.warning(
                    """No features were extracted from %s,
                    this typically means the feature extraction method couldn't find anything.""", filepath)

        max_zernike_distance = max(result, key=lambda x: x[3])[3]
        max_hu_distance = max(result, key=lambda x: x[2])[2]
        max_sift_matches = max(result, key=lambda x: x[1])[1]

        for entry in result:
            #
            sift_percentage = entry[1]/max_sift_matches
            zernike_percentage = 1-entry[3]/max_zernike_distance
            hu_percentage = 1-entry[2]/max_hu_distance

            # The scores are calculated by taking the geometric mean of the percentages.
            # The actual calculation is modified for readability, but the ordering of
            # the operations is the same.
            # The square root is taken of the percentages, these are multiplied and it's
            # squared again to "undo" the square roots. Then finally the cube root is taken.
            entry[4] = ((np.sqrt(sift_percentage) * np.sqrt(zernike_percentage) * np.sqrt(hu_percentage))**2)**(1/3)

        return sorted(result, key=lambda x: x[4], reverse=True)
