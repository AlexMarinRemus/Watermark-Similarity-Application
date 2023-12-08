"""
Extract the features from an image
"""
import logging

import cv2

from feature_extraction.strategies import DetectionStrategy
from feature_extraction.strategies import ExtractionStrategy

logger = logging.getLogger(__name__)


class FeatureExtraction:
    """
    Feature extraction class
    """

    def extract_features(self, image):
        """
        Extracts features from the given image. This is the method that will be called
        by main.py so it should encompass the entire feature extraction process that
        will be used by the system.

        Args:
            image: The image to extract features from

        Returns:
            A list of features extracted from the image, namely (good_sift_descriptors, hu_moments, zernike_moments)
        """
        logger.info("Detecting and extracting features...")

        sift = cv2.SIFT_create()
        ds = DetectionStrategy.SIFT(sift)
        es = ExtractionStrategy.SIFT(sift)
        es_hu_moments = ExtractionStrategy.Hu_moments()
        es_zernike_moments = ExtractionStrategy.Zernike_moments()

        keypoints = ds.detect(image)
        descriptors = es.extract(image, keypoints)
        hu_moments = es_hu_moments.extract(image, keypoints)
        zernike_moments = es_zernike_moments.extract(image)
        return (descriptors, hu_moments, zernike_moments)
