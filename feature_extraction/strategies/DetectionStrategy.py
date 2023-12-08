"""
Provides different strategies for detecting keypoints in an image. The strategies are implemented as subclasses of
Strategy, and can be used interchangeably to use different techniques.
"""
from abc import ABC, abstractmethod
import cv2
import numpy as np


class _Strategy(ABC):
    """
    Abstract class for a detection strategy
    """
    @abstractmethod
    def detect(self, image):
        pass


class SIFT(_Strategy):
    """
    Detects keypoints using the SIFT algorithm
    """

    def __init__(self, sift):
        self.sift = sift

    def detect(self, image):
        return self.sift.detect(image, None)


class Harris(_Strategy):
    """
    Detects keypoints using the Harris corner detection algorithm
    """

    def _binary_to_keypoints(self, image):
        """
        Convert a binary image to a list of keypoints
        """
        points = np.argwhere(image > 0)
        result = []
        for point in points:
            result.append(cv2.KeyPoint(float(point[1]), float(point[0]), 1))

        return tuple(result)

    def detect(self, image):
        return self._binary_to_keypoints(cv2.cornerHarris(image, 2, 3, 0.04))
