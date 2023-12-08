"""
Provides different strategies for extracting features from an image. The strategies are implemented as subclasses of
Strategy, and can be used interchangeably to use different techniques.
"""
from abc import ABC, abstractmethod
import cv2
import numpy as np
import skimage
import scipy as sp
import mahotas.features as mh
from sklearn.decomposition import PCA


class _Helper:
    """
    A class that provides helper methods for feature extraction
    """

    def gabor_images(self, image, energy=False):
        """
        Applies a bank of Gabor filters to the given image. In this case we use 8 orientations (theta)
        and 5 frequencies resulting in 40 filtered images.

        Args:
            image: The image to apply the filters to
            energy: Whether to use the energy of the filtered images instead of the images themselves.
            The energy is the squared value of the filtered image (for non-linearity), smoothed with a Gaussian filter.

        Returns:
            A list of images, each one the result of applying a different Gabor filter to the given image
        """
        filtered_images = []
        for theta in range(8):
            for freq in np.arange(0.1, 0.6, 0.1):
                kernel = np.real(skimage.filters.gabor_kernel(
                    freq, theta=theta * np.pi / 4))
                filtered = cv2.filter2D(image, cv2.CV_8UC1, kernel)

                if energy:
                    filtered = np.power(filtered, 2)
                    filtered = cv2.GaussianBlur(filtered, (5, 5), 0)

                filtered_images.append(filtered)

        return filtered_images

    def train_PCA(self, images):
        """
        Trains a PCA model on the given images. This is used to reduce the dimensionality of the Gabor features.
        This keeps enough components to explain 99% of the variance in the data. Typically this is only 100-200
        components, instead of the thousands of pixels in the original images.

        Args:
            images: The images to train the PCA model on

        Returns:
            A trained PCA model
        """
        feature_vectors = []
        for image in images:
            gabor_images = self.gabor_images(image)
            feature_vectors.extend([sp.fftpack.dct(image).flatten()
                                   for image in gabor_images])

        feature_vectors = np.array(feature_vectors)

        pca = PCA(n_components=0.99)
        pca.fit(feature_vectors)
        # print("PCA components:", pca.n_components_)
        return pca


class _Strategy(ABC):
    """
    Abstract class for a feature extraction strategy
    """
    @abstractmethod
    def extract(self, image, keypoints=None):
        pass


class SIFT(_Strategy):
    """
    Extracts SIFT features from the given image using the given SIFT keypoints
    """

    def __init__(self, sift):
        self.sift = sift

    def extract(self, image, keypoints=None):
        """
        Uses the given SIFT object to extract SIFT descriptors from the given image

        Args:
            image: The image to extract features from
            keypoints: The keypoints to use to extract features
            sift: The SIFT object to use to extract features

        Returns:
            A numpy array of the SIFT descriptors
        """
        assert keypoints is not None, "Keypoints must be provided for SIFT"
        return self.sift.compute(image, keypoints)[1]


class LBP_image(_Strategy):
    """
    Extracts Local Binary Pattern (LBP) features from the given image around the given keypoints.
    """

    def __init__(self, method="default", patch_size=64):
        self.method = method
        self.patch_size = patch_size

    def extract(self, image, keypoints=None):
        """
        Extracts Local Binary Pattern (LBP) features from the given image using the given method.
        This describes a 64x64 patch of pixels around each keypoint with the LBP pixel values.
        By default this is not rotation invariant, but other methods (like "uniform") are.

        Args:
            image: The image to extract features from
            keypoints: The keypoints to use to extract features
            method: The method to use for LBP,
            see https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern

        Returns:
            A numpy array of the LBP features
        """
        assert keypoints is not None, "Keypoints must be provided for LBP image"
        result = []

        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            patch = image[max(y - self.patch_size, 0): min(y + self.patch_size, image.shape[0]),
                          max(x - self.patch_size, 0): min(x + self.patch_size, image.shape[1])]

            lbp = skimage.feature.local_binary_pattern(
                patch, 8, 1, method=self.method)

            result.append(lbp.flatten())

        return np.array(result)


class LBP_histogram(_Strategy):
    """
    Extracts Local Binary Pattern (LBP) histogram as a feature from the given image around the given keypoints.
    """

    def __init__(self, method="default", patch_size=64):
        self.method = method
        self.patch_size = patch_size

    def extract(self, image, keypoints=None):
        """
        Extracts Local Binary Pattern (LBP) features from the given image using the given method.
        This describes a 64x64 patch of pixels around each keypoint with a histogram of the
        LBP pixel values, discarding their locations.
        If no keypoints are provided, the LBP histogram is computed for the entire image.
        By default this is not rotation invariant, but other methods (like "uniform") are.

        Args:
            image: The image to extract features from
            keypoints: The keypoints to use to extract features
            method: The method to use for LBP,
            see https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern

        Returns:
            A numpy array of shape (len(keypoints), 256) containing the LBP histograms for each keypoint
        """
        if keypoints is None:
            lbp = skimage.feature.local_binary_pattern(
                image, 8, 1, method=self.method)
            return [np.histogram(lbp, bins=256, range=(0, 255))[0]]
        result = []

        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            patch = image[max(y - self.patch_size, 0): min(y + self.patch_size, image.shape[0]),
                          max(x - self.patch_size, 0): min(x + self.patch_size, image.shape[1])]

            lbp = skimage.feature.local_binary_pattern(
                patch, 8, 1, method=self.method)
            hist = np.histogram(lbp, bins=256, range=(0, 255))[0]
            result.append(hist)

        return np.array(result).astype(np.float32)


class Gabor(_Strategy):
    """
    Extracts Gabor features from the given image. This is done by applying a bank of Gabor filters to the image
    and using the mean and variance of each as features.
    """

    def extract(self, image, keypoints=None):
        """
        Extracts Gabor features from the given image. This is done by applying a bank of Gabor filters to the
        image and extracting a feature vector for each image. The feature vector consists of the mean and variance.

        Args:
            image: The image to extract features from
            keypoints: The keypoints to use to extract features

        Returns:
            A numpy array of length 2*len(keypoints) containing the mean and variance of each Gabor-filtered image
        """
        filtered_images = _Helper().gabor_images(image, energy=True)

        feature_vectors = [[np.mean(img), np.var(img)] for img in filtered_images]
        return (np.array(feature_vectors)).flatten()


class Gabor_DCT(_Strategy):
    """
    Extracts Gabor-DCT features from the given image. The DCT (Discrete Cosine Transform) is applied to each Gabor-filtered image and the
    flattened DCT is used as a feature vector. A pre-trained PCA model is used to reduce the dimensionality of the
    feature vectors.
    """

    def __init__(self, images):
        """
        Initializes the Gabor-DCT feature extractor by training a PCA model on the given images.
        """
        self.pca_model = _Helper().train_PCA(images)

    def extract(self, image, keypoints=None):
        """
        Extracts Gabor-DCT features from the given image. This is done by applying a bank of Gabor filters to the
        image, then applying the discrete cosine transform to each image and treating the flattened DCT as a feature vector.
        Finally, PCA is used to reduce the dimensionality of the feature vectors.

        Args:
            image: The image to extract features from
            keypoints: The keypoints to use to extract features

        Returns:
            A numpy array containing the PCA-reduced feature vectors
        """

        filtered_images = _Helper().gabor_images(image, energy=True)

        # Take the discrete cosine transform of each image and flatten it
        filtered_images = [sp.fftpack.dct(image).flatten() for image in filtered_images]

        # Use PCA to reduce dimensionality of feature vectors
        feature_vectors = self.pca_model.transform(filtered_images)

        return (np.array(feature_vectors)).flatten()

class Hu_moments(_Strategy):
    """
    Extracts Hu moments from the given image. These are used to describe the shape of the image.
    They are invariant to translation, rotation and scaling, and based on the image's central moments.
    Described in more detail in Ming-Kuei Hu's paper "Visual pattern recognition by moment invariants"
    """
    def extract(self, image, keypoints=None):
        hu_moments = cv2.HuMoments(cv2.moments(image)).flatten()
        # Scale to bring all values in the same range
        return np.array([-1*np.sign(h) * np.log10(abs(h)) if h != 0 else 0 for h in hu_moments]).astype(np.float32)

class Zernike_moments(_Strategy):
    """
    Extracts Zernike moments from the given image. They are based on the image's radial moments,
    and are invariant to translation, rotation and scaling. Zernike polynomials are used to describe
    the shape of the image. The radial moments are computed up to the radius of the image.
    """
    def extract(self, image, keypoints=None):
        zernike_moments = mh.zernike_moments(image, max(image.shape[0], image.shape[1])//2)
        # Scale to bring all values in the same range
        if zernike_moments is None:
            return np.empty(image.shape)
        return np.array([-1*np.sign(h) * np.log10(abs(h)) if h != 0 else 0 for h in zernike_moments]).astype(np.float32)