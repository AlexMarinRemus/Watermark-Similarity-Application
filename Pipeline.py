"""
Object to access individual pipeline steps for the GUI
"""

from skimage.restoration import estimate_sigma

from harmonization.harmonization import Harmonization
from feature_extraction.feature_extraction import FeatureExtraction
from similarity_comparison.compare import Compare
from build_database import loadDB

class Pipeline:
    """
    A class to represent the pipeline of the harmonization process.
    This will make it easier to keep the current state of the image
    after each step.
    """
    def __init__(self, image, raw_image, is_traced, db_path="database/manual_db.pkl"):
        self.raw_image = raw_image
        self.is_traced = is_traced
        self.db_path = db_path
        self.light_noise = estimate_sigma(raw_image, average_sigmas=True) < 1
        self._harmonize = Harmonization(image)

    def pre_process(self, option):
        """
        Pre-processes the image (differently based on if it's traced or not). This should be done before denoising.
        """
        if self.is_traced:
            return self._harmonize.preprocess_traced(option=option+1)
        return self._harmonize.preprocess_untraced()

    def denoise(self, option):
        """
        Denoises the image (different methods are run depending on if the image is untraced, or is traced with heavy
        or light noise). This should be executed before thresholding.
        Args:
            option: A number from 0 to 3 corresponding to denoising strength
        Returns: the denoised image
        """
        if not self.is_traced:
            return self._harmonize.denoise_untraced(sigma_psd=[5, 19, 25, 37][option])
        if not self.light_noise:
            return self._harmonize.denoise_traced_heavy_noise(denoise_sigma=[0.3, 0.5, 0.7, 0.9][option])
        return self._harmonize.get_image()

    def threshold(self, option):
        """
        Thresholds the image (different methods are run depending on if the image is untraced, or is traced with heavy
        or light noise).
        Args:
            option: A number from 0 to 5 corresponding to thresholding strength
        Returns: the thresholded image image
        """
        if not self.is_traced:
            return self._harmonize.threshold_untraced(
                morph_kernel=[(3,3), (3,3), (3,3), (5,5), (5,5), (5,5)][option],
                k=(0.1, 0.2, 0.4, 0.1, 0.2, 0.4)[option])
        if self.light_noise:
            return self._harmonize.threshold_traced_light_noise(k=(0.1, 0.2, 0.4, 0.1, 0.2, 0.4)[option])
        return self._harmonize.threshold_traced_heavy_noise(
            threshold_value=[190, 190, 200, 200, 220, 220][option], \
            closing_shape=[(2,2), (5,5), (2,2), (5,5), (2,2), (5,5)][option])

    def post_process(self, option, wavelet_option):
        """
        Post-processes the image (differently based on if it's traced or not). This should be executed after thresholding.
        """
        if self.is_traced:
            return self._harmonize.post_process_traced(iteration=(1,2,1,2,1,2)[option], raw_img=self.raw_image, wavelet_option=wavelet_option)
        return self._harmonize.post_process_untraced()


    def feature_similarity(self):
        """
        Does the feature extraction of the image, and then calculates the similar images, and returns the first n
        paths for the similar image, as well as their respective similarity scores.
        Args:
            number_to_return: the number of similar images to return.
        Returns: An array of tuples that contain the path of the simlar image and its similarity measure.
        """

        f = FeatureExtraction()
        features = f.extract_features(self._harmonize.get_image())
        db_tuples = loadDB(self.db_path)
        c = Compare(db_tuples)
        ranked_list = c.compare(features)

        return ranked_list

    def get_image(self):
        """
        Gets the image from the pipeline
        """

        return self._harmonize.get_image()
