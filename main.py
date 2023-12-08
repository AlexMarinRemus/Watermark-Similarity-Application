"""
Main module for console running
"""
import argparse
import logging
import sys
import time

import cv2
import matplotlib.pyplot as plt

from build_database import loadDB
from feature_extraction.feature_extraction import FeatureExtraction
from harmonization.harmonization import Harmonization
from similarity_comparison.compare import Compare

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")


def get_args():
    """
    Parses the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="dataset_images/original_dataset/Training/training_traced/1_1.jpg", help="Path to the input image")
    parser.add_argument("--db_path", type=str,
                        default="database/manual_db.pkl", help="Path to the database pickle file")
    parser.add_argument("--is_traced", action="store_true",
                        help="Boolean of if the watermark is traced")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode prints extra information")
    parser.add_argument("--number_of_output", type=int, default=20, help="Number of ranked images \
                        to output")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()
    file_path = args.input_path
    db_path = args.db_path
    is_traced = args.is_traced
    debug = args.debug
    number_of_output = args.number_of_output

    t0 = time.time_ns()
    if file_path.strip() == "":
        logger.exception("The image path is empty")
        sys.exit(0)

    try:
        image = cv2.imread(file_path, 0)
    except Exception as e:
        logger.exception("Exception occurred while code Execution: %s", e)
        sys.exit()

    if image is None:
        logger.error("The image %s could not be read", file_path)
        sys.exit(0)

    h = Harmonization(image)
    image = h.harmonize(is_traced)

    f = FeatureExtraction()
    features = f.extract_features(image)

    c = Compare(loadDB(db_path))
    ranked_list = c.compare(features)[:number_of_output]

    if debug:
        fig, axs = plt.subplots(4, 5)
        for i in range(4):
            for j in range(5):
                try:
                    # score = round(ranked_list[i*4+j][3], 2) # Zernike
                    # score = round(ranked_list[i*4+j][1], 2) # SIFT
                    score = round(ranked_list[i*4+j][3]/ranked_list[i*4+j][1], 2)  #  Combined Zernike and SIFT
                    good_matches = round(ranked_list[i*4+j][1], 2)
                    hu_moments_distance = round(ranked_list[i*4+j][2], 2)
                    zernike_moments_distance = round(ranked_list[i*4+j][3], 2)
                    img = cv2.imread(
                        ranked_list[i*4+j][0], cv2.IMREAD_GRAYSCALE)
                    axs[i][j].imshow(img, cmap="gray")
                    axs[i][j].set_title(
                        f"Rank: {i*5+j+1}, Distance: {round(score, 2)}")

                    axs[i][j].axis("off")

                    fig.suptitle("Ranked list of images")

                except FileNotFoundError:
                    logger.error("Image %s not found", ranked_list[i][0])
        plt.show()

    ranked_list = str(list(zip(range(1, len(ranked_list) + 1), [(x[0], x[4]) for x in ranked_list])))
    ranked_list = ranked_list.replace(")), (", "\n").replace(
        "(", "").replace(")", "").replace("'", "")

    # Important timing to check if it takes 1 month to run
    logger.info("The time elapsed is: %f seconds",
                float(time.time_ns() - t0)/1e9)
    logger.info("Ranking: \n%s", ranked_list)
