"""
Builds the database
"""
import argparse
import logging
import os
import sys

import cv2
import pandas as pd

from feature_extraction.feature_extraction import FeatureExtraction
from harmonization.harmonization import Harmonization

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")


def get_args():
    """
    Parses the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="dataset_images/original_dataset/Training/training_untraced",
                        help="Path to the folder of images to be turned into a database pickle file")
    parser.add_argument("--db_name", type=str,
                        default="db", help="Name of the database pickle file. If not specified, \
                        the default name is db.pkl. Will overwrite existing files.")
    parser.add_argument("--is_traced", action="store_true", help="Boolean of if the watermarks in the folder \
                       are traced or not")
    return parser.parse_args()


def makeDB(file_name, path_images, is_traced):
    """
    Serialize the images

    Args:
        file_name: the name of the database file to be created
        path_images: path to the folder of images to be turned into a database pickle file

    Returns:
        None
    """
    db = {"image_path": [], "features": []}

    f = FeatureExtraction()

    try:
        for i, path in enumerate(os.listdir(path_images)):
            logging.disable(logging.NOTSET)
            logger.info("Processing image %d of %d", i+1,
                        len(os.listdir(path_images)))
            logging.disable(logging.CRITICAL)


            image = cv2.imread(path_images+"/"+path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning("The image %s could not be read", path)
                continue

            h = Harmonization(image)

            db["image_path"].append(path_images+"/"+path)
            harmonized = h.harmonize(is_traced)

            db["features"].append(f.extract_features(harmonized))
    except FileNotFoundError:
        logger.exception("The provided image path does not exist")
        sys.exit()

    logging.disable(logging.NOTSET)
    df = pd.DataFrame(db)

    try:
        # If file exists, append to it, and if it doesn't exist create a new file
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                old_df = pd.read_pickle(f)
                new_df = pd.concat([old_df,df]).reset_index(drop=True)
                new_df.to_pickle(file_name)
            logger.info("Data successfully added to database")
        else:
            df.to_pickle(file_name)
            logger.info("Database successfully created")
        logger.info("Saving the database to %s, file size is %s MB",
                file_name, df.memory_usage(deep=True).sum() / 2**20)
    except OSError:
        logger.exception(
            "The database path is invalid, make sure the path is correct and the program has the right permissions")


def loadDB(file):
    """
    Deserialize the images

    Args:
        file: a database file path to be deserialized

    Returns:
        images: a list of tuples of the form (image_path, features)

    """
    logger.info("Loading the database from %s", file)

    try:
        with open(file, "rb") as f:
            df = pd.read_pickle(f)

        l = list(zip(df.image_path, df.features))
        logger.info("Database successfully loaded")
        return l

    except FileNotFoundError:
        logger.exception("File %s could not be found", file)

if __name__ == "__main__":
    args = get_args()
    makeDB("./database/"+args.db_name+".pkl", args.input_path, is_traced=args.is_traced) 
