"""
Evaluates the system for the given database using the evaluation set.
"""
import argparse
import cv2
import logging
import os
import sys
from alive_progress import alive_bar
import matplotlib.pyplot as plt 
import pandas as pd

from build_database import loadDB
from harmonization.harmonization import Harmonization
from similarity_comparison.compare import Compare
from feature_extraction.feature_extraction import FeatureExtraction

def get_args():
    """
    Parses the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="dataset_images/original_dataset/Evaluation/")
    parser.add_argument("--db_path", type=str, default="database/auto_db.pkl")
    arguments = parser.parse_args()
    return arguments


def compute_ranked_list(image, is_traced):   
    """
    Args:
        image: The image from the evaluation set.
        is_traced: bool expressing whether the image is traced or not.
    
    Returns:
        The list of ranks corresponding to the given image from the 
        database.

    Method that takes an image from the evaluation set and performs harmonization,
    feature extraction, and similarity checking by comparing it to the given 
    database. 
    """ 
    try:
        h = Harmonization(image)
        image = h.harmonize(is_traced)

        f = FeatureExtraction()
        features = f.extract_features(image)

        c = Compare(loadDB(db_path))
        ranked_list = c.compare(features)

    except Exception:
        ranked_list = []

    return ranked_list


def format_ranked_list(image_name, ranked_list):
    """
    Args:
        image_name: The name of the image of the form X_X.jpg/.png.
        ranked_list: The list of rankings for the given image.

    Returns:
        A list of tuples containing the (rank, name, type) of 
        each image match from the ranked list. A match
        is a picture of the same watermark as the input image. 
        'rank' is the index of the match in ranked_list, 'match_name'
        corresponds to the name of the match, 'match_type' is the type
        (either traced or untraced) of the match.

    Method that takes the image_name and its corresponding ranked_list
    and formats it as a list of tuples having the form 
    (rank, match_name, match_type), for matches of the input image
    in the list.
    """
    db_name_type = []
    for img_data in ranked_list:
        path_img = img_data[0].split("/")
        # For every element in the ranked list, extract the image name
        # and the file name which will determine if the database image
        # was traced or not.
        name_img = path_img[len(path_img)-1]
        file_name = path_img[len(path_img)-2]
        # Make a list of tuples of the form (name, type) for each element.
        db_name_type.append((name_img, \
                                    "traced" if file_name.endswith("_traced") \
                                    else "untraced" if file_name.endswith("_untraced") \
                                    else "unknown"))
    db_name_type = list(db_name_type)
    # If the ranked_list was empty, return an empty list.
    if len(db_name_type) == 0:
        return db_name_type
    
    # Extract the elements of the db_name_type list for which the first part of
    # the image name (which corresponds to the index of the watermark in the 
    # dataset) is equal to the one of the given image. Then append to the
    # a tuple of the form (match_rank, match_name, match_type) to the list of 
    # matches_ranks. 
    matches_ranks = [(i+1, match_name, match_type) \
                for i, (match_name, match_type) in enumerate(db_name_type) \
                if match_name.split(".")[0].split("_")[0] == \
                image_name.split(".")[0].split("_")[0]]

    return matches_ranks


def generate_ranking_details(db_name, traced_ranked, untraced_ranked):
    """
    Args: 
        traced_ranked: A list of 
                    (image_name, number_of_top_ranked_matches, matches_ranks)
                    tuples for the traced watermark images from the evaluation 
                    set.
        untraced_ranked: A list of 
                    (image_name, number_of_highly_ranked_matches, matches_ranks)
                    tuples for the untraced watermark images from the evaluation 
                    set.
        db_name: The name of the database being used for evaluation.
    
    Returns: 
        A list of all rank positions for the entire evaluation set.
    
    Method that creates or overwrites the evaluation_ranking_details.txt file, 
    which contains information about the ranks and types of matches for both 
    traced and untraced watermarks in the evaluation dataset. This provides 
    additional insights into the comparison and matching of watermarks of 
    different types compared to watermarks of the same type, as well as the
    general ranking positions distribution of matches belonging to each type.
    A list of the ranking placements for all images will be then returned.
    """
    file_name = f"evaluation/evaluation_results_{db_name}.txt"

    with open(file_name, 'w') as eval_file:
        eval_file.write("Evaluation Set - Traced Watermarks\n\n")
        for image in traced_ranked:
            # Write each element of the traced_ranked list to file
            # after it has been reformatted. 
            image_info = format_string(image)
            eval_file.write(image_info)
        eval_file.write("\nEvaluation Set - Untraced Watermarks\n\n")
        for image in untraced_ranked:
            # Write each element of the untraced_ranked list to file
            # after it has been reformatted. 
            image_info = format_string(image)
            eval_file.write(image_info)

    all_ranks = []
    # Iterate the traced watermarks ranks and extract the ranking position
    for rank in traced_ranked:
        for match in rank[2]:
            all_ranks.append(match[0])
    # Iterate the untraced watermarks ranks and extract the ranking position
    for rank in untraced_ranked:
        for match in rank[2]:
            all_ranks.append(match[0])

    return all_ranks


def format_string(image_info):
    """
    Args:
        image_info: A tuple of the form 
                   (image_name, number_of_top_ranked_matches, matches_ranks).
    
    Returns:
        Formatted string for the given image_info object. 
        
    Method that takes the image_info and makes it into a string that is formatted 
    to describe more explicitly what each variable and parameter of the object means. 
    This will be then written into the evaluation_ranking_details.txt file.
    """

    output_string = (
    f"img-name = {image_info[0]}, top-ranked-match-count = {image_info[1]}\n" 
    )

    for match in image_info[2]:
        output_string +=  (
            f"  - rank={match[0]}, name={match[1]}, type={match[2]}\n"
        )
    
    return output_string


def create_histogram(db_name, all_ranks):
    """
    Args:
        db_name: the name of the database being used for evaluation.
        all_ranks: the list of all rank positions that will be used
                   to compute the ranks distribution.

    Method that generates the histogram showing the distribution of 
    rankings for all the images evaluated.Individual bins contain the numbers
    of matches found in the first 10, 20, 30, 40 and 50 rank positions. 
    The last bin corresponds to the number of rank positions that are above 50.
    """

    # If a rank is above 50, change its value to 60 to have bins with equal widths
    all_ranks = [60 if rank > 50 else rank for rank in all_ranks]
    
    # Generate the rank distribution histogram
    plt.hist(all_ranks, bins=[0,10,20,30,40,50,60],edgecolor='black')    

    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title(f'Rank Distribution for {db_name}')

    # Add labels for the ranking positions
    plt.xticks(ticks=[0,10,20,30,40,50,60],labels=[1,10,20,30,40,50,'>50'])

    # Save the histogram as an image file
    file_name = f"evaluation/histogram_{db_name}.png"
    plt.savefig(file_name)


if __name__ == "__main__":
    args = get_args()
    file_path = args.file_path
    db_path = args.db_path
    # Surpress the logger messages from the other files being called during execution,
    # unless the severity level is higher than ERROR.
    logger=logging.getLogger()
    logger.setLevel(logging.ERROR)
    # Make a new logger that does not surpress the messages with low severity in 
    # this file.
    logger_eval = logging.getLogger("evaluation.py")
    logger_eval.setLevel(logging.INFO)
    
    traced_ranked = []
    untraced_ranked = []
    found_traced = 0
    found_untraced = 0
    total_traced = 0
    total_untraced = 0
    total_images = 0
    top_rank = 0

    logger_eval.info("Computing system evaluation")
    
    try:
        # Iterate through the file directories in order to determine the size of the dataset
        for file in os.listdir(file_path):
            if file.endswith("_traced"):
                file_traced_path = os.path.join(file_path, file)
                # Get the total number of traced watermark images to be evaluated
                total_traced = sum(1 for im in os.listdir(file_traced_path)\
                                    if im.endswith((".jpg", ".png")))
            if file.endswith("_untraced"):
                file_untraced_path = os.path.join(file_path, file)
                # Get the total number of untraced watermark images to be evaluated
                total_untraced = sum(1 for im in os.listdir(file_untraced_path)\
                                    if im.endswith((".jpg", ".png")))
        database = loadDB(db_path)
        # Calculate the total dataset size
        total_images = total_traced + total_untraced + len(database)
        # Matches ranked in the first 10% of the total number of images will be taken into account
        # for the evaluation score
        top_rank = int(total_images/10)
    
        for file in os.listdir(file_path):
            # Check if the folder contains the traced watermark images.
            if file.endswith("_traced"):
                # Load the progress bar for processing traced watermarks.
                with alive_bar(total_traced, dual_line=True, title='Processing Traced Watermarks', \
                               bar='squares', spinner='wait4') as bar1:
                    for im in os.listdir(file_traced_path):
                        if im.endswith((".jpg", ".png")):
                            # For each image in the folder, compute and format its corresponding
                            # ranked_list.
                            image = cv2.imread(os.path.join(file_traced_path, im), cv2.IMREAD_GRAYSCALE)
                            ranked_list = compute_ranked_list(image, is_traced=True)
                            ranked_list = format_ranked_list(im, ranked_list)

                            # Count the number of matches with rank lower than 10.
                            count = sum(position[0] <= top_rank for position in ranked_list)
                            # Append the image name, count and list of ranked matches to the 
                            # traced_ranked list.
                            traced_ranked.append((im, count, ranked_list))
                            # If at least one match has rank lower than 10, then the image is
                            # considered correctly recognized.
                            if count > 0:
                                found_traced += 1
                            # Update the progress bar.
                            bar1()
                
            # Check if the folder contains the untraced watermark images.
            if file.endswith("_untraced"):
                # Load the progress bar for processing untraced watermarks.
                with alive_bar(total_untraced, dual_line=True, title='Processing Untraced Watermarks', \
                               bar='squares', spinner='wait4') as bar2:
                    for im in os.listdir(file_untraced_path):
                        if im.endswith((".jpg", ".png")):
                            # For each image in the folder, compute and format its corresponding
                            # ranked_list.
                            image = cv2.imread(os.path.join(file_untraced_path, im), cv2.IMREAD_GRAYSCALE)
                            ranked_list = compute_ranked_list(image, is_traced=False)
                            ranked_list = format_ranked_list(im, ranked_list)

                            # Count the number of matches with rank lower than 15.
                            count = sum(position[0] <= top_rank for position in ranked_list)
                            # Append the image name, count and list of ranked matches to the 
                            # untraced_ranked list.
                            untraced_ranked.append((im, count, ranked_list))
                            # If at least one match has rank lower than 10, then the image is
                            # considered correctly recognized.
                            if count > 0:
                                found_untraced += 1
                            # Update the progress bar.
                            bar2()
        
        # Create/Overwrite the evaluation_ranking_details.txt file with the contents of
        # traced_ranked and untraced_ranked lists.
        logger_eval.info("Generating the ranking details files...")
        # Only keep the name of the database, without the .pkl
        db_path = db_path.split("/")[1].split(".")[0]
        all_ranks = generate_ranking_details(db_path, traced_ranked, untraced_ranked)
        # Create the histogram with the rank distribution
        create_histogram(db_path, all_ranks)
        logger_eval.info("The ranking details files have been successfully generated!")

    except Exception as e:
        logger_eval.exception("Exception occurred while code Execution: %s", e)
        sys.exit()

    # Print the results for the evaluation.
    print("\n- - - EVALUATION RESULTS - - - \n")
    found = found_traced + found_untraced
    total = total_traced + total_untraced
    print(f"Overall number of watermark images with top-ranked matches: {found} / {total}.")
    print(f"Traced watermark images with matches in the top {top_rank} results: {found_traced} / {total_traced}.")
    print(f"Untraced watermark images with matches in the top {top_rank} results: {found_untraced} / {total_untraced}.")
    # Prints out the result obtained by dividing the number of watermarks having at least one 
    # top-ranked matche by the total number of watermarks as percentage and rounded to 2 decimal 
    # places. Dividing by the maximum between the total number of watermarks and 1 ensures that 
    # division by 0 is avoided when no image is found in the evaluation set (or only 
    # traced/untraced images are passed).
    print(f"Overall accuracy: {round(found*100/max(total, 1), 2)} %.")
    print(f"Accuracy for traced watermark images: {round(found_traced*100/max(total_traced, 1), 2)} %.")
    print(f"Accuracy for untraced watermark images: {round(found_untraced*100/max(total_untraced, 1), 2)} %.")