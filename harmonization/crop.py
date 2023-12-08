"""
This file is used to crop an uncropped watermark image. An uncropped watermark is
a watermark that has rulers framing the image. When the file is run, an input
path must be given, and it crops all images within the directory and all
subdirectories.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot

import harmonization.contrast_enhancement as Contrast


def crop(filepath):
    """
    Crops an image around a given watermark. This works for a specific format of image, where
    a watermark is on a background, and bordered by rulers.

    Args: filepath is the path of the image to be cropped

    Returns: the cropped image
    """

    # Read image
    # To manually crop around rulers: [38:722, 44:981]
    raw_image = cv2.imread(filepath, 0)[38:722, 44:981]

    # Apply CLAHE, to locally contrast stretch
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    image = clahe.apply(raw_image)
    image = Contrast.contrast_stretch(image)

    # Remove shadows and contrast stretch the result
    image = Contrast.remove_shadows(image, np.ones((8, 8)), 25)
    image = Contrast.contrast_stretch(image)

    # Threshold, then dilate to make the edges connect
    _, thresh = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    thresh = cv2.dilate(thresh, kernel)

    # Calculate the percentage of white pixels - this is used as a measure of noise
    white_percent = (len(np.nonzero(thresh.flatten())[
                     0]) / len(thresh.flatten())) * 100

    # If there are not many white pixels, the paper with the watermark is found by
    # finding the contours of the image
    if white_percent <= 30:

        # Find all contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the biggest contour by the area, and get its bounding rectnangle
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # If the biggest contour has approximately the same width and height as the image
        # (meaning the contour is not of the watermark itself), then loop until
        # the biggest watermark that is not a border of the image is found.
        while (w >= image.shape[1] - 20 and h >= image.shape[0] - 20):

            # Remove the contour with the biggest area so that the next biggest contour can be found
            contours = list(contours)
            for i, _ in contours:
                isequal = np.array_equal(max_contour, contours[i])
                if not isinstance(isequal, bool):
                    isequal = all(isequal.flatten())
                if isequal:
                    del contours[i]
                    break
            contours = tuple(contours)
            # Find the next biggest contour by the area
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

        # Return the raw image cropped around the contour points
        return_image = raw_image[y:y+h, x:x+w]

    else:

        # If there is a lot of noise, then that typically means it is a non-traced watermark,
        # which are typically darker, and don't work well with contours. Instead a hard binary
        # threshold is used.
        _, thresh = cv2.threshold(raw_image, 240, 255, cv2.THRESH_BINARY_INV)
        # Use an opening operation to remove noise and isolate the watermark
        kernel = np.ones((15, 15), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Get the indices of the outermost white pixels, which should correspond to the corners of
        # the watermark's bounding box.
        indices = np.nonzero(thresh)
        min_y = min(indices[1])
        max_y = max(indices[1])
        min_x = min(indices[0])
        max_x = max(indices[0])

        # Return the raw image cropped around the contour points
        return_image = raw_image[min_x:max_x, min_y:max_y]

    # Shows the returned image
    matplotlib.pyplot.imshow(return_image, cmap=matplotlib.pyplot.cm.gray)
    matplotlib.pyplot.show()

    # Overwrites the file path with the cropped image
    # cv2.imwrite(filepath, return_image)


def scan_path(path):
    """
    Scans through the given path, checks if it's valid and runs cropping on all files within
    the directory and all sub-directories

    Args: the path to the directory/image

    Returns: None
    """

    if os.path.isdir(path):
        for filename in os.listdir(path):
            scan_path(os.path.join(path, filename))
    elif os.path.isfile(path):
        if not ".DS_Store" in path:
            crop(path)
    else:
        print("Invalid path")


if __name__ == "__main__":
    file_path = ""
    args = sys.argv
    if len(args) == 2:
        file_path = args[1]
        scan_path(file_path)
    else:
        print("Image directory path not specified")
