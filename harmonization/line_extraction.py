"""
Perform rudimentary line removal through spectrum analysis
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

from harmonization.contrast_enhancement import contrast_stretch
from harmonization.sharpen import unsharp_masking_gaussian


def plot_log_magnitude_DFT(fourier):
    """
    Plots the log DFT of the fourier of the image and saves the result as file

    Args:
        fourier: The fourier transform of the image

    Returns: The log DFT of the fourier transform
    """
    # the log is used when plotting the DFT magnitude only for visual convince, to make the values apear more distinct
    spectrum = np.log(np.abs(fourier) + 0.00001)
    plt.imsave("./spectrum.png", spectrum, cmap="gray")
    # plt.imshow(spectrum, cmap = "gray")
    # plt.show()
    return spectrum


def fft(image):
    """
    Performs DFT on the image
    fft2 computes the 2D DFT
    fftshift shifts the 0 frequency to the centre of the spectrum

    Args:
        image: The image to be transformed

    Returns: The transformed image
    """
    return np.fft.fftshift(np.fft.fft2(image))


def identify_low_and_high_centre(threshold, centers, radius=25):
    """
    Identifies the lowest and highest vertical pair of centers in the spectrum
    (corresponding to horizontal lines)

    Args:
        threshold: The thresholded image
        centers: The centers of the connected components
        radius: Optional, the radius of the circles to be drawn around the centers. The default is 25.

    Returns: The set of points to attenuate
    """
    midX = int((threshold.shape[1] + 1)/2)
    # midY = int((threshold.shape[0] + 1)/2)
    miny = threshold.shape[1]
    maxy = 0
    # These correspond to the 2 centers on the vertical line passing through the center
    c_low = None
    c_high = None
    # Allowable pixel margin
    margin = 60
    lowestY = int((threshold.shape[1])/8)
    highestY = int((threshold.shape[1])*7/8)

    # Iterates the connected components centers to find the pair of 2 extremes corresponding to the
    # horizontal lines in the image
    for center in centers:
        x = int(center[0])
        y = int(center[1])
        if lowestY <= y < miny and midX - margin < x < midX + margin:
            c_low = (x, y)
            miny = y
        elif highestY >= y > maxy and midX - margin < x < midX + margin:
            c_high = (x, y)
            maxy = y
    circles = None
    # Checks if the 2 centers are actually close horizontally and far from each other vertically
    if c_low is not None and c_high is not None and maxy - miny > 50 and abs(c_low[0] - c_high[0]) < 20:
        cv2.circle(threshold, c_low, radius, (255), thickness=-1)
        cv2.circle(threshold, c_high, radius, (255), thickness=-1)
        pts = list(points_in_circle_np(radius))
        lower_circle = list(
            map(lambda pt: (pt[0]+c_low[0], pt[1] + c_low[1]), pts))
        upper_circle = list(
            map(lambda pt: (pt[0]+c_high[0], pt[1] + c_high[1]), pts))
        # Adds the 2 circles to be attenuated in the DFT
        circles = lower_circle + upper_circle
        # Adds the points inbetween the 2 circles to remove also denser lines
        for j in range(c_low[1] + radius, c_high[1] - radius):
            central_circle = list(map(lambda pt: (pt[0]+midX, pt[1] + j), pts))
            circles = circles + central_circle

    # Returns the set of points to attenuate
    return circles


def ifft(fourier):
    """
    Performs inverse DFT and recovers the image

    Args:
        fourier: The fourier transform of the image

    Returns: The recovered image
    """
    return np.fft.ifft2(np.fft.ifftshift(fourier)).real


def points_in_circle_np(radius):
    """
    Returns the coordinates of a circle centered at (0, 0) with a provided radius

    Args:
        radius: The radius of the circle

    Returns: The coordinates of the circle
    """
    a = np.arange(radius + 1)
    for x, y in zip(*np.where(a[:, np.newaxis]**2 + a**2 <= radius**2)):
        yield from set(((x, y), (x, -y), (-x, y), (-x, -y),))


def remove_lines(image):
    """
    Removes the lines from an image

    Args:
        image: The image to be processed

    Returns: The image with the lines removed
    """
    DFT = fft(image)
    plot_log_magnitude_DFT(DFT)

    spectr = cv2.imread("./spectrum.png", cv2.IMREAD_GRAYSCALE)

    _, thresh = cv2.threshold(spectr, 170, 255, cv2.THRESH_BINARY)
    #  find connected components
    components = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    #  draw circles around center of components but only for the lowest and highest circle
    circle_areas = identify_low_and_high_centre(thresh, components[3])
    if circle_areas is not None:
        for pt in circle_areas:
            if pt[1] < DFT.shape[0] and pt[0] < DFT.shape[1] and pt[0] >= 0 and pt[1] >= 0:
                DFT[pt[1]][pt[0]] = 0
    plot_log_magnitude_DFT(DFT)

    # plt.imshow(thresh, cmap="gray")
    # plt.title("Decoded Blobs")
    # plt.show()

    image = ifft(DFT)
    return image


def remove_vertical(image):
    """
    Removes the vertical lines from an image

    Args:
        image: The image to be processed

    Returns: The image with the vertical lines removed
    """
    DFT = fft(image)
    plot_log_magnitude_DFT(DFT)
    midX = int((DFT.shape[1] + 1)/2)
    midY = int((DFT.shape[0] + 1)/2)
    xMargin = 150
    yMargin = 10
    for x in range(midX-xMargin, midX + xMargin):
        for y in range(midY-yMargin, midY + yMargin):
            DFT[y][x] = 0
    plot_log_magnitude_DFT(DFT)
    image = ifft(DFT)
    return image


def process_lines(image, iterations=10):
    """
    Performs line removal several times

    Args:
        image: The image to be processed
        iterations: optional, the number of times to perform line removal. Default is 10.
    """
    processed = remove_lines(image)
    for _ in range(iterations):
        processed = remove_lines(processed)
    processed = contrast_stretch(processed)
    return processed


def process(image):
    """
    Performs the full line removal pipeline and returns the image

    Args:
        image: The image to be processed

    Returns: The processed image
    """
    sharpened = unsharp_masking_gaussian(image)
    return process_lines(sharpened)
