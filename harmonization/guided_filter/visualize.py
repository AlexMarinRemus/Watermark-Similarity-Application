"""
Taken from by https://github.com/lisabug/guided-filter
This file defines methods to perform visualisation of images
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_single(img, title=""):
    """
    Plots a single grayscale image
    Args:
        img: image to plot
        title: Title for the plot
    """
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.waitforbuttonpress()


def plot_multiple(imgs, main_title="", titles=""):
    """
    Plots multiple images
    Args:
        imgs: images to plot
        main_title: all-encompassing title
        titles: subtitles for the images
    """
    num_img = len(imgs)
    rows = (num_img + 1) / 2
    plt.figure()
    plt.title(main_title)
    _, axarr = plt.subplots(rows, 2)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axarr[i/2, i%2].imshow(img.astype(np.uint8), cmap="gray")
        axarr[i/2, i%2].set_title(title)
    plt.waitforbuttonpress()