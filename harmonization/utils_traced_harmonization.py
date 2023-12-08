"""
Utility file used by the harmonization.py file for the denoising of
traced watermark images.
"""
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.restoration import estimate_sigma

import harmonization.wavelet_denoising as Wavelet
import harmonization.contrast_enhancement as Contrast
import harmonization.harmonization as Harmonization

def _filter_regions_on_size(image, regions, min_bbox_percent,
                                height_proportion, width_proportion):
    """
    Args:
    regions: the labeled regions of the image
    min_bbox_percent: The percentage of the image size that the size of the bounding
                    box of the region needs to be, at least, in order to be considered
                    valid. Any regions whose bounding boxes are less than (percent * image size)
                    are filtered out.
    height_proportion: the height of a region must be smaller than this proportion
                    of its width (i.e. height <= width * height_proportion). This
                    ensures that regions that are very high and not very wide are removed.
    width_proportion: the width of a region must be smaller than this proportion
                    of its height (i.e. width <= height * width_proportion). This
                    ensures that regions that are very wide and not very high are removed.

    Returns: The array of filtered regions

    Filters the regions of a labeled image based on their height, width, and area.
    """
    regions = np.array(regions)

    # Filters out regions that are too small
    bbox_areas = np.array([region.area_bbox for region in regions])
    min_area = image.shape[0] * image.shape[1] * min_bbox_percent
    large_enough_regions = regions[bbox_areas >= min_area]

    if len(large_enough_regions) == 0:
        return regions

    # use bbox = (min_y, min_x, max_y, max_x) to get heights and widths of areas
    bboxes = np.array([region.bbox for region in large_enough_regions])
    heights = bboxes[:, 2] - bboxes[:, 0]
    widths = bboxes[:, 3] - bboxes[:, 1]

    # Finds indices of regions that satisfy the height and width conditions
    enough_height = heights <= widths * height_proportion
    enough_width = widths <= heights * width_proportion

    # Returns the regions that are not too thin or too short
    return large_enough_regions[enough_height & enough_width]


# Note: The default 'min_bbox_percent' parameter is very broad to account for
# noise. To make it more narrow, values closer to 0.01 can be used. This will
# ideally isolate the location of the watermark more.
def cluster_pixels(image, min_bbox_percent=0.001, height_proportion=5, \
                   width_proportion=5, filter_outliers=True, is_restrictive=True):
    """
    Args:
    min_bbox_percent: The percentage of the image size that the size of the bounding
                    box of the region needs to be, at least, in order to be considered
                    valid. Any regions whos bounding boxes are less than (percent * image size)
                    are filtered out.
    height_proportion: the height of a region must be smaller than this proportion
                    of its width (i.e. height <= width * height_proportion). This
                    ensures that regions that are very high and not very wide are removed.
    width_proportion: the width of a region must be smaller than this proportion
                    of its height (i.e. width <= height * width_proportion). This
                    ensures that regions that are very wide and not very high are removed.
    filter_outliers: a boolean that, when true, filters out all regions whos
                    centroids are outliers from the rest of the centroids.
    is_restrictive: a boolean determining if conditions in the filter_area method will be
                    restrictive or not.

    Returns: The coordinates of the bounding box that encloses the filtered clusters/regions.

    Takes image and clusters the pixels in this image. The clusters
    are filtered such that no cluster that is too small appears, and clusters
    that have centroids that are outliers also do not appear.
    """

    # Uses skimage to group regions of an image, extract the regions
    label_image = label(image)
    regions = regionprops(label_image)
    # Pad 1 pixel on all sides to avoid errors with the contours
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    filtered_regions = _filter_regions_on_size(image, regions, min_bbox_percent, \
        height_proportion, width_proportion)

    # Only filter out the outliers if there are more than 2 regions. If there are less
    # than the outlier calculations could be erroneous.
    if len(filtered_regions) >= 2 and filter_outliers:

        # Get the centroids of all regions
        centroids = list(map(lambda x: x.centroid, filtered_regions))

        # Calculate if each centroid is an outlier using the interquartile range
        q3, q1 = np.percentile(centroids, [75, 25])
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        # Outliers is a boolean array, where the value is True if the corresponding centroid
        # in the centroid array is an outlier, and false otherwise. A centroid is an outlier
        # if either the x and y coordinate are outside of the minimum or maximum of the
        # interquartile range.
        outliers = list(map(lambda x: (x[0] > upper or x[0] < lower) and
                            (x[1] > upper or x[1] < lower), centroids))

        # Filters the filtered_regions further such that none of the outliers appear in
        # the filtered_regions array
        filtered_regions = list(
            filter(lambda coords: not coords[1], list(zip(filtered_regions, outliers))))
        # This line just unzips the zipped array
        filtered_regions = [region[0] for region in filtered_regions]

    # Finds the coordinates of the bounding box that encloses all filtered clusters/regions
    min_x = image.shape[0]
    min_y = image.shape[1]
    max_x = 0
    max_y = 0
    boundingboxes = np.array([region.bbox for region in filtered_regions])
    if len(boundingboxes) > 0:
        min_x = np.min(boundingboxes[:, 0])
        min_y = np.min(boundingboxes[:, 1])
        max_x = np.max(boundingboxes[:, 2])
        max_y = np.max(boundingboxes[:, 3])

    # Obtain the contours from the filtered regions.
    contours = make_contours(image, filtered_regions)
    # cop = np.copy(image)
    # mask = np.zeros(image.shape, dtype=np.uint8)
    # for region in contours:
    #     cv2.drawContours(mask, [region], 0, 255, -1)
    # # Apply the mask to the original image
    # cop = np.where(mask > 0, cop, 0)
    # cv2.imshow("cop",cop)
    # cv2.waitKey()
    # Obtain the regions that are going to be kept in the image
    if len(contours) > 0:
        kept_regions = compute_kept_regions(image, contours, is_restrictive=is_restrictive)
    else:
        kept_regions = []

    # Create a mask for the regions that are kept
    mask = np.zeros(image.shape, dtype=np.uint8)
    for region in kept_regions:
        cv2.drawContours(mask, [region], 0, 255, -1)
    # Apply the mask to the original image
    image = np.where(mask > 0, image, 0)

    # Finds the coordinates of the bounding box that encloses all filtered clusters/regions
    min_x = image.shape[0]
    min_y = image.shape[1]
    max_x = 0
    max_y = 0
    x,y,w,h = cv2.boundingRect(image)
    min_x = min(x, min_x)
    min_y = min(y, min_y)
    max_x = max(x+w, max_x)
    max_y = max(y+h, max_y)
    # Checks if any clusters were found. If not, then returns the coordinates
    # of the image, else it returns the coordinates of the clusters'
    # bounding box.
    if min_x == 0 and min_y == 0 and \
        max_x == 0 and max_y == 0:
        return image, (0, 0, image.shape[0], image.shape[1])
    return image, (min_x, min_y, max_x, max_y)


def harmonize_traced(image, raw_image, wavelet_option):
    """
    Args:
        image: the image to be processed
        raw_image: the original image given as input by the user
        wavelet_option: the wavelet denoising strategy

    Method performing the entire harmonization pipeline without ameliorate_contrast_on_margins.
    """
    image = Wavelet.wavelet_traced(image, option=wavelet_option)
    image = Contrast.contrast_stretch(image)
    image = np.clip(image, 0, 255)
    image = Contrast.remove_shadows(image, np.ones((8,8)), 33)
    h = Harmonization.Harmonization(image=image)

    # Estimate the noise levels in the image, choose the denoising based on
    # noise level
    # print("estimate", estimate_sigma(raw_image, average_sigmas=True))
    if estimate_sigma(raw_image, average_sigmas=True) < 1:
        # Light noise denoising
        image = h.threshold_traced_light_noise()
    else:
        # Heavy noise denoising
        image = h.denoise_traced_heavy_noise()
        image = h.threshold_traced_heavy_noise()

    # Cluster the pixels in the binarized image
    cluster_img, (min_x, min_y, max_x, max_y) = cluster_pixels(image, is_restrictive=False)
    return cluster_img, (min_x, min_y, max_x, max_y)


def compute_kept_regions(image, contours, few_contours = 10, many_contours = 25, \
                         is_restrictive=True):
    """
    Args:
    contours: the contours of the image
    few_contours: an int determining how many contours are considered few.
    many_contours: an int determining how many contours are considered many.

    Returns: The array of kept regions from the input image.

    Takes image, the contours and the number of contours considered few and many. It
    iterates through all the contours, keeps the contours that are not too small and too
    close to the borders of the image. This is done according to the number of total contours
    present in the image.
    """
    if len(contours) > 0:
        # Extract the largest contour of all, this will contain a large part of the watermark.
        largest_contour = max(contours, key = cv2.contourArea)
        # Always keep the contours that are significantly overlapped by the largest one.
        contours_large_overlap = filter_overlap(image, contours, largest_contour)
    else:
        largest_contour = []
        contours_large_overlap = []
    # curious = np.zeros(image.shape)
    # for reg in contours:
    #     cv2.drawContours(curious,[reg],0,(255,255,255),1)
    # cv2.imshow("see", curious)
    # cv2.waitKey()
    kept_regions = []
    if len(contours) >= 1:
        # Check if the image is much wider than it is tall, this generally means that the
        # image consists of text, which is handled separately.
        if image.shape[0] * 2 <= image.shape[1]:
            # Eliminate the contours next to the borders, keep mostly the ones towards the middle of
            # the image.
            kept_regions = filter_close_to_borders(image, contours, image.shape[0]/10, image.shape[1]/10)
            # Eliminate contours that are too small in area.
            filtered_area_contours = filter_by_area(
                image, contours, is_restrictive)
            kept_regions.extend(filtered_area_contours)

        elif len(contours) <= few_contours:
            # If there aren't a lot of contours in the image, just eliminate the ones that are
            # too small.
            kept_regions = filter_by_area(image, contours)
            # If there are more than a set number of contours in the image, also include some
            # contours that have smaller area, but that are not too close to the imae borders.
        elif len(contours) <= many_contours:
            kept_regions = filter_by_area(image, contours)
            filtered_border_contours = filter_close_to_borders(image, contours, image.shape[0]/10,
                                                               image.shape[1]/10)
            kept_regions.extend(filtered_border_contours)
        # If there are too many contours, keep all of them to avoid losing details from the denoised watermark.
        else:
            kept_regions.extend(contours)

    # Add all the regions/contours that were kept in one array.
    kept_regions.extend(contours_large_overlap)

    return kept_regions


def filter_close_to_borders(image, contours, distance_y, distance_x):
    """
    Args:
    contours: the contours of the image
    distance_y: the limit on the y-axis
    distance_x: the limit on the x-axis

    Returns: The array of contours filtered according to the given distances.

    Takes image, the contours and the threshold distance to the two axes. It
    iterates through all the contours, except for the largest one, and checks if
    their sides are not too close to the borders, in which case they get added to the
    array of filtered contours and returned.
    """
    if len(contours) <= 1:
        return []

    # Remove the largest contour, it doesn't need to be considered
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]

    # (x, y, w, h)
    boundingRects = np.array([cv2.boundingRect(contour) for contour in contours])

    # Check if the image is much taller than it is wide, then increase the threshold
    if image.shape[0] * 2 <= image.shape[1]:
        distance_x *= 1.25

    # Check if the contours are too close to the borders of the image.
    cond_1 = boundingRects[:, 0] < distance_x
    cond_2 = np.abs(boundingRects[:, 0] + boundingRects[:, 2] - image.shape[1]) < distance_x
    cond_3 = boundingRects[:, 1] < distance_y
    cond_4 = np.abs(boundingRects[:, 1] + boundingRects[:, 3] - image.shape[0]) < distance_y
    not_too_close = ~(cond_1 | cond_2 | cond_3 | cond_4)

    result = []
    for i in range(len(not_too_close)):
        if not_too_close[i]:
            result.append(contours[i])

    # OR all of them and then do the negation with ~, because none of these should be true
    return result


def filter_overlap(image, contours, c, overlap_val=80):
    """
    Args:
    contours: the contours of the image
    c: the largest contour in the image
    overlap_value: percentage of overlap needed to accept a contour

    Returns: The array of contours filtered according to the overlap level with the
    largest contour.

    Takes image, the contours and the largest contour. It determines the bounding
    boxes for the contours and the largest contour and then determines the level of
    overlap between them. If over a set value of a given contour is inside the largest one,
    then it is probably part of the watermark so it is added to the list of filtered
    contours.
    """
    contours = list(filter(lambda x: x is not c, contours))

    if len(contours) == 0:
        return []

    # Extract coordinates from bounding boxes
    x1_box1, y1_box1, x2_box1, y2_box1 = cv2.boundingRect(c)
    x1_boxes, y1_boxes, x2_boxes, y2_boxes = zip(*[cv2.boundingRect(box) for box in contours])
    x1_boxes, y1_boxes, x2_boxes, y2_boxes = np.array(x1_boxes), np.array(y1_boxes), \
                                                np.array(x2_boxes), np.array(y2_boxes)

    area_box1 = x2_box1*y2_box1
    area_boxes = x2_boxes*y2_boxes

    x_intersections = np.maximum(x1_box1, x1_boxes)
    y_intersections = np.maximum(y1_box1, y1_boxes)
    x_intersections_end = np.minimum(x2_box1+x1_box1, x2_boxes+x1_boxes)
    y_intersections_end = np.minimum(y2_box1+y1_box1, y2_boxes+y1_boxes)

    intersection_widths = np.maximum(0, x_intersections_end - x_intersections)
    intersection_heights = np.maximum(0, y_intersections_end - y_intersections)

    intersection_areas = intersection_widths * intersection_heights

    # Only does the division when area_boxes != 0. Otherwise returns the intersection area
    percentage_overlaps = np.where(np.minimum(area_box1, area_boxes) != 0, \
                                   (intersection_areas / np.minimum(area_box1, area_boxes)) * 100, \
                                   intersection_areas)
    enough_overlap = []

    # Check if the percentage of overlap is over a set value, then add the contour to the list.
    # This has to be a for loop because the contours are not necessarily the same shape.
    for i in range(len(percentage_overlaps)):
        if percentage_overlaps[i] >= overlap_val:
            enough_overlap.append(contours[i])

    return enough_overlap


def filter_by_area(image, contours, is_restrictive=True):
    """
    Args:
    contours: the array of contours in the image.
    is_restrictive: an integer determining if the computation of the method is restrictive
                    with the conditions that need to be satisfied for contours to be kept,
                    or not.

    Returns: The array of filtered contours.

    Takes image and the contours contained in the image. The contours
    are filtered such that no contour that is too small appears, and contours
    that are too far from the middle of the image are eliminated.
    """
    if len(contours) == 0:
        return []
    image_width = image.shape[1] #center_x
    image_height = image.shape[0] #center_y
    # Sort the contours decreasingly by their area, so that the largest area contour is the first.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    kept_regions = [sorted_contours[0]]
    # Check if the image is not much wider than it is tall, this generally means that the
    # image consists of an object, and not necessarily only text which is handled separately.
    if image_height * 2 > image_width:
        if len(sorted_contours) > 1 and cv2.contourArea(sorted_contours[1]) != 0:
            if cv2.contourArea(sorted_contours[0])/cv2.contourArea(sorted_contours[1]) < 13:
                kept_regions.append(sorted_contours[1])
            i = 2
            # Keep all the contours that are large enough compared to the largest one.
            while i < len(sorted_contours) and cv2.contourArea(sorted_contours[i]) > 0:
                if cv2.contourArea(sorted_contours[0])/cv2.contourArea(sorted_contours[i]) < 15:
                    kept_regions.append(sorted_contours[i])
                i += 1
        return kept_regions
    # Handle the case where the width is much larger than the height of the image.
    areas = np.array([cv2.contourArea(contour) for contour in contours])
    moments = [cv2.moments(contour) for contour in contours]
    area_avg = np.mean(areas)

    centroid_ys = [int(moment["m01"] / moment["m00"]) if moment["m00"] != 0 else int(moment["m01"]) for moment in moments]
    distances = np.abs(np.array(centroid_ys) - image_height / 2)
    dist_avg = np.mean(distances)

    # Keep contours that are not far from the middle of the image on the y-axis.
    # This is because text can span the entire x-axis, if it is a word, so we should
    # only eliminate the elements that are very high or very low compared with the others.

    bboxes = np.array([cv2.boundingRect(contour) for contour in contours])
    y_values = bboxes[:, 1]
    heights = bboxes[:, 3]

    area_condition = areas*1.75 >= area_avg
    dist_condition = distances <= dist_avg*1.25
    y_condition = (y_values <= image_height/2) & (y_values+heights >= image_height/2)
    mask = area_condition & (dist_condition | y_condition) if is_restrictive \
    else area_condition | dist_condition | y_condition

    result = []

    for i in range(len(mask)):
        if mask[i]:
            result.append(contours[i])

    return result

def make_contours(image, regions):
    """
    Args:
    regions: the identified regions from the image

    Returns: The array of contours extracted from the regions.

    Takes image and the regions contained in the image. It creates the
    contours of each region separately, then it extracts the outer contour from
    those. This results in regions that are overlapped being contained in only
    one contour, which will make it easier to distinguish the watermark from the
    noise.
    """
    image_zeros = np.zeros(image.shape)
    for region in regions:
        # Extract the bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox
        # Draw a rectangle on the image
        # Make it 1 pixel wilder than the original regions, so that regions that are
        # very close but not overlapped can be extracted together.
        cv2.rectangle(image_zeros, (max(0, min_col-1), max(0, min_row-1)), \
                        (min(image.shape[1]-1,max_col+1), min(image.shape[0]-1,max_row+1)), \
                        (255), 1)

    # Extract the outer contours from the obtained image.
    image_zeros = np.uint8(image_zeros)
    contours,_= cv2.findContours(image_zeros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Return the contours.
    return contours
