
import cv2
import numpy as np


def extract_histogram(image_path):
    """
    Read an image, convert it to HSV,
    and calculate its normalized colour histogram.
    """

    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate 2D histogram using Hue and Saturation
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [36, 32],
        [0, 180, 0, 256]
    )

    # Normalize the histogram
    cv2.normalize(
        hist,
        hist,
        alpha=1,
        norm_type=cv2.NORM_L1
    )

    return hist
