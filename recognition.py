
import cv2
from histogram import extract_histogram


# Reference images
REFERENCE_IMAGES = {
    "Red": "images/red.jpg",
    "Green": "images/green.jpg",
    "Blue": "images/blue.jpg",
    "Yellow": "images/yellow.jpg"
}


def recognize_color(test_image_path):
    """
    Compare the test image histogram with
    reference colour histograms.
    """

    # Extract histogram of test image
    test_hist = extract_histogram(test_image_path)

    best_match = None
    lowest_distance = float("inf")

    # Compare with all reference images
    for color, image_path in REFERENCE_IMAGES.items():

        reference_hist = extract_histogram(image_path)

        # Calculate Bhattacharyya distance
        distance = cv2.compareHist(
            test_hist,
            reference_hist,
            cv2.HISTCMP_BHATTACHARYYA
        )

        print(f"{color}: Distance = {distance:.4f}")

        if distance < lowest_distance:
            lowest_distance = distance
            best_match = color

    return best_match, lowest_distance
