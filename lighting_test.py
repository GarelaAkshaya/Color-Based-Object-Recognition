
import cv2
import os
from recognition import recognize_color


def change_brightness(image_path, factor):
    """
    Increase or decrease image brightness.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Modify the Value channel
    h, s, v = cv2.split(hsv)

    v = cv2.convertScaleAbs(v, alpha=factor)

    # Merge HSV channels
    modified_hsv = cv2.merge([h, s, v])

    # Convert back to BGR
    result = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

    return result


def run_lighting_test():

    os.makedirs("outputs", exist_ok=True)

    reference_images = {
        "Red": "images/red.jpg",
        "Green": "images/green.jpg",
        "Blue": "images/blue.jpg",
        "Yellow": "images/yellow.jpg"
    }

    brightness_levels = [0.5, 0.75, 1.0, 1.25, 1.5]

    for color, image_path in reference_images.items():

        print(f"\nTesting {color} image")

        for factor in brightness_levels:

            # Change brightness
            modified_image = change_brightness(image_path, factor)

            # Save modified image
            output_path = f"outputs/{color.lower()}_{factor}.jpg"

            cv2.imwrite(output_path, modified_image)

            # Recognize modified image
            result, distance = recognize_color(output_path)

            print(
                f"Brightness: {factor} | "
                f"Recognized: {result} | "
                f"Distance: {distance:.4f}"
            )


if __name__ == "__main__":
    run_lighting_test()
