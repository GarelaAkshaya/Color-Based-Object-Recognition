
import cv2
import matplotlib.pyplot as plt
from histogram import extract_histogram
from recognition import recognize_color


def display_histograms():

    colors = {
        "Red": "images/red.jpg",
        "Green": "images/green.jpg",
        "Blue": "images/blue.jpg",
        "Yellow": "images/yellow.jpg"
    }

    plt.figure(figsize=(10, 6))

    for color, image_path in colors.items():

        histogram = extract_histogram(image_path)

        plt.plot(histogram.flatten(), label=color)

    plt.title("Normalized HSV Color Histograms")
    plt.xlabel("Histogram Bins")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/color_histograms.png")
    plt.show()


def main():

    print("COLOR-BASED OBJECT RECOGNITION")
    print("--------------------------------")

    test_image = input(
        "Enter test image path (example: images/red.jpg): "
    )

    try:
        color, distance = recognize_color(test_image)

        print("\nRecognition Result")
        print("------------------")
        print("Recognized Color:", color)
        print("Bhattacharyya Distance:", round(distance, 4))

        display_histograms()

    except (ValueError, cv2.error) as error:
        print("Error:", error)


if __name__ == "__main__":
    main()
