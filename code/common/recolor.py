import os
import cv2
import numpy as np

# Directory containing the JPG images
img_dir = "/path/to/directory"

# Loop over all JPG images in the directory
for filename in os.listdir(img_dir):
    if filename.endswith(".jpg"):
        # Load the image and convert to grayscale
        image = cv2.imread(os.path.join(img_dir, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary mask
        ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw white contours on a black background
        mask = np.zeros_like(gray)
        for contour in contours:
            cv2.drawContours(mask, [contour], 0, 255, -1)

        # Smooth the edges of the white lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Convert red color to white
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask[mask_red > 0] = 255

        # Save the mask as a PNG file with the same name as the input image
        output_path = os.path.join(img_dir, os.path.splitext(filename)[0] + "_processed.png")
        cv2.imwrite(output_path, mask)
