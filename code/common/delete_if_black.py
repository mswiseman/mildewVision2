import os
from PIL import Image

# Set the path to the directory containing the input images
input_dir = "G:/make_patches/"
#input_dir = "C:/Users/michele.wiseman/Desktop/Saliency_based_Grape_PM_Quantification-main/code/Pytorch-UNet/data/masks/"

# Set the threshold for black pixels (as a fraction of the total pixels)
black_threshold = 0.05

# Loop through each PNG file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # Load the input image
        img = Image.open(os.path.join(input_dir, filename))
        #print("Converting image: " + filename)

        # Convert the image to grayscale mode
        img = img.convert("L")

        # Calculate the percentage of black pixels
        num_pixels = img.width * img.height
        num_black_pixels = sum(1 for pixel in img.getdata() if pixel == 0)
        black_fraction = num_black_pixels / num_pixels

        # Delete the file if it has more than the specified threshold of black pixels
        if black_fraction > black_threshold:
            os.remove(os.path.join(input_dir, filename))