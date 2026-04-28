import os
from PIL import Image
import numpy as np

# Specify the folder containing the mask images
mask_folder = r"C:\Users\michele.wiseman\Desktop\Saliency_based_Grape_PM_Quantification-main\code\Pytorch-UNet\data\masks"

# Specify the folder to save the GIF images
output_folder = r"C:\Users\michele.wiseman\Desktop\Saliency_based_Grape_PM_Quantification-main\code\Pytorch-UNet\data\gif_masks"

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all files in the folder
for filename in os.listdir(mask_folder):
    if filename.endswith(".png"):
        # Load the mask image
        mask_path = os.path.join(mask_folder, filename)
        mask_image = Image.open(mask_path)

        # Normalize the pixel values to the range [0, 1]
        mask_array = (1.0 / 255.0) * np.array(mask_image)

        # Convert the array to binary format with values of either 0 or 1
        mask_array = (mask_array > 0.5).astype(np.uint8)

        # Save the binary mask image as a GIF file
        gif_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".gif")
        Image.fromarray(mask_array * 255).save(gif_path, format="GIF")

        # Verify that the pixel values now range from 0 to 1
        unique_mask_values = sorted(set(mask_array.flatten()))
        print(f"Unique mask values for {filename}: {unique_mask_values}")
