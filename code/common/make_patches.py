import os
from PIL import Image
import math

# Set the path to the directory containing the input images
#input_dir = "D:/Stacked/Deposition_Study/2-24-2023_6dpi/1/"
input_dir = "C:/Users/Intel User/Desktop/blackbird_scripts/data/make_patches"

# Set the patch size
patch_size = 224

# Loop through each PNG file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # Load the input image
        img = Image.open(os.path.join(input_dir, filename))

        # Convert the image to RGB mode
        img = img.convert("RGB")

        # Calculate the number of patches in each dimension
        num_patches_wide = math.ceil(img.width / patch_size)
        num_patches_high = math.ceil(img.height / patch_size)

        # Loop through each patch and save it as a separate image
        for w in range(num_patches_wide):
            for h in range(num_patches_high):
                # Calculate the coordinates of the top-left corner of this patch
                left = w * patch_size
                top = h * patch_size
                right = left + patch_size
                bottom = top + patch_size

                # Crop the patch from the input image
                patch = img.crop((left, top, right, bottom))

                # Construct the output filename
                output_filename = os.path.splitext(filename)[0] + f"_patch_{w}_{h}.png"

                output_path = os.path.join(input_dir, output_filename)

                # Save the patch as a separate image in the same directory as the input image
                patch.save(output_path)