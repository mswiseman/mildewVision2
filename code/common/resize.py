from PIL import Image
import os

# Set the input directory, output directory, and target size
input_dir = r'C:/Users/Intel User/Desktop/blackbird_scripts/data/segmentation/segmentation_annotations/test/images'
output_dir = r'C:/Users/Intel User/Desktop/blackbird_scripts/data/segmentation/segmentation_annotations/test/images2'
target_width = 224
target_height = 224

# Loop through all PNG files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # Open the image and resize it
        img = Image.open(os.path.join(input_dir, filename))
        resized_img = img.resize((target_width, target_height))
        
        # Save the resized image to the output directory
        output_path = os.path.join(output_dir, filename)
        resized_img.save(output_path)
