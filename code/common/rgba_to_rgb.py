from PIL import Image
import os

# Set the input and output directories
input_dir = r'D:\Stacked\Mapping_Population_2023\6-27-2023_5dpi\1\013-W2021011-489_R1'
output_dir = r'D:\Stacked\Mapping_Population_2023\6-27-2023_5dpi\1\013-W2021011-489_R1\rgb'

# Loop through all PNG files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # Open the image and convert it to RGB format
        img = Image.open(os.path.join(input_dir, filename))
        rgb_img = img.convert('RGB')
        
        # Save the converted image to the output directory
        output_path = os.path.join(output_dir, filename)
        rgb_img.save(output_path)

#from PIL import Image
#import os

# Set the input and output directories
##input_dir = r'C:\Users\michele.wiseman\Desktop\Saliency_based_Grape_PM_Quantification-main\code\Pytorch-UNet\data\masks'
#output_dir = r'C:\Users\michele.wiseman\Desktop\Saliency_based_Grape_PM_Quantification-main\code\Pytorch-UNet\data'

# Loop through all image files in the input directory
#for filename in os.listdir(input_dir):
#    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Open the image and convert it to grayscale
#        img = Image.open(os.path.join(input_dir, filename)).convert('L')
        
        # Save the converted image to the output directory
#        output_path = os.path.join(output_dir, filename)
#        img.save(output_path)