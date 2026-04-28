import numpy as np
from PIL import Image

# Create a numpy array of your R, G, or B values
# This is a small patch of an infected PM patch

data = [[122, 113, 99, 89, 78, 62, 54, 58, 58, 62, 79, 107],
        [134, 121, 97, 84, 70, 55, 53, 53, 54, 60, 74, 98],
        [126, 116, 99, 82, 66, 49, 48, 52, 55, 60, 68, 88],
        [123, 115, 103, 78, 60, 44, 43, 51, 58, 69, 76, 75],
        [143, 128, 110, 78, 59, 49, 50, 59, 70, 89, 89, 78],
        [139, 130, 121, 110, 90, 71, 66, 64, 85, 120, 126, 130],
        [128, 123, 122, 124, 115, 89, 74, 67, 87, 137, 158, 164],
        [110, 109, 108, 111, 105, 86, 83, 97, 106, 121, 141, 158],
        [95, 97, 102, 108, 104, 89, 87, 112, 123, 115, 125, 145],
        [103, 98, 102, 110, 105, 98, 96, 105, 114, 121, 121, 113],
        [108, 102, 104, 109, 111, 111, 110, 107, 110, 117, 112, 92],
        [101, 108, 108, 112, 123, 133, 126, 116, 115, 106, 84, 59]]

# initial empty arrays for your color channels and then populate them
red_channel_array = np.zeros((12, 12, 3), dtype = np.uint8) # initializes an empty array 
blue_channel_array = np.zeros((12, 12, 3), dtype = np.uint8) # initializes an empty array 
green_channel_array = np.zeros((12, 12, 3), dtype = np.uint8) # initializes an empty array 
red_channel_array[:, :, 0] = data    # red is the first channel
green_channel_array[:, :, 1] = data  # green is the second channel
blue_channel_array[:, :, 2] = data   # blue is the third channel

# Convert to grayscale
gray_array = np.dot(red_channel_array[..., :3], [0.2989, 0.5870, 0.1140]) # 
gray_image = Image.fromarray(gray_array).convert('L') # convert to grayscale

# Create a PIL image from the array
image = Image.fromarray(red_channel_array)     # 
image2 = Image.fromarray(blue_channel_array)
image3 = Image.fromarray(green_channel_array)

# Save to a file
image.save('red_channel.jpg')
image2.save('blue_channel.jpg')
image3.save('green_channel.jpg')
gray_image.save('gray_image.jpg')


