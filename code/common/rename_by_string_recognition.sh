#!/bin/bash

# Set the path to the directory containing the PNG files
path="C:\Users\Intel User\Desktop\Downy\clear"

# Loop through all PNG files in the directory
for file in "$path"/*.png
do
  # Replace "dilated_mask" with "image" in the filename
  new_filename=$(echo "$file" | sed 's/infected/_clear/g')

  # Rename the file
  mv "$file" "$new_filename"
done