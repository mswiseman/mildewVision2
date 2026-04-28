#!/bin/bash

# Define the source and destination directory
SOURCE_DIR="E:/Stacked/Results/ResNet_upth0.8_downth0.3_Feb03_16-24-20_2024"
DESTINATION_DIR1="E:/Results/ResNet_upth0.8_downth0.3_Feb03_16-24-20_2024/5dpi"
DESTINATION_DIR2="E:/Results/ResNet_upth0.8_downth0.3_Feb03_16-24-20_2024/5dpi"

# Create the destination directories if they don't exist
mkdir -p "$DESTINATION_DIR1"
mkdir -p "$DESTINATION_DIR2"

# Find files ending with '_patch_based_class1.png' in subdirectories ending with '10dpi' or '9dpi', then copy them to DESTINATION_DIR1
find "$SOURCE_DIR" -type f -regex ".*\(5dpi\|4dpi\)/.*_patch_based_class1\.png$" -exec cp {} "$DESTINATION_DIR1" \;

echo "Files copied to $DESTINATION_DIR1 successfully."

# Navigate to DESTINATION_DIR1 for renaming .png files
cd "$DESTINATION_DIR1"

# Loop through all .png files in DESTINATION_DIR1
for file in *.png; do
    # Check if the file name starts with '10_' or '9_' and remove the prefix
    if [[ $file == 5_* ]]; then
        newname="${file#5_}"
        mv "$file" "$newname"
    elif [[ $file == 4_* ]]; then
        newname="${file#4_}"
        mv "$file" "$newname"
    fi
done

echo "File names in $DESTINATION_DIR1 updated."

# Navigate to DESTINATION_DIR2 for renaming .png files based on three-digit prefix
cd "$DESTINATION_DIR2"

# Loop through all .png files starting with three digits followed by a dash
for file in [0-9][0-9][0-9]-*.png; do
    # Extract the three-digit prefix
    prefix="${file:0:3}"
    # Remove the prefix and the dash from the original filename
    rest="${file:4}"
    # Construct the new filename with the prefix moved to the end, just before '.png'
    newname="${rest%.*}-${prefix}.png"
    # Rename the file
    mv "$file" "$newname"
done

echo "File names in $DESTINATION_DIR2 updated."
