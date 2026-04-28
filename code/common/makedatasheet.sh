#!/bin/bash

# Set the target directory
directory="/mnt/e/transformation_studies/MWT12/August_21st_2023/tray3/"

# Declare an associative array
declare -A unique_numbers

# Loop through all files in the directory
for filename in "$directory"/*
do
    # Extract the desired number using regex
    number=$(echo "$(basename "$filename")" | grep -o -E '[0-9]{6}')

    # If a number is found and it's not in the associative array, print it and add to the array
    if [ ! -z "$number" ] && [ -z "${unique_numbers[$number]}" ]; then
        echo "$number"
        unique_numbers["$number"]=1
    fi
done
