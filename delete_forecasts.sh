#!/bin/bash

# Define the directory
DIRECTORY="/home/ntaminge/SFWMD_work/predictions"

# Change to the directory
cd "$DIRECTORY"

# Check if the directory change was successful
if [ $? -eq 0 ]; then
    # Delete files starting with A1
    rm -f A1*

    # Delete files ending with .grib2
    rm -f *.grib2
			    
    echo "Files deleted successfully."
 else
    echo "Failed to change directory. Check the directory path and permissions."

fi
