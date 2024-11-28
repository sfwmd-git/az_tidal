#!/bin/bash
python=/usr/bin/python3
path=/home/ntaminge/SFWMD_work/predictions
today=$(date +%Y-%m-%d)

$python $path/py_00_Get\ GFS\ DET\ GRIB\ data.py
$python $path/py_10_Get\ WAVE\ DET\ GRIB\ data.py

# Define the list of sites
sites=("virginia_key" "lake_worth" "key_west" "vaca_key" "port_everglades" "naples_bay" "fort_myers")

# Loop through each site and call the python script
for site in "${sites[@]}"
do
	echo "Running script for site: $site"
	$python $path/py_a0_Calc\ GFS\ DET.py --site "$site"
done

exit 0
