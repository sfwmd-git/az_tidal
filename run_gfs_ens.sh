#!/bin/bash
python=/usr/bin/python3
path=/home/ntaminge/SFWMD_work/predictions
today=$(date +%Y-%m-%d)

$python $path/py_01_Get\ GFS\ ENS\ GRIB\ data.py
$python $path/py_11_Get\ WAVE\ ENS\ GRIB\ data.py

# Define the list of sites
sites=("virginia_key" "lake_worth" "key_west" "vaca_key" "port_everglades" "naples_bay" "fort_myers")

# Loop through each site and call the python script
for site in "${sites[@]}"
do
	echo "Running script for site: $site"
	$python $path/py_b0_Calc\ GFS\ ENS.py --site "$site"
done

exit 0
