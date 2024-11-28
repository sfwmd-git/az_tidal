#!/bin/bash
python=/usr/bin/python3
path=/home/ntaminge/SFWMD_work/predictions
today=$(date +%Y-%m-%d)

# Define the list of sites
sites=("virginia_key" "lake_worth" "key_west" "vaca_key" "port_everglades" "naples_bay" "fort_myers")

# Loop through each site and call the python script
for site in "${sites[@]}"
do
	echo "Running script for site: $site"
	$python $path/py_c0_Calc\ ECMWF\ DET.py --site "$site"
done

exit 0
