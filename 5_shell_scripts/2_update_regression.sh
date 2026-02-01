#!/bin/bash

# ------------------------------------------------------------------------------
# 1. Set up paths
# ------------------------------------------------------------------------------
# Adjust BASE_DIR so that it matches one of the recognized hostnames in your Python code
# i.e., /home/ntaminge/tidal_work or /Users/nate/Documents/tidal_work, etc.

BASE_DIR="/Users/Nate/Documents/tidal_work"

# Path to the CSV file containing the site_name column
CSV_FILE="$BASE_DIR/0_station_info/station_data.csv"

# Path to your Python script (the one that prints the site based on --site)
# You can place this in the same directory or wherever you like.
DOWNLOAD_TIDE_SCRIPT="$BASE_DIR/1_data/py_1_Get Tide Data.py"
DOWNLOAD_ERA5_SCRIPT="$BASE_DIR/1_data/py_2_Get ERA5 Data.py"
CLEAN_TIDE_SCRIPT="$BASE_DIR/1_data/py_3_Clean Tide Data.py"
CLEAN_ERA5_SCRIPT="$BASE_DIR/1_data/py_4_Clean ERA5 Data.py"
CLEAN_REGR_SCRIPT="$BASE_DIR/2_regressions/py_1_Clean Regression Data.py"
RUN_REGR_SCRIPT="$BASE_DIR/test_pvals_2_regressions/test_pvals_py_2_Run Regression.py"

# Move into shell_scripts directory (optional, but often convenient)
cd "$BASE_DIR/5_shell_scripts"

# ------------------------------------------------------------------------------
# 2. Loop through the 'site_name' column in station_data.csv
# ------------------------------------------------------------------------------
# - "tail -n +2" skips the header line of the CSV
# - "cut -d, -f1" extracts the first column. If site_name is in a different column,
#   change the -f argument to the appropriate column number.

sites=$(tail -n +2 "$CSV_FILE" | cut -d, -f1)

# ------------------------------------------------------------------------------
# 3. For each site, run the Python script and print the name
# ------------------------------------------------------------------------------
for site in $sites; do
    echo "Updating Regression for: $site"
    # python3 "$DOWNLOAD_TIDE_SCRIPT" --site "$site"
    # python3 "$DOWNLOAD_ERA5_SCRIPT" --site "$site"
    # python3 "$CLEAN_TIDE_SCRIPT" --site "$site"
    # python3 "$CLEAN_ERA5_SCRIPT" --site "$site"
    # python3 "$CLEAN_REGR_SCRIPT" --site "$site"
    python3 "$RUN_REGR_SCRIPT" --site "$site"
done

