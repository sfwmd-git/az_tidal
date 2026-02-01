import os
import platform
import numpy as np
import pandas as pd
import netCDF4
import cdsapi
import requests
from datetime import datetime
from io import StringIO  # This is the correct import for StringIO
import calendar
from netCDF4 import Dataset, num2date
import cdsapi
import argparse  # <-- Add this line

#------------------------------------------------------------------------------
#
#                 SET BASE DIRECTORY
#
#------------------------------------------------------------------------------

# List of allowed hostnames
allowed_hostnames = [
    'Nates-MacBook-Pro.local', 'UM-C02RQ1S8FVH7', 'nates-mbp.lan',
    'Nates-MBP.lan', 'Nates-MBP.localdomain'
]

# Get the current hostname
hostname = platform.node()

# Define base directory based on hostname
if hostname in allowed_hostnames:
    BASE_DIR = '/Users/nate/Documents/tidal_work/'
elif hostname == 'mango.rsmas.miami.edu':
    BASE_DIR = '/home/ntaminger/tidal_work/'
elif hostname == 'aztidal01p':
    BASE_DIR = '/home/ntaminge/tidal_work/'
else:
    raise ValueError(f"Unknown hostname: {hostname}")

os.chdir(BASE_DIR)
#------------------------------------------------------------------------------
#
#                 SET DIRECTORIES
#
#------------------------------------------------------------------------------
input_dir = os.path.join(BASE_DIR, '0_station_info')
output_dir = os.path.join(BASE_DIR, '1_data', 'raw_water_data')

#------------------------------------------------------------------------------
#
#                 LOAD STATION DATA FROM CSV
#
#------------------------------------------------------------------------------

# Load station-specific data from the site-specific CSV file
csv_file = os.path.join(input_dir, 'station_data.csv')
df = pd.read_csv(csv_file)

#------------------------------------------------------------------------------
#
#                 SET THE SITE
#
#------------------------------------------------------------------------------

site = 'virginia_key'  # Change this as needed for different sites

# # Argument parsing
# parser = argparse.ArgumentParser(description="Run the script for a specific site")
# parser.add_argument('--site', required=True, help='The site to run the script for (e.g., fort_myers)')
# args = parser.parse_args()
# 
# # Use the provided site argument
# site = args.site

site_data = df[df['site_name'] == site]
if site_data.empty:
    raise ValueError(f"Site '{site_name}' not found in the dataset.")

#------------------------------------------------------------------------------
#
#                 DEFINE VARIABLES
#
#------------------------------------------------------------------------------

site_name = site_data['site_name'].iloc[0]
station_ID = site_data['station_ID'].iloc[0]
station_abbr = site_data['station_abbr'].iloc[0]
station_name = site_data['station_name'].iloc[0]
wlon = site_data['wlon'].iloc[0]
elon = site_data['elon'].iloc[0]
slat = site_data['slat'].iloc[0]
nlat = site_data['nlat'].iloc[0]
startyear = 2020

# --- NEW LOGIC: determine last full month up to today's date ---
current_date = datetime.now()
year_end = current_date.year
month_end = current_date.month - 1  # last fully completed month

# If we're currently in January, 'month_end' will be 0, so go to December of the previous year
if month_end == 0:
    month_end = 12
    year_end -= 1

#-------------------------------------------------------------------------------
#
#              PULL ALL TIDE DATA FROM 1996-2021 FROM NOAA API
#
#-------------------------------------------------------------------------------

for year in range(startyear, year_end + 1):
    # If this is the last year in the loop, only go to 'month_end'
    # Otherwise, go through all 12 months
    start_month = 1
    end_month = 12
    if year == year_end:
        end_month = month_end

    print(f"\n=== YEAR: {year} ===")
    for month in range(start_month, end_month + 1):
        _, last_day = calendar.monthrange(year, month)
        startdate   = f"{year}{month:02d}01"
        enddate     = f"{year}{month:02d}{last_day:02d}"

        obs_filename   = f"{station_ID}_observations_{year}_{month:02d}.csv"
        preds_filename = f"{station_ID}_predictions_{year}_{month:02d}.csv"

        obs_filepath   = os.path.join(output_dir, obs_filename)
        preds_filepath = os.path.join(output_dir, preds_filename)

        # # If both already exist, skip downloading
        # if os.path.exists(obs_filepath) and os.path.exists(preds_filepath):
        #     # Remove or comment out this print if you don't want to see "skip" messages:
        #     print(f"  Month {month:02d}: Both files exist, skipping.")
        #     continue

        # Build the URLs (only if needed)
        obs_url = (
            f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
            f"begin_date={startdate}&end_date={enddate}"
            f"&station={station_ID}"
            f"&product=water_level&datum=MLLW&time_zone=gmt"
            f"&units=english&format=csv"
        )
        preds_url = (
            f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
            f"begin_date={startdate}&end_date={enddate}"
            f"&station={station_ID}"
            f"&product=predictions&datum=MLLW&time_zone=gmt"
            f"&units=english&format=csv"
        )

        # Print the URLs only if you're actually about to fetch them
        print(f"  Month {month:02d}: Downloading from:")
        print(f"    OBS  -> {obs_url}")
        print(f"    PREDS -> {preds_url}")

        obs_response   = requests.get(obs_url)
        preds_response = requests.get(preds_url)

        # Save observation CSV if we got an OK response
        if obs_response.status_code == 200:
            obs_data = pd.read_csv(StringIO(obs_response.text))
            obs_data.to_csv(obs_filepath, index=False)

        # Save prediction CSV if we got an OK response
        if preds_response.status_code == 200:
            preds_data = pd.read_csv(StringIO(preds_response.text))
            preds_data.to_csv(preds_filepath, index=False)
