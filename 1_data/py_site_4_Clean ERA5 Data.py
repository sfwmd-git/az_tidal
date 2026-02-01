import os
import platform
import numpy as np
import pandas as pd
import cdsapi
import requests
from datetime import datetime
from io import StringIO  # This is the correct import for StringIO
import calendar
import matplotlib.pyplot as plt
import argparse  # <-- Add this line

#------------------------------------------------------------------------------
#
#                 SET BASE DIRECTORY
#
#------------------------------------------------------------------------------

# Define the base directory based on the hostname
hostname = platform.node()
if hostname == 'Nates-MacBook-Pro.local' or hostname == 'UM-C02RQ1S8FVH7' or hostname == 'nates-mbp.lan' or hostname == 'Nates-MBP.lan':
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
raw_dir = os.path.join(BASE_DIR, '1_data', 'raw_era5_data')
norm_dir = os.path.join(BASE_DIR, '2_regressions', 'normalization_data')
output_dir = os.path.join(BASE_DIR, '1_data', 'clean_era5_data')
plot_dir = os.path.join(BASE_DIR, '4_plots')

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

site = 'lake_worth'  # Change this as needed for different sites

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
startyear = site_data['start_year'].iloc[0]
endyear = datetime.now().year - 1


#----------------------------------------------------------------------------------------------------------------------------------------------------
#
#                   READ ERA5 DATA
#
#----------------------------------------------------------------------------------------------------------------------------------------------------

# List all files in raw_dir matching the station_ID pattern
era_files_all = [
    f for f in os.listdir(raw_dir)
    if f"era5_{station_ID}" in f and os.path.isfile(os.path.join(raw_dir, f))
]

era_all = None  # Initialize the DataFrame to hold all observations

for file in era_files_all:
    file_path = os.path.join(raw_dir, file)
    print(f"Processing file: {file_path}")
    
    # Read the current CSV file
    era_temp = pd.read_csv(file_path)
    
    # Check if 'index' or 'date' exists and standardize
    if 'index' in era_temp.columns:
        era_temp.rename(columns={'index': 'date'}, inplace=True)
    elif 'date' not in era_temp.columns:
        print(f"Neither 'index' nor 'date' column found in {file_path}. Skipping file.")
        continue  # Skip this file if neither column exists

    # Convert 'date' to datetime
    era_temp['date'] = pd.to_datetime(era_temp['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce', utc=True)

    # Concatenate the cleaned DataFrame
    if era_all is None:
        era_all = era_temp  # If first file, initialize the DataFrame
    else:
        era_all = pd.concat([era_all, era_temp], ignore_index=True)

# Convert 'Date Time' column to datetime format after cleaning
era_all['date'] = pd.to_datetime(era_all['date'], format='%Y-%m-%d', errors='coerce', utc=True)

# Drop rows with invalid or missing 'Date Time' values
era_all = era_all.dropna(subset=['date'])

#----------------------------------------------------------------------------------------------------------------------------------------------------
#
#                   CLEAN ERA5 DATA
#
#----------------------------------------------------------------------------------------------------------------------------------------------------

era_all['slp'] = era_all['slp'] / 100.0  # Divide 'slp' by 100
# Extract only the date part (year-month-day) for grouping
era_all['day'] = era_all['date'].dt.date

ds = (
    era_all.groupby('day')
    .agg({
        'slp': 'mean',
        'u10': 'mean',
        'v10': 'mean',
        'sst': 'mean',
        'swh': 'mean'
    })
    .reset_index()
)

# Optional: filter by a start date if needed
startdate = f'{startyear}-01-01'
ds = ds[ds['day'] >= pd.to_datetime(startdate).date()]

# Step 2: Calculate standard deviation
stdev = (
    ds.iloc[:, 1:]  # Select columns 2:6 (slp, u10, v10, sst, swh)
    .std()
    .to_frame()
    .T  # Transpose to match the required structure
    .reset_index(drop=True)
)
stdev['data'] = 'sd'
stdev = stdev[['data'] + stdev.columns[:-1].tolist()]  # Reorder columns

# Step 3: Calculate mean
means = (
    ds.iloc[:, 1:]  # Select columns 2:6 (slp, u10, v10, sst, swh)
    .mean()
    .to_frame()
    .T  # Transpose to match the required structure
    .reset_index(drop=True)
)
means['data'] = 'mean'
means = means[['data'] + means.columns[:-1].tolist()]  # Reorder columns

# Step 4: Combine mean and standard deviation into a single DataFrame
combined_df = pd.concat([means, stdev], ignore_index=True)
combined_df.to_csv(f"{norm_dir}/{station_ID}_normalizations.csv", index=False)

# Extract the means and standard deviations explicitly
means = combined_df.loc[combined_df['data'] == 'mean'].iloc[0, 1:].astype(float)
stdevs = combined_df.loc[combined_df['data'] == 'sd'].iloc[0, 1:].astype(float)

# Create a new DataFrame 'era' with normalized columns
era = ds.copy()
era['slp_sd'] = (era['slp'] - means['slp']) / stdevs['slp']
era['u10_sd'] = (era['u10'] - means['u10']) / stdevs['u10']
era['v10_sd'] = (era['v10'] - means['v10']) / stdevs['v10']
era['sst_sd'] = (era['sst'] - means['sst']) / stdevs['sst']
era['swh_sd'] = (era['swh'] - means['swh']) / stdevs['swh']

era.rename(columns={'day': 'date'}, inplace=True)
era.to_csv(f"{output_dir}/{station_ID}_clean_era5_data.csv", index=False)
