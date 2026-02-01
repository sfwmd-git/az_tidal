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
tide_dir = os.path.join(BASE_DIR, '1_data', 'clean_water_data')
era_dir = os.path.join(BASE_DIR, '1_data', 'clean_era5_data')
norm_dir = os.path.join(BASE_DIR, '2_regressions', 'normalization_data')
output_dir = os.path.join(BASE_DIR, '2_regressions', 'input_data')
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
startyear = site_data['start_year'].iloc[0]
endyear = datetime.now().year - 1

#----------------------------------------------------------------------------------------------------------------------------------------------------
#
#                   COMBINE ERA AND TIDE
#
#----------------------------------------------------------------------------------------------------------------------------------------------------
# Read the ERA5 data
era = pd.read_csv(f"{era_dir}/{station_ID}_clean_era5_data.csv")
# Read the tide gauge data
obs = pd.read_csv(f"{tide_dir}/{station_ID}_clean_water_level.csv")

# Extract the 'date' column from the ERA5 data
dates = era['date']

# Join the `obs` and `era` DataFrames on the "date" column
ds = (
    era.merge(obs, on="date", how="left")
    .loc[:, ["date", "departure_adj", "slp", "slp_sd", "u10_sd", "v10_sd", "sst_sd", "swh_sd"]]
)

#----------------------------------------------------------------------------------------------------------------------------------------------------
#
#                   REMOVE ROWS WITH EXTREME VALUES (>3 STD FROM MEAN)
#
#----------------------------------------------------------------------------------------------------------------------------------------------------

# Define a threshold for extreme values (e.g., z-score > 3)
z_threshold = 3
upper_limit = 4

# Option 1: Replace extreme values with NaN
def clean_extreme_values_with_nan(df, columns, threshold, limit):
    df_cleaned = df.copy()
    for col in columns:
        df_cleaned[col] = np.where(df_cleaned[col].abs() > threshold, np.nan, df_cleaned[col])
    df_cleaned['swh_sd'] = np.where(df_cleaned['swh_sd'].abs() > limit, np.nan, df_cleaned['swh_sd'])
    return df_cleaned

# Specify the columns to check for extreme values (excluding 'date' and 'departure_adj')
columns_to_clean = ['slp_sd', 'u10_sd', 'v10_sd', 'sst_sd']
# Apply Option 1: Replace extreme values with NaN
ds_clean_nan = clean_extreme_values_with_nan(ds, columns_to_clean, z_threshold, upper_limit)


# Get a boolean mask of rows with at least one NaN value (excluding 'date' column)
rows_with_nan = ds_clean_nan[['departure_adj', 'slp', 'slp_sd', 'u10_sd', 'v10_sd', 'sst_sd', 'swh_sd']].isna().any(axis=1)
# Replace all values (except 'date') with NaN for rows with at least one NaN
ds_clean_nan.loc[rows_with_nan, ds_clean_nan.columns!= 'date'] = pd.NA
# Print the number of rows modified
print(f"Modified {rows_with_nan.sum()} rows")

ds_clean_nan.to_csv(f"{output_dir}/{station_ID}_regression_input.csv", index=False)

