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
import matplotlib.pyplot as plt
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
raw_dir = os.path.join(BASE_DIR, '1_data', 'raw_water_data')
output_dir = os.path.join(BASE_DIR, '1_data', 'clean_water_data')
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

# Argument parsing
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
startyear = 1996
endyear = datetime.now().year - 1

#----------------------------------------------------------------------------------------------------------------------------------------------------
#
#                   READ OBSERVATIONS DATA
#
#----------------------------------------------------------------------------------------------------------------------------------------------------

# List all files in raw_dir matching the station_ID pattern
obs_files_all = [
    f for f in os.listdir(raw_dir)
    if f"{station_ID}_observations" in f and os.path.isfile(os.path.join(raw_dir, f))
]

obs_all = None  # Initialize the DataFrame to hold all observations

for file in obs_files_all:
    file_path = os.path.join(raw_dir, file)
    print(f"Processing file: {file_path}")
    
    # Read the current CSV file
    obs_temp = pd.read_csv(file_path)
    
    # Filter out rows with invalid 'Date Time' entries
    valid_rows = ~obs_temp['Date Time'].str.contains("Error:", na=False)
    obs_temp = obs_temp[valid_rows]
    
    # Concatenate the cleaned DataFrame
    if obs_all is None:
        obs_all = obs_temp  # If first file, initialize the DataFrame
    else:
        obs_all = pd.concat([obs_all, obs_temp], ignore_index=True)

# Convert 'Date Time' column to datetime format after cleaning
obs_all['Date Time'] = pd.to_datetime(obs_all['Date Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce', utc=True)

# Drop rows with invalid or missing 'Date Time' values
obs_all = obs_all.dropna(subset=['Date Time'])

#----------------------------------------------------------------------------------------------------------------------------------------------------
#
#                   READ PREDICTION DATA
#
#----------------------------------------------------------------------------------------------------------------------------------------------------

# List all files in raw_dir matching the station_ID pattern
pred_files_all = [
    f for f in os.listdir(raw_dir)
    if f"{station_ID}_predictions" in f and os.path.isfile(os.path.join(raw_dir, f))
]

pred_all = None  # Initialize the DataFrame to hold all predervations

for file in pred_files_all:
    file_path = os.path.join(raw_dir, file)
    print(f"Processing file: {file_path}")
    
    # Read the current CSV file
    pred_temp = pd.read_csv(file_path)
    
    # Filter out rows with invalid 'Date Time' entries
    valid_rows = ~pred_temp['Date Time'].str.contains("Error:", na=False)
    pred_temp = pred_temp[valid_rows]
    
    # Concatenate the cleaned DataFrame
    if pred_all is None:
        pred_all = pred_temp  # If first file, initialize the DataFrame
    else:
        pred_all = pd.concat([pred_all, pred_temp], ignore_index=True)

# Convert 'Date Time' column to datetime format after cleaning
pred_all['Date Time'] = pd.to_datetime(pred_all['Date Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce', utc=True)

# Drop rows with invalid or missing 'Date Time' values
pred_all = pred_all.dropna(subset=['Date Time'])
#------------------------------------------------------------------------------
#
#              INITIAL CLEAN AND SUMMARIZE OBS AND PRED DATA DAILY
#
#------------------------------------------------------------------------------

# Process obsAll data
dsObs = (
    obs_all[['Date Time', ' Water Level']]  # Select relevant columns
    .rename(columns={'Date Time': 'date', ' Water Level': 'water_level'})  # Rename columns
    .assign(date=lambda x: pd.to_datetime(x['date']).dt.date)  # Convert to date
    .groupby('date', as_index=False)  # Group by date
    .agg(water_level=('water_level', 'max'), n=('water_level', 'count'))  # Summarize data
    .query('n >= 180')  # Filter rows where n >= 180
)

# Process predAll data
dsPred = (
    pred_all[['Date Time', ' Prediction']]  # Select relevant columns
    .rename(columns={'Date Time': 'date', ' Prediction': 'prediction'})  # Rename columns
    .assign(date=lambda x: pd.to_datetime(x['date']).dt.date)  # Convert to date
    .groupby('date', as_index=False)  # Group by date
    .agg(prediction=('prediction', 'max'))  # Summarize data
)

#------------------------------------------------------------------------------
#
#              COMBINE DATA AND CALCULATE TRENDLINE OF OBSERVATIONS
#
#------------------------------------------------------------------------------

# Merge dsPred and dsObs on 'date' and keep only specified columns
dsCombo = (
    dsPred.merge(dsObs, on='date', how='left')  # Perform a left join on the 'date' column
    .assign(departure=lambda x: x['water_level'] - x['prediction'])  # Calculate departure
    [['date', 'water_level', 'prediction', 'departure']]  # Keep only specific columns
)

#-------------------------------------------------------------------------------
#
#              CALCULATE AND ADJUST FOR SLR
#
#-------------------------------------------------------------------------------

# Sort by date to ensure proper rolling calculations
dsCombo = dsCombo.sort_values(by='date')
# Define the rolling window (3 years in days)
rolling_window = 365 * 3
# Calculate the rolling mean (sea level rise approximation)
dsCombo['slr'] = (
  dsCombo['departure']
  .rolling(window=rolling_window, min_periods=1, center=False)
  .mean()
)
# Calculate the adjusted departure
dsCombo['departure_adj'] = dsCombo['departure'] - dsCombo['slr']

#-------------------------------------------------------------------------------
#
#              UPDATE SLR IN SITE DATA
#
#-------------------------------------------------------------------------------

# 1) Get the last SLR value from dsCombo
last_slr_value = dsCombo['slr'].iloc[-1]
# 2) Print the last SLR value
print(f"Last SLR Value: {last_slr_value}")
# 3) Update df (make sure the 'slr_adjustment' column exists or is created)
df.loc[df['site_name'] == site, 'slr_adjustment'] = last_slr_value
# -- EXTRA LINE: reorder the columns so that 'slr_adjustment' is right after 'start_year' --
df = df.reindex(columns=df.columns.drop('slr_adjustment').insert(
    df.columns.get_loc('start_year')+1, 'slr_adjustment'
))
# 4) Save the updated df to station_data.csv (same location it was originally read from)
df.to_csv(csv_file, index=False)

#-------------------------------------------------------------------------------
#
#              REMOVE OUTLIERS
#
#-------------------------------------------------------------------------------

# Calculate the mean and standard deviation of 'departure_adj'
mean_departure_adj = dsCombo['departure_adj'].mean()
std_departure_adj = dsCombo['departure_adj'].std()

# Filter out rows where 'departure_adj' is outside 3 standard deviations
filtered_dsCombo = dsCombo[
    (dsCombo['departure_adj'] >= mean_departure_adj - 3 * std_departure_adj) &
    (dsCombo['departure_adj'] <= mean_departure_adj + 3 * std_departure_adj)
]

#-------------------------------------------------------------------------------
#
#              SAVE DATA AND CREATE PLOTS
#
#-------------------------------------------------------------------------------

filtered_dsCombo.to_csv(f"{output_dir}/{station_ID}_clean_water_level.csv", index=False)


# Plot water_level, prediction, and departure
plt.figure(figsize=(12, 6))

# Plot water_level
plt.plot(filtered_dsCombo['date'], filtered_dsCombo['water_level'], label='Water Level', linestyle='-', linewidth=1)
# Plot prediction
plt.plot(filtered_dsCombo['date'], filtered_dsCombo['prediction'], label='Prediction', linestyle='-', linewidth=1)
# Plot departure
plt.plot(filtered_dsCombo['date'], filtered_dsCombo['departure'], label='Departure', linestyle='--', linewidth=1)
# Plot departure_adj
plt.plot(filtered_dsCombo['date'], filtered_dsCombo['departure_adj'], label='SLR Adjusted Departure', linestyle='-', linewidth=1, color = 'purple')
# Plot slr
plt.plot(filtered_dsCombo['date'], filtered_dsCombo['slr'], label='Sea Level Rise (SLR)', linestyle='-', linewidth=1, color = 'red')

# Add title and labels
plt.title(f'{station_name} Water Level, Prediction, and Departure by Date')
plt.xlabel('Date')
plt.ylabel('Values')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot (adjust the extension if you prefer .pdf or .jpg)
plot_filename = f"{site_name}_{station_ID}_water_level_adjustments.png"
plt.savefig(os.path.join(plot_dir, plot_filename), dpi=300)
