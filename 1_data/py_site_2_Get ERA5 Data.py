import os
import platform
import numpy as np
import pandas as pd
import netCDF4
import requests
from datetime import datetime
from io import StringIO  # This is the correct import for StringIO
import calendar
from netCDF4 import Dataset, num2date
import cdsapi
import zipfile
import xarray as xr



#------------------------------------------------------------------------------
#
#                 SET BASE DIRECTORY
#
#------------------------------------------------------------------------------

# Define the base directory based on the hostname
hostname = platform.node()
if hostname == 'Nates-MacBook-Pro.local' or hostname == 'UM-C02RQ1S8FVH7' or hostname == 'nates-mbp.lan' or hostname == 'Nates-MBP.lan':
    BASE_DIR = '/Users/nate/Documents/tidal_work/'
elif hostname == 'Nates-MBP.localdomain':
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
output_dir = os.path.join(BASE_DIR, '1_data', 'raw_era5_data')

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
startyear = site_data['start_year'].iloc[0]
endyear = datetime.now().year - 1


# ------------------------------------------------------------------------------
#             
#             PULL ERA5 Data
# 
# ------------------------------------------------------------------------------

# Initialize CDS API client
c = cdsapi.Client()

for year in range(startyear, endyear + 1):
    # The CSV we plan to create
    csv_filename = os.path.join(output_dir, f"era5_{station_ID}_{year}.csv")
    # Skip if the CSV already exists
    if os.path.exists(csv_filename):
        print(f"CSV for year {year} already exists at '{csv_filename}' -- skipping download.")
        continue
    print(f"Retrieving data for year: {year}")
    # Define dataset and request parameters for the ATMOS data
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "sea_surface_temperature"
            ],
        "year": [str(year)],
        "month": ["01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
            ],
        "day": [
            "01", "02", "03", "04", "05", "06",
            "07", "08", "09", "10", "11", "12",
            "13", "14", "15", "16", "17", "18",
            "19", "20", "21", "22", "23", "24",
            "25", "26", "27", "28", "29", "30", "31"
        ],
        "time": [
            "00:00", "03:00", "06:00",
            "09:00", "12:00", "15:00",
            "18:00", "21:00"
        ],
        "data_format": "grib",
        "area": [nlat, wlon, slat, elon]  # Bounding box: [N, W, S, E]
    }
    
    # Initialize CDS API client and download data
    client = cdsapi.Client()
    output_file = "download.grib"
    client.retrieve(dataset, request).download(output_file)
    
    # Open and inspect the downloaded GRIB file
    ds = xr.open_dataset(output_file, engine="cfgrib")
    print(ds)  # Prints the dataset structure and variables
    
    # 2. Extract variables and average for each time step
    # Variables: 'msl', 'u10', 'v10', 'sst', 'swh' (Ensure all variables are available in `ds`)
    slp = ds["msl"]  # Mean sea level pressure
    u10 = ds["u10"]  # Zonal wind component at 10m
    v10 = ds["v10"]  # Meridional wind component at 10m
    sst = ds["sst"]  # Sea surface temperature
    # Convert time to pandas datetime
    time_var = ds["time"].values  # Extract time
    dtime = pd.to_datetime(time_var)
    
    # Create a DataFrame to store averages
    df = pd.DataFrame(columns=["slp", "u10", "v10", "sst", "swh"], index=dtime)
    
    # Fill the DataFrame with mean values
    df["slp"] = slp.mean(dim=["latitude", "longitude"]).values
    df["u10"] = u10.mean(dim=["latitude", "longitude"]).values
    df["v10"] = v10.mean(dim=["latitude", "longitude"]).values
    df["sst"] = sst.mean(dim=["latitude", "longitude"]).values

    # 3. Remove the GRIB file
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed '{output_file}' from disk.")
    
    # Define dataset and request parameters for the WAVE data
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["significant_height_of_combined_wind_waves_and_swell"],
        "year": [str(year)],
        "month": ["01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
            ],
        "day": [
            "01", "02", "03", "04", "05", "06",
            "07", "08", "09", "10", "11", "12",
            "13", "14", "15", "16", "17", "18",
            "19", "20", "21", "22", "23", "24",
            "25", "26", "27", "28", "29", "30", "31"
        ],
        "time": [
            "00:00", "03:00", "06:00",
            "09:00", "12:00", "15:00",
            "18:00", "21:00"
        ],
        "data_format": "grib",
        "area": [nlat, wlon, slat, elon]  # Bounding box: [N, W, S, E]
    }
    
    # Initialize CDS API client and download data
    client = cdsapi.Client()
    output_file = "download.grib"
    client.retrieve(dataset, request).download(output_file)
    
    # Open and inspect the downloaded GRIB file
    ds = xr.open_dataset(output_file, engine="cfgrib")
    print(ds)  # Prints the dataset structure and variables
    
    # 2. Extract variables and average for each time step
    # Variables: 'swh' (Ensure all variables are available in `ds`)
    swh = ds["swh"]  # Significant Wave Height
    # Fill the DataFrame with mean values
    df["swh"] = swh.mean(dim=["latitude", "longitude"]).values
    
    # Reset the index to turn it into a column and rename it to 'date'
    df = df.reset_index().rename(columns={"index": "date"})
    
    # 3. Remove the GRIB file
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed '{output_file}' from disk.")
    df.to_csv(csv_filename, index=False)
    print(f"Wrote CSV '{csv_filename}'")
    
    
