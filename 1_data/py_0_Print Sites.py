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

# site = 'virginia_key'  # Change this as needed for different sites
# Argument parsing
parser = argparse.ArgumentParser(description="Run the script for a specific site")
parser.add_argument('--site', required=True, help='The site to run the script for (e.g., fort_myers)')
args = parser.parse_args()

# Use the provided site argument
site = args.site

print(site)
