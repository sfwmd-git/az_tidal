import os
import platform
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import time
from pytz import timezone
import pytz
import requests
import pygrib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math
import argparse

#------------------------------------------------------------------------------
#
#                 SET MANUAL
#
#------------------------------------------------------------------------------

# # Uncomment this setting site manually
# site = 'naples_bay'  # Change this as needed for different sites

# Uncomment this section if using csv file
parser = argparse.ArgumentParser(description="Run the script for a specific site")
parser.add_argument('--site', required=True, help='The site to run the script for (e.g., fort_myers)')
args = parser.parse_args()
# Use the provided site argument
site = args.site

model = "GFS"
forecast_type = "ENS"

sleep_time = 4  # If this remains constant
db = 0  # If this remains constant
members = 30  # If this remains constant
experiment = ""

#------------------------------------------------------------------------------
#
#                 SET DIRECTORIES
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

if experiment == "":
    test = experiment
else:
    test = f"{experiment}_"

stationinfo_subdir = f'0_station_info/'
logo_subdir = f'0_station_info/logos/'
forecast_subdir = f'1_data/forecasts/{model}/'
regression_subdir = f'{test}2_regressions/'
predictions_subdir = f'{test}3_predictions/'

# Define other directories relative to the base directory and site-specific subdirectories
META_DIR = os.path.join(BASE_DIR, stationinfo_subdir)
LOGO_DIR = os.path.join(BASE_DIR, logo_subdir)
FORECAST_DIR = os.path.join(BASE_DIR, forecast_subdir)
NORM_DIR = os.path.join(BASE_DIR, regression_subdir, "normalization_data/")
COEF_DIR = os.path.join(BASE_DIR, regression_subdir, "regression_use_data/")
OUTPUT_DIR = os.path.join(BASE_DIR, predictions_subdir)

os.chdir(BASE_DIR)

#------------------------------------------------------------------------------
#
#                 LOAD STATION DATA FROM CSV
#
#------------------------------------------------------------------------------

# Load station-specific data from CSV
csv_file = os.path.join(META_DIR, 'station_data.csv')
df = pd.read_csv(csv_file)

# Filter the row for the specified site
station_data = df[df['site_name'] == site].iloc[0]

#------------------------------------------------------------------------------
#
#                 ASSIGN VARIABLES MANUALLY
#
#------------------------------------------------------------------------------

model_under = model.lower()
forecast_under = forecast_type.lower()
# forecast GRIB Dims.
forecast_nlat_idx = int(station_data[f'{model_under}_{forecast_under}_forecast_nlat_idx'])
forecast_slat_idx = int(station_data[f'{model_under}_{forecast_under}_forecast_slat_idx'])
forecast_wlon_idx = int(station_data[f'{model_under}_{forecast_under}_forecast_wlon_idx'])
forecast_elon_idx = int(station_data[f'{model_under}_{forecast_under}_forecast_elon_idx'])

# wave GRIB Dims.
wave_nlat_idx = int(station_data[f'{model_under}_{forecast_under}_wave_nlat_idx'])
wave_slat_idx = int(station_data[f'{model_under}_{forecast_under}_wave_slat_idx'])
wave_wlon_idx = int(station_data[f'{model_under}_{forecast_under}_wave_wlon_idx'])
wave_elon_idx = int(station_data[f'{model_under}_{forecast_under}_wave_elon_idx'])

#------------------------------------------------------------------------------
#
#                 ASSIGN VARIABLES FROM CSV
#
#------------------------------------------------------------------------------

DATUM = "MHHW"  # If this remains constant
station_ID = str(station_data['station_ID'])
station_abbr = station_data['station_abbr']
station_name = station_data['station_name']
slr_adjustment = station_data['slr_adjustment']
wlon = station_data['wlon'] + 360
elon = station_data['elon'] + 360
slat = station_data['slat']
nlat = station_data['nlat']
MHHW_MLLW_offset = station_data['MHHW_MLLW_offset']
minor_flood = station_data['minor_flood']
moderate_flood = station_data['moderate_flood']
major_flood = station_data['major_flood']
navd_offset = station_data['navd_offset']
ngvd_offset = station_data['ngvd_offset']

# Build each directory name and store in a list
directories_to_make = [
    os.path.join(OUTPUT_DIR, f"{model}_{forecast_type}_all", f"{station_ID}_{site}"),
    os.path.join(OUTPUT_DIR, f"{model}_{forecast_type}_graphs", f"{station_ID}_{site}"),
    os.path.join(OUTPUT_DIR, f"{model}_{forecast_type}_master", f"{station_ID}_{site}"),
    os.path.join(OUTPUT_DIR, f"{model}_{forecast_type}_components", f"{station_ID}_{site}")
]

# Create each directory if it doesn't exist
for dir_path in directories_to_make:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Ensured directory exists: {dir_path}")

#------------------------------------------------------------------------------
#
#                 SET URLs
#
#------------------------------------------------------------------------------
# Dates
today = datetime.now() - timedelta(days=db)
yesterday = today - timedelta(days=1)
ten_day = today + timedelta(days=9)

# Date Strings
yesterday_str = yesterday.strftime('%Y%m%d')
today_str = today.strftime('%Y%m%d')
ten_day_str = ten_day.strftime('%Y%m%d')

# Tide URLs
station_tide_url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={yesterday_str}&end_date={yesterday_str}&station={station_ID}&product=water_level&datum={DATUM}&time_zone=lst_ldt&units=english&format=csv"
station_pred_url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={yesterday_str}&end_date={ten_day_str}&station={station_ID}&product=predictions&datum={DATUM}&time_zone=lst_ldt&units=english&format=csv"
station_tide_url_NAVD = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={yesterday_str}&end_date={yesterday_str}&station={station_ID}&product=water_level&datum=NAVD&time_zone=lst_ldt&units=english&format=csv"
station_pred_url_NAVD = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={yesterday_str}&end_date={ten_day_str}&station={station_ID}&product=predictions&datum=NAVD&time_zone=lst_ldt&units=english&format=csv"

#------------------------------------------------------------------------------
#
#                 SET LAST HOUR DEPARTURE
#
#------------------------------------------------------------------------------

# Obtain water level data from NOAA
station_tide = pd.read_csv(station_tide_url)
station_pred = pd.read_csv(station_pred_url)

# Clean water level data from NOAA
station_tide["Date Time"] = pd.to_datetime(station_tide["Date Time"])
station_wobs = (
    station_tide
    .groupby([
        station_tide["Date Time"].dt.date.rename("Date"), 
        station_tide["Date Time"].dt.hour.rename("Hour")
    ])[" Water Level"]  # <- note there's a leading space in your CSV column name
    .mean()
    .reset_index()
)
# Optionally rename " Water Level" to something else:
station_wobs = station_wobs.rename(columns={" Water Level": "Observation"})
station_wobs["Date"] = pd.to_datetime(station_wobs["Date"])

# Clean prediction data from NOAA
station_pred["Date Time"] = pd.to_datetime(station_pred["Date Time"])
station_maxs = (
    station_pred
    .groupby([
        station_pred["Date Time"].dt.date.rename("Date"), 
        station_pred["Date Time"].dt.hour.rename("Hour")
    ])[" Prediction"]  # <- note there's a leading space in your CSV column name
    .mean()
    .reset_index()
)
# Optionally rename " Water Level" to something else:
station_maxs = station_maxs.rename(columns={" Prediction": "Prediction"})
station_maxs["Date"] = pd.to_datetime(station_maxs["Date"])

# Add in SLR constant
station_maxs["SLR"] = slr_adjustment
station_maxs["SLR_adj"] = station_maxs["Prediction"] + station_maxs["SLR"]

# Build forecast dataframe
station_fors = pd.merge(station_maxs, station_wobs, on=["Date", "Hour"], how="left")
station_fors["Departure"] = station_fors["Observation"] - station_fors["SLR_adj"]

# 1) Filter for rows where Hour = 23 and Departure is non-null
valid_rows = station_fors[
    (station_fors["Hour"] == 23) & (station_fors["Departure"].notna())
]
# 2) Get the last (most recent) departure from that filtered set
if not valid_rows.empty:
    last_dep = valid_rows["Departure"].iloc[-1]
    last_slr = valid_rows["SLR"].iloc[-1]
else:
    last_dep = None
    last_slr = None


#------------------------------------------------------------------------------
#
#                 GET TIDE DATA TOGETHER
#
#------------------------------------------------------------------------------

# Obtain water level data from NOAA
station_tide = pd.read_csv(station_tide_url)
station_pred = pd.read_csv(station_pred_url)

# Clean water level data from NOAA
station_tide["Date Time"] = pd.to_datetime(station_tide["Date Time"])
station_wobs = station_tide.groupby(station_tide["Date Time"].dt.date)[" Water Level"].max().reset_index().rename(columns={"Date Time": "Date", " Water Level":"Observation"})
station_wobs["Date"] = pd.to_datetime(station_wobs["Date"])

station_pred["Date Time"] = pd.to_datetime(station_pred["Date Time"])
station_maxs = station_pred.groupby(station_pred["Date Time"].dt.date)[" Prediction"].max().reset_index().rename(columns={"Date Time": "Date", " Prediction":"Prediction"})
station_maxs["Date"] = pd.to_datetime(station_maxs["Date"])

# Add in SLR constant
station_maxs["SLR"] = slr_adjustment
station_maxs["SLR_adj"] = station_maxs["Prediction"] + station_maxs["SLR"]

# Build forecast dataframe
station_fors = pd.merge(station_maxs, station_wobs, on="Date", how="outer")
station_fors["Departure"] = station_fors["Observation"] - station_fors["SLR_adj"]
station_fors["day"] = range(11)
print(station_fors)

#------------------------------------------------------------------------------
#                 GET TIDE DATA TOGETHER - NAVD
#------------------------------------------------------------------------------

# Obtain water level data from NOAA
station_tide_NAVD = pd.read_csv(station_tide_url_NAVD)
station_pred_NAVD = pd.read_csv(station_pred_url_NAVD)

# Clean water level data from NOAA
station_tide_NAVD["Date Time"] = pd.to_datetime(station_tide_NAVD["Date Time"])
station_wobs_NAVD = station_tide_NAVD.groupby(station_tide_NAVD["Date Time"].dt.date)[" Water Level"].max().reset_index().rename(columns={"Date Time": "Date", " Water Level":"Observation"})
station_wobs_NAVD["Date"] = pd.to_datetime(station_wobs_NAVD["Date"])

station_pred_NAVD["Date Time"] = pd.to_datetime(station_pred_NAVD["Date Time"])
station_maxs_NAVD = station_pred_NAVD.groupby(station_pred_NAVD["Date Time"].dt.date)[" Prediction"].max().reset_index().rename(columns={"Date Time": "Date", " Prediction":"Prediction"})
station_maxs_NAVD["Date"] = pd.to_datetime(station_maxs_NAVD["Date"])

# Add in SLR constant
station_maxs_NAVD["SLR"] = slr_adjustment
station_maxs_NAVD["SLR_adj"] = station_maxs_NAVD["Prediction"] + station_maxs_NAVD["SLR"]

# Build forecast dataframe
station_fors_NAVD = pd.merge(station_maxs_NAVD, station_wobs_NAVD, on="Date", how="outer")
station_fors_NAVD["Departure"] = station_fors_NAVD["Observation"] - station_fors_NAVD["SLR_adj"]
station_fors_NAVD["day"] = range(11)
print(station_fors_NAVD)


#------------------------------------------------------------------------------
#
#                 GET GFS DATA TOGETHER
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                 Functions
#------------------------------------------------------------------------------

# Getting SST
def get_average_sst(file):
    try:
        gr = pygrib.open(file)
        msg = gr.message(2)  # Use .message(2) instead of indexing with [2]
        sst_vals = msg.values
        avg_sst = sst_vals[forecast_nlat_idx:forecast_slat_idx+1, forecast_wlon_idx:forecast_elon_idx+1].mean()
        return avg_sst
    except Exception:
        print(f"Skipping file {file} due to lack of sst data")
        return None
    finally:
        gr.close()

# Define the file name generation functions
def generate_gfs_file_name(date, hour, forecast_hour):
    formatted_date = date.strftime("%Y%m%d")
    return f"gfsall.{formatted_date}.t{hour:02d}z.f{forecast_hour:03d}.grib2"

def process_gfs_grib_files():
    data = []

    today = datetime.now() - timedelta(days=db)
    two_days_ago = today - timedelta(days=2)
    one_day_ago = today - timedelta(days=1)

    utc = pytz.utc
    est = pytz.timezone('US/Eastern')

    for day, num_files, hours in [(two_days_ago, 3, [6, 12, 18]), (one_day_ago, 4, range(0, 24, 6)), (today, 2, range(0, 6, 6))]:
        for hour in hours:
            file_name = generate_gfs_file_name(day, hour, 0)
            if num_files > 0:
                try:
                    avg_sst = get_average_sst(file_name)
                    if avg_sst is not None:
                        datetime_utc = utc.localize(datetime.combine(day, datetime.min.time()) + timedelta(hours=hour))
                        datetime_est = datetime_utc.astimezone(est)
                        data.append({'datetime_est': datetime_est, 'average_sst': avg_sst})
                except Exception:
                    print(f"Skipping file {file_name} because GFS atmos forecast did not run")
                num_files -= 1

    for forecast_hour in range(3, 240, 3):
        file_name = generate_gfs_file_name(today, 0, forecast_hour)
        try:
            avg_sst = get_average_sst(file_name)
            if avg_sst is not None:
                datetime_utc = utc.localize(datetime.combine(today, datetime.min.time()) + timedelta(hours=6 + forecast_hour))
                datetime_est = datetime_utc.astimezone(est)
                data.append({'datetime_est': datetime_est, 'average_sst': avg_sst})
        except Exception:
            print(f"Skipping file {file_name} because GFS atmos forecast did not run")

    return pd.DataFrame(data, columns=['datetime_est', 'average_sst'])


# Getting GFS Ensembles together 

def get_average_slp(file):
    try:
        gr = pygrib.open(file)
        msg = gr.message(3)  # Use .message(1) instead of indexing with [1]
        slp_vals = msg.values
        avg_slp = slp_vals[forecast_nlat_idx:forecast_slat_idx+1, forecast_wlon_idx:forecast_elon_idx+1].mean()
        return avg_slp
    except Exception:
        print(f"Skipping file {file} due to lack of slp data")
        return None
    finally:
        gr.close()

def get_average_u10(file):
    try:
        gr = pygrib.open(file)
        msg = gr.message(1)  # Use .message(3) instead of indexing with [3]
        u10_vals = msg.values
        avg_u10 = u10_vals[forecast_nlat_idx:forecast_slat_idx+1, forecast_wlon_idx:forecast_elon_idx+1].mean()
        return avg_u10
    except Exception:
        print(f"Skipping file {file} due to lack of u10 data")
        return None
    finally:
        gr.close()

def get_average_v10(file):
    try:
        gr = pygrib.open(file)
        msg = gr.message(2)  # Use .message(4) instead of indexing with [4]
        v10_vals = msg.values
        avg_v10 = v10_vals[forecast_nlat_idx:forecast_slat_idx+1, forecast_wlon_idx:forecast_elon_idx+1].mean()
        return avg_v10
    except Exception:
        print(f"Skipping file {file} due to lack of v10 data")
        return None
    finally:
        gr.close()

def generate_gefs_file_name(date, hour, forecast_hour, member):
    formatted_date = date.strftime("%Y%m%d")
    return f"gep{member:02d}.{formatted_date}.t{hour:02d}z.f{forecast_hour:03d}.grib2"

def process_gefs_grib_files(member):
    data = []

    today = datetime.now() - timedelta(days=db)
    two_days_ago = today - timedelta(days=2)
    one_day_ago = today - timedelta(days=1)

    utc = pytz.utc
    est = pytz.timezone('US/Eastern')

    for day, num_files, hours in [(two_days_ago, 3, [6, 12, 18]), (one_day_ago, 4, range(0, 24, 6)), (today, 1, range(0, 6, 6))]:
        for hour in hours:
            file_name = generate_gefs_file_name(day, hour, 0, member)
            if num_files > 0:
                try:
                    avg_slp = get_average_slp(file_name)
                    avg_u10 = get_average_u10(file_name)
                    avg_v10 = get_average_v10(file_name)
                    if avg_slp is not None and avg_u10 is not None and avg_v10 is not None:
                        datetime_utc = utc.localize(datetime.combine(day, datetime.min.time()) + timedelta(hours=hour))
                        datetime_est = datetime_utc.astimezone(est)
                        data.append({'datetime_est': datetime_est, 'average_slp': avg_slp, 'average_u10': avg_u10, 'average_v10': avg_v10})
                except Exception:
                    print(f"Skipping file {file_name} because GFS atmos forecast did not run")
                num_files -= 1

    for forecast_hour in range(3, 243, 3):
        file_name = generate_gefs_file_name(today, 0, forecast_hour, member)
        try:
            avg_slp = get_average_slp(file_name)
            avg_u10 = get_average_u10(file_name)
            avg_v10 = get_average_v10(file_name)
            if avg_slp is not None and avg_u10 is not None and avg_v10 is not None:
                datetime_utc = utc.localize(datetime.combine(today, datetime.min.time()) + timedelta(hours=6 + forecast_hour))
                datetime_est = datetime_utc.astimezone(est)
                data.append({'datetime_est': datetime_est, 'average_slp': avg_slp, 'average_u10': avg_u10, 'average_v10': avg_v10})
        except Exception:
            print(f"Skipping file {file_name} because GFS atmos forecast did not run")

    return pd.DataFrame(data, columns=['datetime_est', 'average_slp', 'average_u10', 'average_v10'])


# Getting Wave Ensembles together 
  
def get_average_swh(file):
    try:
        gr = pygrib.open(file)
        msg = gr.message(1)  # Attempt to get the 5th message
        swh_vals = msg.values
        avg_swh = swh_vals[wave_nlat_idx:wave_slat_idx+1, wave_wlon_idx:wave_elon_idx+1].mean()
        return avg_swh
    except Exception:
        print(f"Skipping file {file} due to lack of swh data")
        return None
    finally:
        gr.close()

def generate_wave_file_name(date, hour, forecast_hour, member):
    formatted_date = date.strftime("%Y%m%d")
    return f"gefs.wave.p{member:02d}.{formatted_date}.t{hour:02d}z.f{forecast_hour:03d}.grib2"

def process_wave_grib_files(member):
    data = []

    today = datetime.now() - timedelta(days=db)
    two_days_ago = today - timedelta(days=2)
    one_day_ago = today - timedelta(days=1)

    utc = pytz.utc
    est = pytz.timezone('US/Eastern')

    for day, num_files, hours in [(two_days_ago, 3, [6, 12, 18]), (one_day_ago, 4, range(0, 24, 6)), (today, 2, range(0, 6, 6))]:
        for hour in hours:
            file_name = generate_wave_file_name(day, hour, 0, member)
            if num_files > 0:
                try:
                    avg_swh = get_average_swh(file_name)
                    if avg_swh is not None:
                        datetime_utc = utc.localize(datetime.combine(day, datetime.min.time()) + timedelta(hours=hour))
                        datetime_est = datetime_utc.astimezone(est)
                        data.append({'datetime_est': datetime_est, 'average_swh': avg_swh})
                except Exception:
                    print(f"Skipping file {file_name} because GFS wave forecast did not run")
                num_files -= 1

    for forecast_hour in range(3, 240, 3):
        file_name = generate_wave_file_name(today, 0, forecast_hour, member)
        try:
            avg_swh = get_average_swh(file_name)
            if avg_swh is not None:
                datetime_utc = utc.localize(datetime.combine(today, datetime.min.time()) + timedelta(hours=6 + forecast_hour))
                datetime_est = datetime_utc.astimezone(est)
                data.append({'datetime_est': datetime_est, 'average_swh': avg_swh})
        except Exception:
            print(f"Skipping file {file_name} because GFS wave forecast did not run")

    return pd.DataFrame(data, columns=['datetime_est', 'average_swh'])  

#------------------------------------------------------------------------------
#                 FUNCTIONS TO PUT IT ALL TOGETHER AND CREATE THE FORECAST
#------------------------------------------------------------------------------

def create_forecast(member):
    os.chdir(FORECAST_DIR)
    sst = process_gfs_grib_files()
    #print(sst)
    sst['datetime_est'] = pd.to_datetime(sst['datetime_est'])
    # Remove timezone information and set the index to 'datetime_est'
    sst.index = sst['datetime_est']
    sst = sst.drop(columns=['datetime_est'])
    # Calculate daily average for 'sst'
    sst = sst.resample('D').mean()
    
    #------------------------------------------------------------------------------
    #                 GET AND CALC GEFS
    #------------------------------------------------------------------------------
    os.chdir(FORECAST_DIR)
    gfs = process_gefs_grib_files(member)
    #print(gfs)
    gfs['datetime_est'] = pd.to_datetime(gfs['datetime_est'])
    # Remove timezone information and set the index to 'datetime_est'
    gfs.index = gfs['datetime_est']
    gfs = gfs.drop(columns=['datetime_est'])
    # Calculate daily average for 'gfs'
    gfs = gfs.resample('D').mean()
    
    #------------------------------------------------------------------------------
    #                 READ AND CALC SWH 
    #------------------------------------------------------------------------------
    os.chdir(FORECAST_DIR)
    swh = process_wave_grib_files(member)
    #print(swh)
    swh['datetime_est'] = pd.to_datetime(swh['datetime_est'])
    # Remove timezone information and set the index to 'datetime_est'
    swh.index = swh['datetime_est']
    swh = swh.drop(columns=['datetime_est'])
    # Calculate daily average for 'swh'
    swh = swh.resample('D').mean()
    
    #------------------------------------------------------------------------------
    #                 COMBINE DATA
    #------------------------------------------------------------------------------
    # Reset the index for both DataFrames, converting the datetime to the desired format
    swh_reset = swh.reset_index().rename(columns={'datetime_est': 'date'})
    swh_reset['date'] = swh_reset['date'].dt.strftime('%Y-%m-%d')
    
    gfs_reset = gfs.reset_index().rename(columns={'datetime_est': 'date'})
    gfs_reset['date'] = gfs_reset['date'].dt.strftime('%Y-%m-%d')
    
    sst_reset = sst.reset_index().rename(columns={'datetime_est': 'date'})
    sst_reset['date'] = sst_reset['date'].dt.strftime('%Y-%m-%d')
    
    # Merge the DataFrames on the 'date' column
    prelim = pd.merge(sst_reset, gfs_reset, on='date')
    forecast = pd.merge(prelim, swh_reset, on='date')
    # Rename the 'average_swh' column to 'swh'
    forecast = forecast.rename(columns={'average_swh': 'swh', 'average_slp': 'prmslmsl', 'average_sst': 'tmpsfc', 'average_u10': 'ugrd10m', 'average_v10': 'vgrd10m'})
    
    forecast1 = forecast
    
    #------------------------------------------------------------------------------
    #                 GET NORMALIZATIONS
    #------------------------------------------------------------------------------

    # fix for naples_bay first
    if site == "naples_bay":
      station_ID = "8725110"
    else:
      station_ID = str(station_data['station_ID'])
      
    norms = pd.read_csv(NORM_DIR+station_ID+"_normalizations.csv")
    
    forecast['slp_delta'] = (forecast['prmslmsl'] / 100) - (forecast.loc[1, 'prmslmsl'] / 100)
    # Calculate the 3-day running mean for tmpsfc, ugrd10m, vgrd10m, and swh columns
    forecast['tmpsfc_3day_mean'] = forecast['tmpsfc'].rolling(window=3).mean()
    forecast['ugrd10m_3day_mean'] = forecast['ugrd10m'].rolling(window=3).mean()
    forecast['vgrd10m_3day_mean'] = forecast['vgrd10m'].rolling(window=3).mean()
    forecast['swh_3day_mean'] = forecast['swh'].rolling(window=3).mean()
    
    # Normalize the 3-day running mean columns using norms dataframe
    forecast['sst_norm'] = (forecast['tmpsfc_3day_mean'] - norms.loc[0, 'sst']) / norms.loc[1, 'sst']
    forecast['u10_norm'] = (forecast['ugrd10m_3day_mean'] - norms.loc[0, 'u10']) / norms.loc[1, 'u10']
    forecast['v10_norm'] = (forecast['vgrd10m_3day_mean'] - norms.loc[0, 'v10']) / norms.loc[1, 'v10']
    forecast['swh_norm'] = (forecast['swh_3day_mean'] - norms.loc[0, 'swh']) / norms.loc[1, 'swh']
    
    forecast_norm = forecast[['date', 'slp_delta', 'sst_norm', 'u10_norm', 'v10_norm', 'swh_norm']]
    forecast_norm = forecast_norm.copy()
    forecast_norm['date'] = pd.to_datetime(forecast_norm['date'])
    forecast_norm['date'] = pd.to_datetime(forecast_norm['date'])
    print(forecast_norm)
    
    return forecast_norm

def build_from_datum(forecast_norm, station_fors):
    # fix for naples_bay first
    station_ID = str(station_data['station_ID'])
    # Define the start date
    start_date = today.date()
    # Create a list of dates from start_date to 9 days ahead
    dates = [start_date + timedelta(days=i) for i in range(-1, 10)]
    # Create a list of days_ahead values
    days_ahead = list(range(0, 11))
    # Create an empty DataFrame called "predictions" with date and days_ahead columns
    predictions = pd.DataFrame({"date": dates, "days_ahead": days_ahead})
    predictions['date'] = pd.to_datetime(predictions['date'])
    # Merge the DataFrames
    predictions1 = predictions.merge(station_fors, left_on="date", right_on="Date", how="left")
    predictions1 = predictions1.merge(forecast_norm, on="date", how="left")
    predictions = predictions1.merge(coefs, left_on="days_ahead", right_on="day", how="left")
    # print(predictions)
    # Calculate the effects
    predictions["slp_effect"] = predictions["slp_delta"] * -0.0328084
    predictions["departure_effect"] = station_fors["Departure"].iloc[0] * predictions["dep_coef"]
    predictions["u10_effect"] = predictions["u10_norm"] * predictions["u10_coef"]
    predictions["v10_effect"] = predictions["v10_norm"] * predictions["v10_coef"]
    predictions["sst_effect"] = predictions["sst_norm"] * predictions["sst_coef"]
    predictions["swh_effect"] = predictions["swh_norm"] * predictions["swh_coef"]
    # Add 'noaa' and 'noaa_slr' columns
    predictions["noaa"] = predictions["Prediction"]
    predictions["noaa_slr"] = predictions["SLR_adj"]
    # Calculate my_prediction
    predictions["station_SLP_prediction"] = (
        predictions["noaa_slr"]
        + predictions["slp_effect"]
    )
    # Calculate my_prediction
    predictions[f"prediction_{member:02d}"] = (
        predictions["noaa_slr"]
        + predictions["slp_effect"]
        + predictions["departure_effect"]
        + predictions["u10_effect"]
        + predictions["v10_effect"]
        + predictions["sst_effect"]
        + predictions["swh_effect"]
    )
    pred = predictions[
        [
            "date",
            "days_ahead",
            "Observation",
            "noaa",
            "noaa_slr",
            f"prediction_{member:02d}",
        ]
    ]

    return pred 

def create_prediction(member):
    # Get forecast
    forecast_norm = create_forecast(member)
    pred_MHHW = build_from_datum(forecast_norm,station_fors)
    pred_NAVD = build_from_datum(forecast_norm,station_fors_NAVD)
    
    return pred_MHHW, pred_NAVD
    

#------------------------------------------------------------------------------
#                 GET MONTE CARLO
#------------------------------------------------------------------------------
# fix for naples_bay first
if site == "naples_bay":
  station_ID = "8725110"
else:
  station_ID = str(station_data['station_ID'])

coefs = pd.read_csv(COEF_DIR+station_ID+"_use_data.csv")

coefs = coefs.reset_index(drop=True)

#------------------------------------------------------------------------------
#
#                 PUT IT ALL TOGETHER
#
#------------------------------------------------------------------------------
os.chdir(FORECAST_DIR) # Make sure you set to the forecast_dir
station_ID = str(station_data['station_ID']) # fix this for naples bay

for member in range(1,members+1):
    pred_MHHW, pred_NAVD = create_prediction(member)
    if member == 1:
      ensemble_MHHW = pred_MHHW
      ensemble_NAVD = pred_NAVD
    else:
      pred_MHHW =  pred_MHHW[["date",f"prediction_{member:02d}"]]
      ensemble_MHHW = pd.merge(ensemble_MHHW, pred_MHHW, on='date', how='left')
      pred_NAVD =  pred_NAVD[["date",f"prediction_{member:02d}"]]
      ensemble_NAVD = pd.merge(ensemble_NAVD, pred_NAVD, on='date', how='left')

#------------------------------------------------------------------------------
#                 ANALYZE MHHW OUPUT
#------------------------------------------------------------------------------
# Step 1: Select columns containing "prediction"
prediction_columns_MHHW = ensemble_MHHW.filter(like='prediction')
# Step 2: Calculate the required statistics
ensemble_MHHW['mean_prediction'] = prediction_columns_MHHW.mean(axis=1)
ensemble_MHHW['percentile_10_prediction'] = prediction_columns_MHHW.apply(lambda x: np.percentile(x, 10), axis=1)
ensemble_MHHW['percentile_25_prediction'] = prediction_columns_MHHW.apply(lambda x: np.percentile(x, 25), axis=1)
ensemble_MHHW['percentile_75_prediction'] = prediction_columns_MHHW.apply(lambda x: np.percentile(x, 75), axis=1)
ensemble_MHHW['percentile_90_prediction'] = prediction_columns_MHHW.apply(lambda x: np.percentile(x, 90), axis=1)

# ensemble_MHHW.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_MHHW_all_members.csv", index=False)
ensemble_MHHW.to_csv(f"{OUTPUT_DIR}/{model}_{forecast_type}_all/{station_ID}_{site}/{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_MHHW_all_members.csv", index=False)

# Keep only the relevant columns
savefile = ensemble_MHHW[
    [
        "date",
        "days_ahead",
        "noaa",
        "noaa_slr",
        "mean_prediction",
        "percentile_10_prediction",
        "percentile_25_prediction",
        "percentile_75_prediction",
        "percentile_90_prediction"
    ]
] 
# savefile.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_MHHW_summary_daily.csv", index=False)
savefile.to_csv(f"{OUTPUT_DIR}/{model}_{forecast_type}_master/{station_ID}_{site}/{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_MHHW_summary_daily.csv", index=False)


#------------------------------------------------------------------------------
#                 NOW MAKE IT NAVD88
#------------------------------------------------------------------------------
# Step 1: Select columns containing "prediction"
prediction_columns = ensemble_NAVD.filter(like='prediction')
# Step 2: Calculate the required statistics
ensemble_NAVD['mean_prediction'] = prediction_columns.mean(axis=1)
ensemble_NAVD['percentile_10_prediction'] = prediction_columns.apply(lambda x: np.percentile(x, 10), axis=1)
ensemble_NAVD['percentile_25_prediction'] = prediction_columns.apply(lambda x: np.percentile(x, 25), axis=1)
ensemble_NAVD['percentile_75_prediction'] = prediction_columns.apply(lambda x: np.percentile(x, 75), axis=1)
ensemble_NAVD['percentile_90_prediction'] = prediction_columns.apply(lambda x: np.percentile(x, 90), axis=1)

# ensemble_NAVD.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_NAVD_all_members.csv", index=False)
ensemble_NAVD.to_csv(f"{OUTPUT_DIR}/{model}_{forecast_type}_all/{station_ID}_{site}/{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_NAVD_all_members.csv", index=False)

# Keep only the relevant columns
savefile = ensemble_NAVD[
    [
        "date",
        "days_ahead",
        "noaa",
        "noaa_slr",
        "mean_prediction",
        "percentile_10_prediction",
        "percentile_25_prediction",
        "percentile_75_prediction",
        "percentile_90_prediction"
    ]
] 
# savefile.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_NAVD_summary_daily.csv", index=False)
savefile.to_csv(f"{OUTPUT_DIR}/{model}_{forecast_type}_master/{station_ID}_{site}/{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_NAVD_summary_daily.csv", index=False)


#------------------------------------------------------------------------------
#
#                 PLOT IT
#
#------------------------------------------------------------------------------
# Read the logos
sfwmd_logo = mpimg.imread(LOGO_DIR+"sfwmd.png")
miami_logo = mpimg.imread(LOGO_DIR+"umiami.png")

# code to plot
fig, ax = plt.subplots(figsize=(12.5, 7.5))
ax2 = ax.twinx()

ax.set_position([0.1, 0.25, 0.8, 0.65]) # [left, bottom, width, height]

# Assuming the main ax takes up most of the figure, with small margins
# Define positions for logos underneath the main ax
logo_height = 0.12  # adjust if necessary
sfwmd_logo_width = logo_height * (sfwmd_logo.shape[1] / sfwmd_logo.shape[0])  # width according to sfwmd_logo dimensions
miami_logo_width = logo_height * (miami_logo.shape[1] / miami_logo.shape[0])  # width according to miami_logo dimensions
spacing = -0.035  # space between logos

# Positioning sfwmd_logo
ax_sfwmd = fig.add_axes([0.03, 0.03, sfwmd_logo_width, logo_height])
ax_sfwmd.imshow(sfwmd_logo)
ax_sfwmd.axis('off')

# Positioning miami_logo next to sfwmd_logo
ax_miami = fig.add_axes([0.0 + sfwmd_logo_width + spacing, 0.03, miami_logo_width, logo_height])
ax_miami.imshow(miami_logo)
ax_miami.axis('off')

# rest of code
ax.plot(ensemble_MHHW['date'], ensemble_MHHW['Observation'], marker='o', markersize=8, markerfacecolor='none', markeredgecolor='blue', label="Last Water Level Observation")
ax.plot(ensemble_MHHW['date'], ensemble_MHHW['noaa'], color = (0.8, 0, 0), label="NOAA Prediction", linewidth=2)
ax.plot(ensemble_MHHW['date'], ensemble_MHHW['noaa_slr'], color = 'orange', label="NOAA + SLR", linewidth=2)
ax.plot(ensemble_MHHW['date'], ensemble_MHHW['mean_prediction'], color='cyan', label="Ensemble Mean", linewidth=3)
ax.axhline(y=minor_flood, color='orange', linestyle='--', label="Minor Flooding Threshold", linewidth=1.5)  # This line was added
ax.axhline(y=moderate_flood, color='red', linestyle='--', label="Moderate Flooding Threshold", linewidth=1.5)  # This line was added
ax.axhline(y=major_flood, color='purple', linestyle='--', label="Major Flooding Threshold", linewidth=1.5)  # This line was added
# Adding fill between the percentile lines
ax.fill_between(ensemble_MHHW['date'], ensemble_MHHW['percentile_10_prediction'], ensemble_MHHW['percentile_90_prediction'], color='lightgray', alpha=0.5)  # Fills between 10th and 90th percentile
ax.fill_between(ensemble_MHHW['date'], ensemble_MHHW['percentile_25_prediction'], ensemble_MHHW['percentile_75_prediction'], color='darkgray', alpha=0.5)  # Fills between 25th and 75th percentile

# Extract y-values from the specified lines
y_values_to_consider = [
    ensemble_MHHW['Observation'],
    ensemble_MHHW['noaa'],
    ensemble_MHHW['noaa_slr'],
    ensemble_MHHW['mean_prediction'],
    ensemble_MHHW['percentile_10_prediction'],
    ensemble_MHHW['percentile_90_prediction'],
]

# Add the constant value for "Minor Flooding Threshold"
y_values_to_consider.append([minor_flood] * len(ensemble_MHHW['date']))

# Determine the minimum and maximum y-values
y_min = min([np.nanmin(y) for y in y_values_to_consider]) - 0.01
y_max = max([np.nanmax(y) for y in y_values_to_consider]) + 0.01

y_min = math.floor(y_min * 5) / 5  # Round down to nearest 0.5
y_max = math.ceil(y_max * 5) / 5  # Round up to nearest 0.5

# Set the y-axis limits
ax.set_ylim(y_min, y_max)
ax.set_yticks(np.arange(y_min, y_max + 0.2, 0.2))

# Set the labels
ax.set_xlabel("Date")
ax.set_ylabel("Daily Highest Water Level (ft. above MHHW)")
ax2.set_ylabel(f"Water Level (ft. above NAVD88) \n Add {ngvd_offset} ft. for NGVD29")
ax.set_title(f"{station_name} Maximum Daily Tide Predictions for {ensemble_MHHW['date'].iloc[1].strftime('%m-%d-%y')}\nGFS Ensemble Forecast")

# set twin scale (convert degree celsius to fahrenheit)
navd = lambda mhhw: mhhw + navd_offset
# get left axis limits
ymin, ymax = ax.get_ylim()
# apply function and set transformed values to right axis limits
ax2.set_ylim((navd(ymin),navd(ymax)))
# set an invisible artist to twin axes
# Set y-ticks on the second y-axis based on those of the first y-axis
ax2.set_yticks(navd(np.array(ax.get_yticks())))
# Hide the top and bottom tick labels on the second y-axis
tick1_labels = ax.get_yticklabels()
if tick1_labels:
    tick1_labels[0].set_visible(False)
    tick1_labels[-1].set_visible(False)
tick2_labels = ax2.get_yticklabels()
if tick2_labels:
    tick2_labels[0].set_visible(False)
    tick2_labels[-1].set_visible(False)
# to prevent falling back to initial values on rescale events
ax2.plot([],[])

# Format x-axis to display only month and day
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Create patches for the filled areas
patch_10_90 = mpatches.Patch(color='lightgray', alpha=0.5, label='10th/90th Percentile')
patch_25_75 = mpatches.Patch(color='darkgray', alpha=0.5, label='25th/75th Percentile')

# Get the existing legend handles and labels from your plot
handles, labels = ax.get_legend_handles_labels()

# Turn them into a dictionary for easy lookups: {label: handle}
legend_dict = dict(zip(labels, handles))

# Build the final ordered lists of handles and labels:
#  -- Column 1: Last Water Level Observation, NOAA Prediction, NOAA + SLR
#  -- Column 2: Ensemble Mean, 10/90 Percentile, 25/75 Percentile
#  -- Column 3: Minor Flooding Threshold, Moderate Flooding Threshold, Major Flooding Threshold

handles_ordered = [
    legend_dict["Last Water Level Observation"],
    legend_dict["NOAA Prediction"],
    legend_dict["NOAA + SLR"],
    legend_dict["Ensemble Mean"],  # Make sure your code has a line labeled "Ensemble Mean"
    patch_10_90,
    patch_25_75,
    legend_dict["Minor Flooding Threshold"],
    legend_dict["Moderate Flooding Threshold"],
    legend_dict["Major Flooding Threshold"]
]

labels_ordered = [
    "Last Water Level Observation",
    "NOAA Prediction",
    "NOAA + SLR",
    "Ensemble Mean",
    "10th/90th Percentile",
    "25th/75th Percentile",
    "Minor Flooding Threshold",
    "Moderate Flooding Threshold",
    "Major Flooding Threshold"
]

# Create the legend with 3 columns
ax.legend(handles_ordered, labels_ordered,
          loc='upper right', bbox_to_anchor=(1, -0.15), ncol=3)

ax.set_facecolor('white')

# Add gridlines
ax.grid(True)

# Slant dates on x-axis diagonally upward to the right
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Save the graph as a PNG file
figure_name = f"{OUTPUT_DIR}/{model}_{forecast_type}_graphs/{station_ID}_{site}/{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_DAILY.png"
plt.savefig(figure_name, format='png', bbox_inches='tight', dpi=300)

# Save the latest graph as PNG
figure_name = f"{OUTPUT_DIR}/{model}_{forecast_type}_graphs/{station_ID}_{site}/latest_{station_abbr}_{model}_{forecast_type}_DAILY.png"
plt.savefig(figure_name, format='png', bbox_inches='tight', dpi=300)

# plt.show()


#------------------------------------------------------------------------------
#
#                 BUILD 6 MINUTE PREDICTIONS MHHW
#
#------------------------------------------------------------------------------
pred = ensemble_MHHW

noaa_preds = station_pred.copy()
noaa_preds = noaa_preds.rename(columns={"Date Time": "datetime", " Prediction": "noaa"})
noaa_preds['date'] = noaa_preds['datetime'].dt.date
noaa_preds['date'] = pd.to_datetime(noaa_preds['date'])
noaa_preds = noaa_preds[['date'] + [col for col in noaa_preds.columns if col != 'date']]
# noaa_preds = noaa_preds.iloc[240:]
# noaa_preds.reset_index(drop=True, inplace=True)
pred.at[0, 'mean_prediction'] = pred['Observation'].iloc[0]
pred.at[0, 'percentile_10_prediction'] = pred['Observation'].iloc[0]
pred.at[0, 'percentile_25_prediction'] = pred['Observation'].iloc[0]
pred.at[0, 'percentile_75_prediction'] = pred['Observation'].iloc[0]
pred.at[0, 'percentile_90_prediction'] = pred['Observation'].iloc[0]

# Merge the two dataframes on date, find offsets and run interpolation
merged_df = pd.merge(noaa_preds, pred, on=['date', 'noaa'], how='left')

# THIS LINE FOR MAKING THE ADJUSTMENT
merged_df.loc[240, 'noaa_slr'] = merged_df.loc[240, 'noaa'] + last_slr
merged_df.loc[240, 'mean_prediction'] = merged_df.loc[240, 'noaa_slr'] + last_dep

merged_df["noaa_slr_offset"] = merged_df['noaa_slr'] - merged_df['noaa']
merged_df["noaa_mean_offset"] = merged_df['mean_prediction'] - merged_df['noaa']
merged_df["noaa_10_offset"] = merged_df['percentile_10_prediction'] - merged_df['noaa']
merged_df["noaa_25_offset"] = merged_df['percentile_25_prediction'] - merged_df['noaa']
merged_df["noaa_75_offset"] = merged_df['percentile_75_prediction'] - merged_df['noaa']
merged_df["noaa_90_offset"] = merged_df['percentile_90_prediction'] - merged_df['noaa']
merged_df[['noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset']] = merged_df[['noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset']].interpolate(method='linear')
merged_df[['noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset']] = merged_df[['noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset']].fillna(method='bfill')

# Convert original columns to interpolated values
merged_df['noaa_slr'] = merged_df['noaa'] + merged_df['noaa_slr_offset']
merged_df['mean_prediction'] = merged_df['noaa'] + merged_df['noaa_mean_offset']
merged_df['percentile_10_prediction'] = merged_df['noaa'] + merged_df['noaa_10_offset']
merged_df['percentile_25_prediction'] = merged_df['noaa'] + merged_df['noaa_25_offset']
merged_df['percentile_75_prediction'] = merged_df['noaa'] + merged_df['noaa_75_offset']
merged_df['percentile_90_prediction'] = merged_df['noaa'] + merged_df['noaa_90_offset']

# drop columns
merged_df = merged_df.drop(columns=['days_ahead', 'noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset','date'])

# Combine and merge with tide observations
tide_obs = station_tide[["Date Time", " Water Level"]].rename(columns={"Date Time": "datetime", " Water Level": "observations"})
plot_data = pd.merge(tide_obs, merged_df, on=['datetime'], how='right')
plot_data.loc[:239, 'noaa_slr'] = np.nan
plot_data.loc[:239, 'mean_prediction'] = np.nan
plot_data.loc[:239, 'percentile_10_prediction'] = np.nan
plot_data.loc[:239, 'percentile_25_prediction'] = np.nan
plot_data.loc[:239, 'percentile_75_prediction'] = np.nan
plot_data.loc[:239, 'percentile_90_prediction'] = np.nan
# Keep only the relevant columns
plot_data = plot_data[
    [
        "datetime",
        "observations",
        "noaa",
        "noaa_slr",
        "mean_prediction",
        "percentile_10_prediction",
        "percentile_25_prediction",
        "percentile_75_prediction",
        "percentile_90_prediction"
    ]
] 
# plot_data.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_MHHW_summary_6min.csv", index=False)
plot_data.to_csv(f"{OUTPUT_DIR}/{model}_{forecast_type}_master/{station_ID}_{site}/{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_MHHW_summary_6min.csv", index=False)

#------------------------------------------------------------------------------
#
#                 BUILD 6 MINUTE PREDICTIONS NAVD
#
#------------------------------------------------------------------------------
pred = ensemble_NAVD

noaa_preds = station_pred_NAVD.copy()
noaa_preds = noaa_preds.rename(columns={"Date Time": "datetime", " Prediction": "noaa"})
noaa_preds['date'] = noaa_preds['datetime'].dt.date
noaa_preds['date'] = pd.to_datetime(noaa_preds['date'])
noaa_preds = noaa_preds[['date'] + [col for col in noaa_preds.columns if col != 'date']]
# noaa_preds = noaa_preds.iloc[240:]
# noaa_preds.reset_index(drop=True, inplace=True)
pred.at[0, 'mean_prediction'] = pred['Observation'].iloc[0]
pred.at[0, 'percentile_10_prediction'] = pred['Observation'].iloc[0]
pred.at[0, 'percentile_25_prediction'] = pred['Observation'].iloc[0]
pred.at[0, 'percentile_75_prediction'] = pred['Observation'].iloc[0]
pred.at[0, 'percentile_90_prediction'] = pred['Observation'].iloc[0]

# Merge the two dataframes on date, find offsets and run interpolation
merged_df = pd.merge(noaa_preds, pred, on=['date', 'noaa'], how='left')

# THIS LINE FOR MAKING THE ADJUSTMENT
merged_df.loc[240, 'noaa_slr'] = merged_df.loc[240, 'noaa'] + last_slr
merged_df.loc[240, 'mean_prediction'] = merged_df.loc[240, 'noaa_slr'] + last_dep

merged_df["noaa_slr_offset"] = merged_df['noaa_slr'] - merged_df['noaa']
merged_df["noaa_mean_offset"] = merged_df['mean_prediction'] - merged_df['noaa']
merged_df["noaa_10_offset"] = merged_df['percentile_10_prediction'] - merged_df['noaa']
merged_df["noaa_25_offset"] = merged_df['percentile_25_prediction'] - merged_df['noaa']
merged_df["noaa_75_offset"] = merged_df['percentile_75_prediction'] - merged_df['noaa']
merged_df["noaa_90_offset"] = merged_df['percentile_90_prediction'] - merged_df['noaa']
merged_df[['noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset']] = merged_df[['noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset']].interpolate(method='linear')
merged_df[['noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset']] = merged_df[['noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset']].fillna(method='bfill')

# Convert original columns to interpolated values
merged_df['noaa_slr'] = merged_df['noaa'] + merged_df['noaa_slr_offset']
merged_df['mean_prediction'] = merged_df['noaa'] + merged_df['noaa_mean_offset']
merged_df['percentile_10_prediction'] = merged_df['noaa'] + merged_df['noaa_10_offset']
merged_df['percentile_25_prediction'] = merged_df['noaa'] + merged_df['noaa_25_offset']
merged_df['percentile_75_prediction'] = merged_df['noaa'] + merged_df['noaa_75_offset']
merged_df['percentile_90_prediction'] = merged_df['noaa'] + merged_df['noaa_90_offset']

# drop columns
merged_df = merged_df.drop(columns=['days_ahead', 'noaa_slr_offset', 'noaa_mean_offset', 'noaa_10_offset', 'noaa_25_offset', 'noaa_75_offset', 'noaa_90_offset','date'])

# Combine and merge with tide observations
tide_obs = station_tide[["Date Time", " Water Level"]].rename(columns={"Date Time": "datetime", " Water Level": "observations"})
plot_data_NAVD = pd.merge(tide_obs, merged_df, on=['datetime'], how='right')
plot_data_NAVD.loc[:239, 'noaa_slr'] = np.nan
plot_data_NAVD.loc[:239, 'mean_prediction'] = np.nan
plot_data_NAVD.loc[:239, 'percentile_10_prediction'] = np.nan
plot_data_NAVD.loc[:239, 'percentile_25_prediction'] = np.nan
plot_data_NAVD.loc[:239, 'percentile_75_prediction'] = np.nan
plot_data_NAVD.loc[:239, 'percentile_90_prediction'] = np.nan
# Keep only the relevant columns
plot_data_NAVD = plot_data_NAVD[
    [
        "datetime",
        "observations",
        "noaa",
        "noaa_slr",
        "mean_prediction",
        "percentile_10_prediction",
        "percentile_25_prediction",
        "percentile_75_prediction",
        "percentile_90_prediction"
    ]
] 
plot_data_NAVD.to_csv(f"{OUTPUT_DIR}/{model}_{forecast_type}_master/{station_ID}_{site}/{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_NAVD_summary_6min.csv", index=False)

#------------------------------------------------------------------------------
#
#                 PLOT IT AGAIN!
#
#------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 7.5))
ax2 = ax.twinx()

ax.set_position([0.1, 0.25, 0.8, 0.65])  # [left, bottom, width, height]

# Assuming the main ax takes up most of the figure, with small margins
# Define positions for logos underneath the main ax
logo_height = 0.12  # adjust if necessary
sfwmd_logo_width = logo_height * (sfwmd_logo.shape[1] / sfwmd_logo.shape[0])  # width according to sfwmd_logo dimensions
miami_logo_width = logo_height * (miami_logo.shape[1] / miami_logo.shape[0])  # width according to miami_logo dimensions
spacing = -0.035  # space between logos

# Positioning sfwmd_logo
ax_sfwmd = fig.add_axes([0.03, 0.03, sfwmd_logo_width, logo_height])
ax_sfwmd.imshow(sfwmd_logo)
ax_sfwmd.axis('off')

# Positioning miami_logo next to sfwmd_logo
ax_miami = fig.add_axes([0.0 + sfwmd_logo_width + spacing, 0.03, miami_logo_width, logo_height])
ax_miami.imshow(miami_logo)
ax_miami.axis('off')

ax.plot(plot_data['datetime'], plot_data['observations'], color='b', label="Observations", linewidth=1.0)
ax.plot(plot_data['datetime'], plot_data['noaa'], color=(0.8, 0, 0), label="NOAA Prediction", linewidth=0.8)
ax.plot(plot_data['datetime'], plot_data['noaa_slr'], color='orange', label="NOAA + SLR", linewidth=0.8)
ax.plot(plot_data['datetime'], plot_data['mean_prediction'], color='cyan', label="Ensemble Mean", linewidth=1.2)
ax.axhline(y=minor_flood, color='orange', linestyle='--', label="Minor Flooding Threshold", linewidth=1.5)  # This line was added
ax.axhline(y=moderate_flood, color='red', linestyle='--', label="Moderate Flooding Threshold", linewidth=1.5)  # This line was added
ax.axhline(y=major_flood, color='purple', linestyle='--', label="Major Flooding Threshold", linewidth=1.5)  # This line was added
# Adding fill between the percentile lines
ax.fill_between(plot_data['datetime'], plot_data['percentile_10_prediction'], plot_data['percentile_90_prediction'], color='lightgray', alpha=0.5)  # Fills between 10th and 90th percentile
ax.fill_between(plot_data['datetime'], plot_data['percentile_25_prediction'], plot_data['percentile_75_prediction'], color='darkgray', alpha=0.5)  # Fills between 25th and 75th percentile


# Extract y-values from the specified lines
y_values_to_consider = [
    plot_data['observations'],
    plot_data['noaa'],
    plot_data['noaa_slr'],
    plot_data['mean_prediction'],
    plot_data['percentile_10_prediction'],
    plot_data['percentile_90_prediction'],
]

# Add the constant value for "Minor Flooding Threshold"
y_values_to_consider.append([minor_flood] * len(plot_data['datetime']))

# Determine the minimum and maximum y-values
y_min = min([np.nanmin(y) for y in y_values_to_consider]) - 0.01
y_max = max([np.nanmax(y) for y in y_values_to_consider]) + 0.01

y_min = math.floor(y_min * 2) / 2  # Round down to nearest 0.5
y_max = math.ceil(y_max * 2) / 2  # Round up to nearest 0.5

ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))

# Set the y-axis limits
ax.set_ylim(y_min, y_max)

# Set the labels
ax.set_xlabel("Date")
ax.set_ylabel("Water Level (ft. above MHHW)")
ax2.set_ylabel(f"Water Level (ft. above NAVD88) \n Add {ngvd_offset} ft. for NGVD29")
ax.set_title(f"{station_name} 6-Minute Tide Predictions for {ensemble_MHHW['date'].iloc[1].strftime('%m-%d-%y')}\nGFS Ensemble Forecast")

# set twin scale (convert degree celsius to fahrenheit)
navd = lambda mhhw: mhhw + navd_offset
# get left axis limits
ymin, ymax = ax.get_ylim()
# apply function and set transformed values to right axis limits
ax2.set_ylim((navd(ymin),navd(ymax)))
# set an invisible artist to twin axes
# Set y-ticks on the second y-axis based on those of the first y-axis
ax2.set_yticks(navd(np.array(ax.get_yticks())))
# Hide the top and bottom tick labels on the second y-axis
tick1_labels = ax.get_yticklabels()
if tick1_labels:
    tick1_labels[0].set_visible(False)
    tick1_labels[-1].set_visible(False)
tick2_labels = ax2.get_yticklabels()
if tick2_labels:
    tick2_labels[0].set_visible(False)
    tick2_labels[-1].set_visible(False)
# to prevent falling back to initial values on rescale events
ax2.plot([],[])


# Format x-axis to display only month and day
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Create patches for the filled areas
patch_10_90 = mpatches.Patch(color='lightgray', alpha=0.5, label='10th/90th Percentile')
patch_25_75 = mpatches.Patch(color='darkgray', alpha=0.5, label='25th/75th Percentile')

# Get the existing legend handles and labels from your plot
handles, labels = ax.get_legend_handles_labels()

# Turn them into a dictionary for easy lookups: {label: handle}
legend_dict = dict(zip(labels, handles))

# Build the final ordered lists of handles and labels:
#  -- Column 1: Last Water Level Observation, NOAA Prediction, NOAA + SLR
#  -- Column 2: Ensemble Mean, 10/90 Percentile, 25/75 Percentile
#  -- Column 3: Minor Flooding Threshold, Moderate Flooding Threshold, Major Flooding Threshold

handles_ordered = [
    legend_dict["Observations"],
    legend_dict["NOAA Prediction"],
    legend_dict["NOAA + SLR"],
    legend_dict["Ensemble Mean"],  # Make sure your code has a line labeled "Ensemble Mean"
    patch_10_90,
    patch_25_75,
    legend_dict["Minor Flooding Threshold"],
    legend_dict["Moderate Flooding Threshold"],
    legend_dict["Major Flooding Threshold"]
]

labels_ordered = [
    "Observations",
    "NOAA Prediction",
    "NOAA + SLR",
    "Ensemble Mean",
    "10th/90th Percentile",
    "25th/75th Percentile",
    "Minor Flooding Threshold",
    "Moderate Flooding Threshold",
    "Major Flooding Threshold"
]

# Create the legend with 3 columns
ax.legend(handles_ordered, labels_ordered,
          loc='upper right', bbox_to_anchor=(1, -0.15), ncol=3)

ax.set_facecolor('white')

# Add gridlines
ax.grid(True)

# Slant dates on x-axis diagonally upward to the right
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Make space for the legend below the plot
# fig.subplots_adjust(bottom=0.25)  # adjust this value if needed

# Save the graph as a PNG file
figure_name = f"{OUTPUT_DIR}/{model}_{forecast_type}_graphs/{station_ID}_{site}/{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_6MIN.png"
plt.savefig(figure_name, format='png', bbox_inches='tight', dpi=300)

# Save the latest graph as PNG
figure_name = f"{OUTPUT_DIR}/{model}_{forecast_type}_graphs/{station_ID}_{site}/latest_{station_abbr}_{model}_{forecast_type}_6MIN.png"
plt.savefig(figure_name, format='png', bbox_inches='tight', dpi=300)
 
# plt.show()
