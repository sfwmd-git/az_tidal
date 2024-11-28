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
#                 SET THE SITE
#
#------------------------------------------------------------------------------

#site = 'fort_myers'  # Change this as needed for different sites
# Argument parsing
parser = argparse.ArgumentParser(description="Run the script for a specific site")
parser.add_argument('--site', required=True, help='The site to run the script for (e.g., fort_myers)')
args = parser.parse_args()

# Use the provided site argument
site = args.site

#------------------------------------------------------------------------------
#
#                 SET DIRECTORIES
#
#------------------------------------------------------------------------------

# Define the base directory based on the hostname
hostname = platform.node()
if hostname == 'Nates-MacBook-Pro.local' or hostname == 'UM-C02RQ1S8FVH7' or hostname == 'nates-mbp.lan':
    BASE_DIR = '/Users/nate/Documents/SFWMD_work/'
elif hostname == 'mango.rsmas.miami.edu':
    BASE_DIR = '/home/ntaminger/SFWMD_work/'
elif hostname == 'aztidal01p':
    BASE_DIR = '/home/ntaminge/SFWMD_work/'
else:
    raise ValueError(f"Unknown hostname: {hostname}")

regression_subdir = f'regression_work/{site}/regression_data/'
predictions_subdir = f'predictions/{site}/'

# Define other directories relative to the base directory and site-specific subdirectories
DATA_DIR = os.path.join(BASE_DIR, regression_subdir)
OUTPUT_DIR = os.path.join(BASE_DIR, predictions_subdir)
LOGO_DIR = os.path.join(BASE_DIR, 'logos/')
pwd = os.path.join(BASE_DIR, 'predictions/')
os.chdir(pwd)

#------------------------------------------------------------------------------
#
#                 LOAD STATION DATA FROM CSV
#
#------------------------------------------------------------------------------

# Load station-specific data from CSV
csv_file = os.path.join(BASE_DIR, 'metadata', 'stationdata.csv')
df = pd.read_csv(csv_file)

# Filter the row for the specified site
station_data = df[df['site_name'] == site].iloc[0]

#------------------------------------------------------------------------------
#
#                 ASSIGN VARIABLES MANUALLY
#
#------------------------------------------------------------------------------
model = "ECMWF"
forecast_type = "ENS"
sleep_time = 4  # If this remains constant
db = 0  # If this remains constant
members = 50  # If this remains constant

# forecast GRIB Dims.
forecast_nlat_idx = station_data['ecmwf_ens_forecast_nlat_idx']
forecast_slat_idx = station_data['ecmwf_ens_forecast_slat_idx']
forecast_wlon_idx = station_data['ecmwf_ens_forecast_wlon_idx']
forecast_elon_idx = station_data['ecmwf_ens_forecast_elon_idx']

# wave GRIB Dims.
wave_nlat_idx = station_data['ecmwf_ens_wave_nlat_idx']
wave_slat_idx = station_data['ecmwf_ens_wave_slat_idx']
wave_wlon_idx = station_data['ecmwf_ens_wave_wlon_idx']
wave_elon_idx = station_data['ecmwf_ens_wave_elon_idx']

#------------------------------------------------------------------------------
#
#                 ASSIGN VARIABLES FROM CSV
#
#------------------------------------------------------------------------------

DATUM = "MHHW"  # If this remains constant
station_ID = str(station_data['station_ID'])
station_abbr = station_data['station_abbr']
station_name = station_data['station_name']
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
#                 GET TIDE DATA TOGETHER
#
#------------------------------------------------------------------------------
# Set working directory
os.chdir(DATA_DIR)

# Read SLR slope data
station_slr = pd.read_csv(station_ID+"_slr.csv")
first_date = pd.to_datetime(station_slr.iloc[0]["date"])
v_trend = station_slr.iloc[0]["v_trend"]
slope = station_slr.iloc[0]["slope"]

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
station_maxs["SLR"] = [(d - first_date).days * slope + v_trend for d in station_maxs["Date"]]
station_maxs["SLR_adj"] = station_maxs["Prediction"] + station_maxs["SLR"]

# Build forecast dataframe
station_fors = pd.merge(station_maxs, station_wobs, on="Date", how="outer")
station_fors["Departure"] = station_fors["Observation"] - station_fors["SLR_adj"]
station_fors["day"] = range(11)
station_fors_MHHW = station_fors
print(station_fors_MHHW)

#------------------------------------------------------------------------------
#                 GET TIDE DATA TOGETHER - NAVD
#------------------------------------------------------------------------------
# Set working directory
os.chdir(DATA_DIR)

# Read SLR slope data
station_slr_NAVD = pd.read_csv(station_ID+"_slr.csv")
first_date_NAVD = pd.to_datetime(station_slr_NAVD.iloc[0]["date"])
v_trend_NAVD = station_slr_NAVD.iloc[0]["v_trend"]
slope_NAVD = station_slr_NAVD.iloc[0]["slope"]

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
station_maxs_NAVD["SLR"] = [(d - first_date_NAVD).days * slope_NAVD + v_trend_NAVD for d in station_maxs_NAVD["Date"]]
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
def get_average_slp(file, member):
    with pygrib.open(file) as gr:
        for msg in gr:
            if (msg.name == 'Mean sea level pressure' and
                msg.perturbationNumber == member):
                vals = msg.values
                avg = np.mean(vals[forecast_nlat_idx:forecast_slat_idx+1, forecast_wlon_idx:forecast_elon_idx+1])  # Calculate the average over the specified slice
                return avg
    return None

def get_average_sst(file, member):
    with pygrib.open(file) as gr:
        for msg in gr:
            if (msg.name == 'Sea surface temperature' and
                msg.perturbationNumber == member):
                vals = msg.values
                avg = np.mean(vals[forecast_nlat_idx:forecast_slat_idx+1, forecast_wlon_idx:forecast_elon_idx+1])  # Calculate the average over the specified slice
                return avg
    return None
  
def get_average_u10(file, member):
    with pygrib.open(file) as gr:
        for msg in gr:
            if (msg.name == '10 metre U wind component' and
                msg.perturbationNumber == member):
                vals = msg.values
                avg = np.mean(vals[forecast_nlat_idx:forecast_slat_idx+1, forecast_wlon_idx:forecast_elon_idx+1])  # Calculate the average over the specified slice
                return avg
    return None
  
def get_average_v10(file, member):
    with pygrib.open(file) as gr:
        for msg in gr:
            if (msg.name == '10 metre V wind component' and
                msg.perturbationNumber == member):
                vals = msg.values
                avg = np.mean(vals[forecast_nlat_idx:forecast_slat_idx+1, forecast_wlon_idx:forecast_elon_idx+1])  # Calculate the average over the specified slice
                return avg
    return None
  
def get_average_swh(file, member):
    with pygrib.open(file) as gr:
        for msg in gr:
            if (msg.name == 'Significant height of combined wind waves and swell' and
                msg.perturbationNumber == member):
                vals = msg.values
                avg = np.mean(vals[wave_nlat_idx:wave_slat_idx+1, wave_wlon_idx:wave_elon_idx+1])  # Calculate the average over the specified slice
                return avg
    return None

def generate_forecast_file_name(creation_date, forecast_hour_difference):
    # Formatting the creation date as Month (MM) Day (DD) Hour (HH) Minute (MM)
    formatted_creation_date = creation_date.strftime("%m%d%H%M")
    # Calculate the target date by adding the forecast_hour_difference to the creation date
    target_date = creation_date + timedelta(hours=forecast_hour_difference)
    # Determine if creation date is equal to target date (excluding minutes for comparison)
    if creation_date.strftime("%m%d%H") == target_date.strftime("%m%d%H"):
        # Adjust the minute in the target date format to "01" if the dates are equal
        formatted_target_date = target_date.strftime("%m%d%H") + "00"
    else:
        # Format the target date: Month (MM) Day (DD) Hour (HH) Minute (MM)
        formatted_target_date = target_date.strftime("%m%d%H%M")
    # Combine the formatted dates according to the specified structure
    return f"A1E{formatted_creation_date}{formatted_target_date}1"


def process_forecast_grib_files(member):
    data = []

    today = datetime.now() - timedelta(days=0)
    two_days_ago = today - timedelta(days=2)
    one_day_ago = today - timedelta(days=1)

    utc = pytz.utc
    est = pytz.timezone('US/Eastern')

    forecasts = [
        # From two days ago
        (two_days_ago.replace(hour=0, minute=0, second=0, microsecond=0), 6),
        (two_days_ago.replace(hour=12, minute=0, second=0, microsecond=0), 0),
        (two_days_ago.replace(hour=12, minute=0, second=0, microsecond=0), 6),
        # From one day ago
        (one_day_ago.replace(hour=0, minute=0, second=0, microsecond=0), 0),
        (one_day_ago.replace(hour=0, minute=0, second=0, microsecond=0), 6),
        (one_day_ago.replace(hour=12, minute=0, second=0, microsecond=0), 0),
        (one_day_ago.replace(hour=12, minute=0, second=0, microsecond=0), 6),
        # From today
        (today.replace(hour=0, minute=0, second=0, microsecond=0), 0),
        (today.replace(hour=0, minute=0, second=0, microsecond=0), 6),
        (today.replace(hour=0, minute=0, second=0, microsecond=0), 12),
        (today.replace(hour=0, minute=0, second=0, microsecond=0), 18),
    ]
    # Adding forecasts for the next ten days, from today at 00Z for 00, 06, 12, & 18 hours
    for day in range(1, 10): # Start from 1 to include the next day up to ten days ahead
        target_day = today + timedelta(days=day)
        for hour in [0, 6, 12, 18]:
            forecasts.append((today.replace(hour=0, minute=0, second=0, microsecond=0), hour + day * 24))
    # forecasts.append((today.replace(hour=0, minute = 0, second = 0, microsecond = 0), 0 + 10 * 24))
    print(forecasts)
    # Generate and print file names for all forecasts
    for creation_date, forecast_hour_difference in forecasts:
        file_name = generate_forecast_file_name(creation_date, forecast_hour_difference)
        print(file_name)
        try:
            avg_slp = get_average_slp(file_name, member)
            avg_sst = get_average_sst(file_name, member)
            avg_u10 = get_average_u10(file_name, member)
            avg_v10 = get_average_v10(file_name, member)
            forecast_time = creation_date + timedelta(hours=forecast_hour_difference)
            datetime_utc = utc.localize(forecast_time)
            datetime_est = datetime_utc.astimezone(est)
            data.append({'datetime_est': datetime_est, 'average_slp': avg_slp, 'average_sst': avg_sst, 'average_u10': avg_u10, 'average_v10': avg_v10})
        except FileNotFoundError:
            print(f"File {file_name} not found")

    return pd.DataFrame(data, columns=['datetime_est', 'average_slp', 'average_sst', 'average_u10', 'average_v10'])

def generate_wave_file_name(creation_date, forecast_hour_difference):
    # Formatting the creation date as Month (MM) Day (DD) Hour (HH) Minute (MM)
    formatted_creation_date = creation_date.strftime("%m%d%H%M")
    # Calculate the target date by adding the forecast_hour_difference to the creation date
    target_date = creation_date + timedelta(hours=forecast_hour_difference)
    # Determine if creation date is equal to target date (excluding minutes for comparison)
    if creation_date.strftime("%m%d%H") == target_date.strftime("%m%d%H"):
        # Adjust the minute in the target date format to "01" if the dates are equal
        formatted_target_date = target_date.strftime("%m%d%H") + "00"
    else:
        # Format the target date: Month (MM) Day (DD) Hour (HH) Minute (MM)
        formatted_target_date = target_date.strftime("%m%d%H%M")
    # Combine the formatted dates according to the specified structure
    return f"A1W{formatted_creation_date}{formatted_target_date}1"


def process_wave_grib_files(member):
    data = []

    today = datetime.now() - timedelta(days=0)
    two_days_ago = today - timedelta(days=2)
    one_day_ago = today - timedelta(days=1)

    utc = pytz.utc
    est = pytz.timezone('US/Eastern')

    forecasts = [
        # From two days ago
        (two_days_ago.replace(hour=0, minute=0, second=0, microsecond=0), 6),
        (two_days_ago.replace(hour=12, minute=0, second=0, microsecond=0), 0),
        (two_days_ago.replace(hour=12, minute=0, second=0, microsecond=0), 6),
        # From one day ago
        (one_day_ago.replace(hour=0, minute=0, second=0, microsecond=0), 0),
        (one_day_ago.replace(hour=0, minute=0, second=0, microsecond=0), 6),
        (one_day_ago.replace(hour=12, minute=0, second=0, microsecond=0), 0),
        (one_day_ago.replace(hour=12, minute=0, second=0, microsecond=0), 6),
        # From today
        (today.replace(hour=0, minute=0, second=0, microsecond=0), 0),
        (today.replace(hour=0, minute=0, second=0, microsecond=0), 6),
        (today.replace(hour=0, minute=0, second=0, microsecond=0), 12),
        (today.replace(hour=0, minute=0, second=0, microsecond=0), 18),
    ]
    # Adding forecasts for the next ten days, from today at 00Z for 00, 06, 12, & 18 hours
    for day in range(1, 10): # Start from 1 to include the next day up to ten days ahead
        target_day = today + timedelta(days=day)
        for hour in [0, 6, 12, 18]:
            forecasts.append((today.replace(hour=0, minute=0, second=0, microsecond=0), hour + day * 24))
    forecasts.append((today.replace(hour=0, minute = 0, second = 0, microsecond = 0), 0 + 10 * 24))
    print(forecasts)
    # Generate and print file names for all forecasts
    for creation_date, forecast_hour_difference in forecasts:
        file_name = generate_wave_file_name(creation_date, forecast_hour_difference)
        print(file_name)
        try:
            avg_swh = get_average_swh(file_name, member)
            forecast_time = creation_date + timedelta(hours=forecast_hour_difference)
            datetime_utc = utc.localize(forecast_time)
            datetime_est = datetime_utc.astimezone(est)
            data.append({'datetime_est': datetime_est, 'average_swh': avg_swh})
        except FileNotFoundError:
            print(f"File {file_name} not found")

    return pd.DataFrame(data, columns=['datetime_est', 'average_swh'])
  

#------------------------------------------------------------------------------
#                 FUNCTIONS TO PUT IT ALL TOGETHER AND CREATE THE FORECAST
#------------------------------------------------------------------------------
def create_forecast(member):
    os.chdir(pwd)
    #------------------------------------------------------------------------------
    #                 GET AND CALC GEFS
    #------------------------------------------------------------------------------
    os.chdir(pwd)
    gfs = process_forecast_grib_files(member)
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
    os.chdir(pwd)
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
    
    # Merge the DataFrames on the 'date' column
    forecast = pd.merge(gfs_reset, swh_reset, on='date')
    # Rename the 'average_swh' column to 'swh'
    forecast = forecast.rename(columns={'average_swh': 'swh', 'average_slp': 'prmslmsl', 'average_sst': 'tmpsfc', 'average_u10': 'ugrd10m', 'average_v10': 'vgrd10m'})
    
    forecast1 = forecast
    
    #------------------------------------------------------------------------------
    #                 GET NORMALIZATIONS
    #------------------------------------------------------------------------------
    norms = pd.read_csv(DATA_DIR+station_ID+"_normalizations.csv")
    
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
    pred_MHHW = build_from_datum(forecast_norm,station_fors_MHHW)
    pred_NAVD = build_from_datum(forecast_norm,station_fors_NAVD)
    
    return pred_MHHW, pred_NAVD
    

#------------------------------------------------------------------------------
#                 GET MONTE CARLO
#------------------------------------------------------------------------------
coefs = pd.read_csv(DATA_DIR+station_abbr+"_mean_monte_carlo_all_day_prediction_model_USETHIS.csv")

coefs = coefs.reset_index(drop=True)

#------------------------------------------------------------------------------
#
#                 PUT IT ALL TOGETHER
#
#------------------------------------------------------------------------------
for member in range(1,members+1):
    print(member)
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

ensemble_MHHW.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_MHHW_all_members.csv", index=False)

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
savefile.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_MHHW_summary_daily.csv", index=False)


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

ensemble_NAVD.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_NAVD_all_members.csv", index=False)

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
savefile.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_NAVD_summary_daily.csv", index=False)


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
spacing = 0.00  # space between logos

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
ax.plot(ensemble_MHHW['date'], ensemble_MHHW['mean_prediction'], color='cyan', label="Emsemble Mean", linewidth=2)
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
ax.set_title(f"{station_name} Maximum Daily Tide Predictions for {ensemble_MHHW['date'].iloc[1].strftime('%m-%d-%y')}\n{model} Ensemble Forecast")

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

# Get the legend handles and labels
handles, labels = ax.get_legend_handles_labels()

# Add custom patches for filled areas to handles and labels
handles.extend([patch_10_90, patch_25_75])
labels.extend(['10th/90th Percentile', '25th/75th Percentile'])

# Sort handles and labels based on the linestyle of the handles
sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[0].get_linestyle())
sorted_handles, sorted_labels = zip(*sorted_handles_labels)

# Place the legend outside the plot area, below the x-axis, in two columns
ax.legend(handles=sorted_handles, labels=sorted_labels, loc='upper right', bbox_to_anchor=(1, -0.15), ncol=2)
ax.set_facecolor('white')

# Add gridlines
ax.grid(True)

# Slant dates on x-axis diagonally upward to the right
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Save the graph as a JPG file
figure_name = f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_DAILY.png"
plt.savefig(figure_name, format='png', bbox_inches='tight', dpi=300)

# Save the graph as a PNG file
figure_name = f"{OUTPUT_DIR}latest_{station_abbr}_{model}_{forecast_type}_DAILY.png"
plt.savefig(figure_name, format='png', bbox_inches='tight', dpi=300)

plt.show()


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
plot_data.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_MHHW_summary_6min.csv", index=False)

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
plot_data_NAVD.to_csv(f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_NAVD_summary_6min.csv", index=False)

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
spacing = 0.00  # space between logos

# Positioning sfwmd_logo
ax_sfwmd = fig.add_axes([0.03, 0.03, sfwmd_logo_width, logo_height])
ax_sfwmd.imshow(sfwmd_logo)
ax_sfwmd.axis('off')

# Positioning miami_logo next to sfwmd_logo
ax_miami = fig.add_axes([0.0 + sfwmd_logo_width + spacing, 0.03, miami_logo_width, logo_height])
ax_miami.imshow(miami_logo)
ax_miami.axis('off')

ax.plot(plot_data['datetime'], plot_data['observations'], color='b', label="Observations", linewidth=1.2)
ax.plot(plot_data['datetime'], plot_data['noaa'], color=(0.8, 0, 0), label="NOAA Prediction", linewidth=1.2)
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
ax.set_title(f"{station_name} 6-Minute Tide Predictions for {ensemble_MHHW['date'].iloc[1].strftime('%m-%d-%y')}\n{model} Ensemble Forecast")

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

# Get the legend handles and labels
handles, labels = ax.get_legend_handles_labels()

# Add custom patches for filled areas to handles and labels
handles.extend([patch_10_90, patch_25_75])
labels.extend(['10th/90th Percentile', '25th/75th Percentile'])

# Sort handles and labels based on the linestyle of the handles
sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[0].get_linestyle())
sorted_handles, sorted_labels = zip(*sorted_handles_labels)

# Place the legend outside the plot area, below the x-axis, in two columns
ax.legend(handles=sorted_handles, labels=sorted_labels, loc='upper right', bbox_to_anchor=(1, -0.15), ncol=2)
ax.set_facecolor('white')

# Add gridlines
ax.grid(True)

# Slant dates on x-axis diagonally upward to the right
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Make space for the legend below the plot
# fig.subplots_adjust(bottom=0.25)  # adjust this value if needed

# Save the graph as a PNG file
figure_name = f"{OUTPUT_DIR}{today.strftime('%Y-%m-%d')}_{station_abbr}_{model}_{forecast_type}_6MIN.png"
plt.savefig(figure_name, format='png', bbox_inches='tight', dpi=300)

# Save the graph as a PNG file
figure_name = f"{OUTPUT_DIR}latest_{station_abbr}_{model}_{forecast_type}_6MIN.png"
plt.savefig(figure_name, format='png', bbox_inches='tight', dpi=300)

plt.show()
