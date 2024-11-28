import os
import platform
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob


#-------------------------------------------------------------------------------
#
#              ESTABLISH GENERAL VARIABLES
#
#-------------------------------------------------------------------------------

# Define the base directory based on the hostname
hostname = platform.node()
if hostname == 'Nates-MacBook-Pro.local' or hostname == 'nates-mbp.lan' or hostname == 'UM-C02RQ1S8FVH7':
    BASE_DIR = '/Users/nate/Documents/SFWMD_work/'
elif hostname == 'mango.rsmas.miami.edu':
    BASE_DIR = '/home/ntaminger/SFWMD_work/'
elif hostname == 'aztidal01p':
    BASE_DIR = '/home/ntaminge/SFWMD_work/'
else:
    raise ValueError(f"Unknown hostname: {hostname}")

pwd = os.path.join(BASE_DIR, 'predictions/')
meta = pd.read_csv(f"{BASE_DIR}metadata/stationdata.csv")
db = 0

#-------------------------------------------------------------------------------
#
#              SET DIRECTORIES
#
#-------------------------------------------------------------------------------
vk_path = f"{pwd}virginia_key/"
lw_path = f"{pwd}lake_worth/"
pe_path = f"{pwd}port_everglades/"
vaca_path = f"{pwd}vaca_key/"
kw_path = f"{pwd}key_west/"
naples_path = f"{pwd}naples/"
naplesBay_path = f"{pwd}naples_bay/"
fm_path = f"{pwd}fort_myers/"

OUTPUT_DIR = f"{pwd}all_stations/"

paths = {
    "virginia_key": vk_path,
    "lake_worth": lw_path,
    "port_everglades": pe_path,
    "vaca_key": vaca_path,
    "key_west": kw_path,
    "naples": naples_path,
    "naples_bay": naplesBay_path,
    "fort_myers": fm_path
}

names = {
    "virginia_key": "Virginia Key",
    "lake_worth": "Lake Worth",
    "port_everglades": "Port Everglades",
    "vaca_key": "Vaca Key",
    "key_west": "Key West",
    "naples": "Naples",
    "naples_bay": "Naples Bay",
    "fort_myers": "Fort Myers"
}

shorts= {
    "virginia_key": "VK",
    "lake_worth": "LW",
    "port_everglades": "PE",
    "vaca_key": "Vaca",
    "key_west": "KW",
    "naples": "Naples",
    "naples_bay": "NaplesBay",
    "fort_myers": "FM"
}
#-------------------------------------------------------------------------------
#
#              READ DETERMINISTIC DAILY
#
#-------------------------------------------------------------------------------
today_date = datetime.date.today() - datetime.timedelta(days = db)

# Function to process each file
def process_daily_det_file(file_path, tide_gauge_name, model_name, short_name, model_short):
    df = pd.read_csv(file_path)
    df["Run Date"] = today_date
    df.rename(columns={"date": "Forecast Date", "days_ahead": "Lead Time (Days)", "noaa":"NOAA Prediction (ft. NAVD88)", "noaa_slr":"NOAA Prediction + SLR (ft. NAVD88)", "station_prediction":"Tide Forecast (ft. NAVD88)"}, inplace=True)
    df.drop(columns=["station_SLP_prediction"], inplace=True)
    df["Date/Time (ET)"] = pd.to_datetime(df["Forecast Date"]).dt.strftime("%Y-%m-%dT00:00:00")
    df["Model"] = model_name
    df["Tide Gauge"] = tide_gauge_name
    df["Graph"] = f"{today_date}_{short_name}_{model_short}_DAILY.png"
    return df

# Loop through directories and process files
all_dfs = []
for key, path in paths.items():
    tide_gauge_name = names[key]
    short_name = shorts[key]
    ecmwf_files = glob.glob(f"{path}{today_date}*ECMWF_DET_NAVD*daily.csv")
    gfs_files = glob.glob(f"{path}{today_date}*GFS_DET_NAVD*daily.csv")
    
    for ecmwf_file in ecmwf_files:
        df_ecmwf = process_daily_det_file(ecmwf_file, tide_gauge_name, "ECMWF Deterministic", short_name, "ECMWF_DET")
        all_dfs.append(df_ecmwf)
    
    for gfs_file in gfs_files:
        df_gfs = process_daily_det_file(gfs_file, tide_gauge_name, "GFS Deterministic", short_name, "GFS_DET")
        all_dfs.append(df_gfs)

# Combine all dataframes into a single dataframe
if all_dfs:
    det_daily_df = pd.concat(all_dfs, ignore_index=True)
    print(det_daily_df)  # or save the final dataframe to a CSV or other operations
else:
    det_daily_df = pd.DataFrame()
    print("No files found matching the criteria.")

det_daily = pd.merge(det_daily_df, meta, left_on="Tide Gauge", right_on="station_name", how="left")
det_daily['Flood Category (NWS)'] = 'None'

# Apply the conditions in order from highest threshold to lowest
det_daily.loc[det_daily['Tide Forecast (ft. NAVD88)'] > det_daily['major_flood'], 'Flood Category (NWS)'] = 'Major'
det_daily.loc[(det_daily['Tide Forecast (ft. NAVD88)'] > det_daily['moderate_flood']) & 
              (det_daily['Tide Forecast (ft. NAVD88)'] <= det_daily['major_flood']), 'Flood Category (NWS)'] = 'Moderate'
det_daily.loc[(det_daily['Tide Forecast (ft. NAVD88)'] > det_daily['minor_flood']) & 
              (det_daily['Tide Forecast (ft. NAVD88)'] <= det_daily['moderate_flood']), 'Flood Category (NWS)'] = 'Minor'

required_columns = [
    'Run Date', 'Forecast Date', 'Lead Time (Days)', 'Date/Time (ET)', 'Model', 
    'Tide Gauge', 'NOAA Prediction (ft. NAVD88)', 'NOAA Prediction + SLR (ft. NAVD88)', 
    'Tide Forecast (ft. NAVD88)', 'Flood Category (NWS)', 'Graph'
]

# Select and reorder the columns
det_daily = det_daily[required_columns]
det_daily = det_daily.round(2)
det_daily.to_csv(f"{OUTPUT_DIR}{today_date.strftime('%Y-%m-%d')}_ALL_STATIONS_COMBINED_DAILY.csv", index=False)

#-------------------------------------------------------------------------------
#
#              READ DETERMINISTIC 6 MINUTE
#
#-------------------------------------------------------------------------------
today_date = datetime.date.today() - datetime.timedelta(days = db)

# Function to process each file
def process_6min_det_file(file_path, tide_gauge_name, model_name, short_name, model_short):
    print(file_path)
    df = pd.read_csv(file_path)
    df["Run Date"] = today_date
    df.rename(columns={"datetime": "Date/Time (ET)", "observations":"Observation (ft. NAVD88)", "noaa":"NOAA Prediction (ft. NAVD88)", "noaa_slr":"NOAA Prediction + SLR (ft. NAVD88)", "station_prediction":"Tide Forecast (ft. NAVD88)"}, inplace=True)
    df.drop(columns=["station_SLP_prediction"], inplace=True)
    df["Date/Time (ET)"] = pd.to_datetime(df["Date/Time (ET)"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["Forecast Date"] = pd.to_datetime(df["Date/Time (ET)"]).dt.strftime("%Y-%m-%d")
    df["Model"] = model_name
    df["Tide Gauge"] = tide_gauge_name
    df["Graph"] = f"{today_date}_{short_name}_{model_short}_6MIN.png"
    print(df.columns)
    return df

# Loop through directories and process files
all_dfs = []
for key, path in paths.items():
    tide_gauge_name = names[key]
    short_name = shorts[key]
    ecmwf_files = glob.glob(f"{path}{today_date}*ECMWF_DET_NAVD*min.csv")
    gfs_files = glob.glob(f"{path}{today_date}*GFS_DET_NAVD*min.csv")
    
    for ecmwf_file in ecmwf_files:
        df_ecmwf = process_6min_det_file(ecmwf_file, tide_gauge_name, "ECMWF Deterministic", short_name, "ECMWF_DET")
        all_dfs.append(df_ecmwf)
    
    for gfs_file in gfs_files:
        df_gfs = process_6min_det_file(gfs_file, tide_gauge_name, "GFS Deterministic", short_name, "GFS_DET")
        all_dfs.append(df_gfs)

# Combine all dataframes into a single dataframe
if all_dfs:
    det_min_df = pd.concat(all_dfs, ignore_index=True)
    print(det_min_df)  # or save the final dataframe to a CSV or other operations
else:
    det_min_df = pd.DataFrame()
    print("No files found matching the criteria.")

det_min = pd.merge(det_min_df, meta, left_on="Tide Gauge", right_on="station_name", how="left")
det_min['Flood Category (NWS)'] = 'None'
print(det_min)

# Apply the conditions in order from highest threshold to lowest
det_min.loc[det_min['Tide Forecast (ft. NAVD88)'] > det_min['major_flood'], 'Flood Category (NWS)'] = 'Major'
det_min.loc[(det_min['Tide Forecast (ft. NAVD88)'] > det_min['moderate_flood']) & 
              (det_min['Tide Forecast (ft. NAVD88)'] <= det_min['major_flood']), 'Flood Category (NWS)'] = 'Moderate'
det_min.loc[(det_min['Tide Forecast (ft. NAVD88)'] > det_min['minor_flood']) & 
              (det_min['Tide Forecast (ft. NAVD88)'] <= det_min['moderate_flood']), 'Flood Category (NWS)'] = 'Minor'

required_columns = [
    'Run Date', 'Forecast Date', 'Date/Time (ET)', 'Model', 'Tide Gauge', 
    'Observation (ft. NAVD88)', 'NOAA Prediction (ft. NAVD88)', 'NOAA Prediction + SLR (ft. NAVD88)', 
    'Tide Forecast (ft. NAVD88)', 'Flood Category (NWS)', 'Graph'
]

print(det_min.columns)
# Select and reorder the columns
det_min = det_min[required_columns]
det_min = det_min.round(2)
det_min.to_csv(f"{OUTPUT_DIR}{today_date.strftime('%Y-%m-%d')}_ALL_STATIONS_COMBINED_6MIN.csv", index=False)

#-------------------------------------------------------------------------------
#
#              READ ENSEMBLE DAILY
#
#-------------------------------------------------------------------------------
today_date = datetime.date.today() - datetime.timedelta(days = db)

# Function to process each file
def process_daily_ens_file(file_path, tide_gauge_name, model_name, short_name, model_short):
    df = pd.read_csv(file_path)
    df["Run Date"] = today_date
    df.rename(columns={"date": "Forecast Date", "days_ahead": "Lead Time (Days)", "noaa":"NOAA Prediction (ft. NAVD88)", "noaa_slr":"NOAA Prediction + SLR (ft. NAVD88)", "mean_prediction":"Tide Forecast (ft. NAVD88)"}, inplace=True)
    df.rename(columns={"percentile_10_prediction": "10th Percentile (ft. NAVD88)", "percentile_25_prediction": "25th Percentile (ft. NAVD88)", "percentile_75_prediction": "75th Percentile (ft. NAVD88)", "percentile_90_prediction": "90th Percentile (ft. NAVD88)"}, inplace=True)
    df["Date/Time (ET)"] = pd.to_datetime(df["Forecast Date"]).dt.strftime("%Y-%m-%dT00:00:00")
    df["Model"] = model_name
    df["Tide Gauge"] = tide_gauge_name
    df["Graph"] = f"{today_date}_{short_name}_{model_short}_DAILY.png"
    return df

# Loop through directories and process files
all_dfs = []
for key, path in paths.items():
    tide_gauge_name = names[key]
    short_name = shorts[key]
    ecmwf_files = glob.glob(f"{path}{today_date}*ECMWF_ENS_NAVD*daily.csv")
    gfs_files = glob.glob(f"{path}{today_date}*GFS_ENS_NAVD*daily.csv")
    
    for ecmwf_file in ecmwf_files:
        df_ecmwf = process_daily_ens_file(ecmwf_file, tide_gauge_name, "ECMWF Ensemble", short_name, "ECMWF_ENS")
        all_dfs.append(df_ecmwf)
    
    for gfs_file in gfs_files:
        df_gfs = process_daily_ens_file(gfs_file, tide_gauge_name, "GFS Ensemble", short_name, "GFS_ENS")
        all_dfs.append(df_gfs)

# Combine all dataframes into a single dataframe
if all_dfs:
    ens_daily_df = pd.concat(all_dfs, ignore_index=True)
    print(ens_daily_df)  # or save the final dataframe to a CSV or other operations
else:
    ens_daily_df = pd.DataFrame()
    print("No files found matching the criteria.")

ens_daily = pd.merge(ens_daily_df, meta, left_on="Tide Gauge", right_on="station_name", how="left")
ens_daily['Flood Category (NWS)'] = 'None'

# Apply the conditions in order from highest threshold to lowest
ens_daily.loc[ens_daily['90th Percentile (ft. NAVD88)'] > ens_daily['major_flood'], 'Flood Category (NWS)'] = 'Major'
ens_daily.loc[(ens_daily['90th Percentile (ft. NAVD88)'] > ens_daily['moderate_flood']) & 
              (ens_daily['90th Percentile (ft. NAVD88)'] <= ens_daily['major_flood']), 'Flood Category (NWS)'] = 'Moderate'
ens_daily.loc[(ens_daily['90th Percentile (ft. NAVD88)'] > ens_daily['minor_flood']) & 
              (ens_daily['90th Percentile (ft. NAVD88)'] <= ens_daily['moderate_flood']), 'Flood Category (NWS)'] = 'Minor'

required_columns = [
    'Run Date', 'Forecast Date', 'Lead Time (Days)', 'Date/Time (ET)', 'Model', 
    'Tide Gauge', 'NOAA Prediction (ft. NAVD88)', 'NOAA Prediction + SLR (ft. NAVD88)', 
    'Tide Forecast (ft. NAVD88)', '10th Percentile (ft. NAVD88)',
     '25th Percentile (ft. NAVD88)', '75th Percentile (ft. NAVD88)', '90th Percentile (ft. NAVD88)',
     'Flood Category (NWS)', 'Graph'
]

# Select and reorder the columns
ens_daily = ens_daily[required_columns]
ens_daily = ens_daily.round(2)
# ens_daily.to_csv(f"{OUTPUT_DIR}{today_date.strftime('%Y-%m-%d')}_ALL_STATIONS_ENS_DAILY.csv", index=False)
# daily first
combined_daily = pd.concat([ens_daily, det_daily.reindex(columns=ens_daily.columns)], ignore_index=True)
combined_daily.to_csv(f"{OUTPUT_DIR}{today_date.strftime('%Y-%m-%d')}_ALL_STATIONS_COMBINED_DAILY.csv", index=False)


#-------------------------------------------------------------------------------
#
#              READ DETERMINISTIC 6 MINUTE
#
#-------------------------------------------------------------------------------
today_date = datetime.date.today() - datetime.timedelta(days = db)

# Function to process each file
def process_6min_ens_file(file_path, tide_gauge_name, model_name, short_name, model_short):
    df = pd.read_csv(file_path)
    df["Run Date"] = today_date
    df.rename(columns={"datetime": "Date/Time (ET)", "observations":"Observation (ft. NAVD88)", "noaa":"NOAA Prediction (ft. NAVD88)", "noaa_slr":"NOAA Prediction + SLR (ft. NAVD88)", "mean_prediction":"Tide Forecast (ft. NAVD88)"}, inplace=True)
    df.rename(columns={"percentile_10_prediction": "10th Percentile (ft. NAVD88)", "percentile_25_prediction": "25th Percentile (ft. NAVD88)", "percentile_75_prediction": "75th Percentile (ft. NAVD88)", "percentile_90_prediction": "90th Percentile (ft. NAVD88)"}, inplace=True)
    df["Date/Time (ET)"] = pd.to_datetime(df["Date/Time (ET)"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["Forecast Date"] = pd.to_datetime(df["Date/Time (ET)"]).dt.strftime("%Y-%m-%d")
    df["Model"] = model_name
    df["Tide Gauge"] = tide_gauge_name
    df["Graph"] = f"{today_date}_{short_name}_{model_short}_6MIN.png"
    return df

# Loop through directories and process files
all_dfs = []
for key, path in paths.items():
    tide_gauge_name = names[key]
    short_name = shorts[key]
    ecmwf_files = glob.glob(f"{path}{today_date}*ECMWF_ENS_NAVD*min.csv")
    gfs_files = glob.glob(f"{path}{today_date}*GFS_ENS_NAVD*min.csv")
    
    for ecmwf_file in ecmwf_files:
        df_ecmwf = process_6min_ens_file(ecmwf_file, tide_gauge_name, "ECMWF Ensemble", short_name, "ECMWF_ENS")
        all_dfs.append(df_ecmwf)
    
    for gfs_file in gfs_files:
        df_gfs = process_6min_ens_file(gfs_file, tide_gauge_name, "GFS Ensemble", short_name, "GFS_ENS")
        all_dfs.append(df_gfs)

# Combine all dataframes into a single dataframe
if all_dfs:
    ens_min_df = pd.concat(all_dfs, ignore_index=True)
    print(ens_min_df)  # or save the final dataframe to a CSV or other operations
else:
    ens_min_df = pd.DataFrame()
    print("No files found matching the criteria.")

ens_min = pd.merge(ens_min_df, meta, left_on="Tide Gauge", right_on="station_name", how="left")
ens_min['Flood Category (NWS)'] = 'None'

# Apply the conditions in order from highest threshold to lowest
ens_min.loc[ens_min['90th Percentile (ft. NAVD88)'] > ens_min['major_flood'], 'Flood Category (NWS)'] = 'Major'
ens_min.loc[(ens_min['90th Percentile (ft. NAVD88)'] > ens_min['moderate_flood']) & 
              (ens_min['90th Percentile (ft. NAVD88)'] <= ens_min['major_flood']), 'Flood Category (NWS)'] = 'Moderate'
ens_min.loc[(ens_min['90th Percentile (ft. NAVD88)'] > ens_min['minor_flood']) & 
              (ens_min['90th Percentile (ft. NAVD88)'] <= ens_min['moderate_flood']), 'Flood Category (NWS)'] = 'Minor'

required_columns = [
    'Run Date', 'Forecast Date', 'Date/Time (ET)', 'Model', 'Tide Gauge', 'Observation (ft. NAVD88)', 
    'NOAA Prediction (ft. NAVD88)', 'NOAA Prediction + SLR (ft. NAVD88)', 'Tide Forecast (ft. NAVD88)', 
    '10th Percentile (ft. NAVD88)', '25th Percentile (ft. NAVD88)', '75th Percentile (ft. NAVD88)', 
    '90th Percentile (ft. NAVD88)', 'Flood Category (NWS)', 'Graph'
]

# Select and reorder the columns
ens_min = ens_min[required_columns]
ens_min = ens_min.round(2)
# ens_min.to_csv(f"{OUTPUT_DIR}{today_date.strftime('%Y-%m-%d')}_ALL_STATIONS_ENS_6MIN.csv", index=False)

#-------------------------------------------------------------------------------
#
#              COMBINE FORECAST TYPES
#
#-------------------------------------------------------------------------------
# minutes second
combined_min = pd.concat([ens_min, det_min.reindex(columns=ens_min.columns)], ignore_index=True)
combined_min.to_csv(f"{OUTPUT_DIR}{today_date.strftime('%Y-%m-%d')}_ALL_STATIONS_COMBINED_6MIN.csv", index=False)

