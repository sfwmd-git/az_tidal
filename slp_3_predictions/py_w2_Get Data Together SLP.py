import os
import platform
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob
import re

experiment = "slp"
db = 0

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
predictions_subdir = f'{test}3_predictions/'
output_subdir = f'{test}3_predictions/all_stations'


# Define other directories relative to the base directory and site-specific subdirectories
META_DIR = os.path.join(BASE_DIR, stationinfo_subdir)
PRED_DIR = os.path.join(BASE_DIR, predictions_subdir)
OUTPUT_DIR = os.path.join(BASE_DIR, output_subdir)

os.chdir(BASE_DIR)

# ------------------------------------------------------------------------------
#
# 2. Read station_data.csv for site metadata (flood thresholds, station names)
#
# ------------------------------------------------------------------------------

csv_file = os.path.join(META_DIR, 'station_data.csv')
df_meta = pd.read_csv(csv_file)

# Suppose station_data.csv has columns like: 
#   site_name, station_name, minor_flood, moderate_flood, major_flood, etc.

# We'll store them in a dict for easy lookups, or we can just merge later.

# ------------------------------------------------------------------------------
#
# 3. Define which "_master" directories exist and how they map to "Model" labels
#
# ------------------------------------------------------------------------------
master_dirs = {
    "GFS_DET_master":   "GFS Deterministic",
    "GFS_ENS_master":   "GFS Ensemble",
    "ECMWF_DET_master": "ECMWF Deterministic",
    "ECMWF_ENS_master": "ECMWF Ensemble",
}

# We'll define patterns for daily vs. 6min. 
# Example: "*NAVD*_daily.csv" or "*NAVD*_6min.csv"
PATTERN_DAILY = "*NAVD*_daily.csv"
PATTERN_6MIN  = "*NAVD*_6min.csv"

# ------------------------------------------------------------------------------
#
# 4. Helper functions to read & rename columns for daily vs. 6-min
#
# ------------------------------------------------------------------------------
def process_daily_file(file_path, model_label):
    """
    Example function that reads a 'daily' CSV and renames columns 
    to a standard set. Adjust these as needed.
    """
    df = pd.read_csv(file_path)
    # Example rename
    rename_map = {
        "date":                   "Forecast Date",
        "days_ahead":            "Lead Time (Days)",
        "noaa":                  "NOAA Prediction (ft. NAVD88)",
        "noaa_slr":              "NOAA Prediction + SLR (ft. NAVD88)",
        # Either GFS DET/ENS or ECMWF DET might have "station_prediction"
        "station_prediction":     "Tide Forecast (ft. NAVD88)",
        # ECMWF ENS might have "mean_prediction" for the ensemble mean
        "mean_prediction":        "Tide Forecast (ft. NAVD88)",
        # ECMWF ENS or GFS ENS might have percentiles:
        "percentile_10_prediction": "10th Percentile (ft. NAVD88)",
        "percentile_25_prediction": "25th Percentile (ft. NAVD88)",
        "percentile_75_prediction": "75th Percentile (ft. NAVD88)",
        "percentile_90_prediction": "90th Percentile (ft. NAVD88)"
    }
    df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)
    
    # Add extra columns
    df["Model"] = model_label
    # For daily data, let's define a "Date/Time (ET)" = midnight
    if "Forecast Date" in df.columns:
        df["Date/Time (ET)"] = pd.to_datetime(df["Forecast Date"]).dt.strftime("%Y-%m-%dT00:00:00")
    else:
        df["Date/Time (ET)"] = ""  # fallback if needed
    return df

def process_6min_file(file_path, model_label):
    """
    Reads a 6-minute CSV (either GFS or ECMWF, DET or ENS) and renames columns
    to a unified standard. The rename map includes both deterministic and ensemble
    possibilities (e.g., station_prediction, mean_prediction, percentile_*).
    """
    df = pd.read_csv(file_path)

    # Combined rename map for GFS or ECMWF, DET or ENS
    rename_map = {
        # Old script had "datetime" -> "Date/Time (ET)" and "observations" -> "Observation (ft. NAVD88)"
        "datetime":   "Date/Time (ET)",
        "observations": "Observation (ft. NAVD88)",

        # Shared columns for NOAA predictions
        "noaa":       "NOAA Prediction (ft. NAVD88)",
        "noaa_slr":   "NOAA Prediction + SLR (ft. NAVD88)",

        # GFS DET/ENS might use "station_prediction"
        "station_prediction": "Tide Forecast (ft. NAVD88)",
        # ECMWF ENS might have "mean_prediction" for the ensemble mean
        "mean_prediction":    "Tide Forecast (ft. NAVD88)",

        # Ensemble percentiles
        "percentile_10_prediction": "10th Percentile (ft. NAVD88)",
        "percentile_25_prediction": "25th Percentile (ft. NAVD88)",
        "percentile_75_prediction": "75th Percentile (ft. NAVD88)",
        "percentile_90_prediction": "90th Percentile (ft. NAVD88)",

        # If your 6-min files also have "days_ahead" or "date", you can map them here:
        "days_ahead": "Lead Time (Days)",
        "date":       "Forecast Date"
    }

    # Rename only columns that actually exist
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Mark the model (e.g. "GFS Deterministic", "ECMWF Ensemble")
    df["Model"] = model_label

    # If "Date/Time (ET)" is present, parse it to ISO format
    if "Date/Time (ET)" in df.columns:
        df["Date/Time (ET)"] = pd.to_datetime(df["Date/Time (ET)"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        # Also define "Forecast Date" from that if not already set
        if "Forecast Date" not in df.columns:
            df["Forecast Date"] = pd.to_datetime(df["Date/Time (ET)"]).dt.strftime("%Y-%m-%d")
    else:
        # fallback if no datetime column
        df["Forecast Date"] = ""

    return df

def build_daily_graph_filename(row, today_str):
    """
    Returns something like '2025-03-26_FT_GFS_DET_DAILY.png'
    or '2025-03-26_FT_ECMWF_DAILY.png' depending on row["Model"].
    """
    station_abbr = row.get("station_abbr", "???")  # fallback if missing
    model = row.get("Model", "")
    
    # Decide GFS vs. ECMWF short label
    if "GFS" in model.upper():
        model_short = "GFS"
    elif "ECMWF" in model.upper():
        model_short = "ECMWF"
    else:
        model_short = "UNKNOWN"
        
    # Decide DET vs. ENS short label
    if "DETERMINISTIC" in model.upper():
        forecast_short = "DET"
    elif "ENSEMBLE" in model.upper():
        forecast_short = "ENS"
    else:
        forecast_short = "UNKNOWN"
    
    return f"{today_str}_{station_abbr}_{model_short}_{forecast_short}_DAILY.png"

def build_6min_graph_filename(row, today_str):
    """
    Returns something like '2025-03-26_FT_GFS_DET_DAILY.png'
    or '2025-03-26_FT_ECMWF_DAILY.png' depending on row["Model"].
    """
    station_abbr = row.get("station_abbr", "???")  # fallback if missing
    model = row.get("Model", "")
    
    # Decide GFS vs. ECMWF short label
    if "GFS" in model.upper():
        model_short = "GFS"
    elif "ECMWF" in model.upper():
        model_short = "ECMWF"
    else:
        model_short = "UNKNOWN"
        
    # Decide DET vs. ENS short label
    if "DETERMINISTIC" in model.upper():
        forecast_short = "DET"
    elif "ENSEMBLE" in model.upper():
        forecast_short = "ENS"
    else:
        forecast_short = "UNKNOWN"
    
    return f"{today_str}_{station_abbr}_{model_short}_{forecast_short}_6MIN.png"

# ------------------------------------------------------------------------------
#
# 5. Main logic: find daily/6min files in each master_dir -> site -> CSV
#
# ------------------------------------------------------------------------------
today_str = (datetime.date.today() - datetime.timedelta(days=db)).strftime('%Y-%m-%d') # e.g. '2025-03-25'
all_daily = []
all_6min  = []

for mdir, model_label in master_dirs.items():
    # e.g. /predictions/GFS_DET_master
    top_path = os.path.join(PRED_DIR, mdir)
    
    # inside each master directory, you presumably have subfolders named 
    # after site_name, e.g. "virginia_key", "lake_worth", etc.
    # We'll do a loop over subdirectories
    if not os.path.isdir(top_path):
        continue  # skip if directory doesn't exist
    
    # list subdirs
    site_folders = [d for d in os.listdir(top_path) 
                    if os.path.isdir(os.path.join(top_path, d))]
    
    for site_name in site_folders:
        # build the site path
        site_path = os.path.join(top_path, site_name)
        
        # Find daily CSVs
        daily_files = glob.glob(os.path.join(site_path, f"{today_str}*{PATTERN_DAILY}"))
        # e.g. "2025-03-25*NAVD*_daily.csv"
        for f in daily_files:
            df_daily = process_daily_file(f, model_label)
            # store site_name in a column so we can merge with station_data
            df_daily["site_name"] = site_name
            all_daily.append(df_daily)
        
        # Find 6-min CSVs
        min_files = glob.glob(os.path.join(site_path, f"{today_str}*{PATTERN_6MIN}"))
        for f in min_files:
            df_6min = process_6min_file(f, model_label)
            df_6min["site_name"] = site_name
            all_6min.append(df_6min)

# ------------------------------------------------------------------------------
# 6. Combine daily data, merge with station_data, apply flood thresholds
# ------------------------------------------------------------------------------
if all_daily:
    daily_df = pd.concat(all_daily, ignore_index=True)
else:
    daily_df = pd.DataFrame()

daily_df['site_name'] = daily_df['site_name'].str.replace(r'^\d+_', '', regex=True)

# Merge in metadata
daily_merged = pd.merge(daily_df, df_meta, on="site_name", how="left")
# Suppose station_data.csv has columns: [site_name, minor_flood, moderate_flood, major_flood, station_abbr, station_name, etc.]

# Example flood threshold logic: 
# (We only do this if we have 'Tide Forecast (ft. NAVD88)' and 'major_flood' columns)
daily_merged['Flood Category (NWS)'] = 'None'
mask_major = daily_merged['Tide Forecast (ft. NAVD88)'] > daily_merged['major_flood']
daily_merged.loc[mask_major, 'Flood Category (NWS)'] = 'Major'

mask_mod = (daily_merged['Tide Forecast (ft. NAVD88)'] > daily_merged['moderate_flood']) & ~mask_major
daily_merged.loc[mask_mod, 'Flood Category (NWS)'] = 'Moderate'

mask_min = (daily_merged['Tide Forecast (ft. NAVD88)'] > daily_merged['minor_flood']) & ~mask_mod & ~mask_major
daily_merged.loc[mask_min, 'Flood Category (NWS)'] = 'Minor'

# Round numeric columns
daily_merged = daily_merged.round(2)

# 1. Set Run Date to today's date
daily_merged['Run Date'] = today_str
daily_merged["Graph"] = daily_merged.apply(lambda row: build_daily_graph_filename(row, today_str), axis=1)
daily_merged["Tide Gauge"] = daily_merged["station_name"]
daily_merged["Test"] = "SLP Regression"


# 2. Reorder columns
ordered_cols = [
    'Run Date',
    'Forecast Date',
    'Lead Time (Days)',
    'Date/Time (ET)',
    'Model',
    'Tide Gauge',
    'Test',
    'NOAA Prediction (ft. NAVD88)',
    'NOAA Prediction + SLR (ft. NAVD88)',
    'Tide Forecast (ft. NAVD88)',
    '10th Percentile (ft. NAVD88)',
    '25th Percentile (ft. NAVD88)',
    '75th Percentile (ft. NAVD88)',
    '90th Percentile (ft. NAVD88)',
    'Flood Category (NWS)',
    'Graph'
]

daily_merged = daily_merged.reindex(columns=ordered_cols)

# Save daily combined
out_daily = os.path.join(OUTPUT_DIR, f"{today_str}_ALL_STATIONS_COMBINED_DAILY.csv")
daily_merged.to_csv(out_daily, index=False)
print(f"Saved daily combined to {out_daily}")

# ------------------------------------------------------------------------------
#
# 7. Combine 6-min data, merge with station_data, apply flood thresholds
#
# ------------------------------------------------------------------------------
if all_6min:
    min_df = pd.concat(all_6min, ignore_index=True)
else:
    min_df = pd.DataFrame()

min_df['site_name'] = min_df['site_name'].str.replace(r'^\d+_', '', regex=True)

min_merged = pd.merge(min_df, df_meta, on="site_name", how="left")
min_merged['Flood Category (NWS)'] = 'None'

mask_major = min_merged['Tide Forecast (ft. NAVD88)'] > min_merged['major_flood']
min_merged.loc[mask_major, 'Flood Category (NWS)'] = 'Major'

mask_mod = (min_merged['Tide Forecast (ft. NAVD88)'] > min_merged['moderate_flood']) & ~mask_major
min_merged.loc[mask_mod, 'Flood Category (NWS)'] = 'Moderate'

mask_min = (min_merged['Tide Forecast (ft. NAVD88)'] > min_merged['minor_flood']) & ~mask_mod & ~mask_major
min_merged.loc[mask_min, 'Flood Category (NWS)'] = 'Minor'

min_merged = min_merged.round(2)

# 1. Set Run Date to today's date
min_merged['Run Date'] = today_str
min_merged["Graph"] = min_merged.apply(lambda row: build_6min_graph_filename(row, today_str), axis=1)
min_merged["Tide Gauge"] = min_merged["station_name"]
min_merged["Test"] = "SLP Regression"

# 2. Reorder columns
ordered_cols = [
    'Run Date',
    'Forecast Date',
    'Date/Time (ET)',
    'Model',
    'Tide Gauge',
    'Test',
    'Observation (ft. NAVD88)',
    'NOAA Prediction (ft. NAVD88)',
    'NOAA Prediction + SLR (ft. NAVD88)',
    'Tide Forecast (ft. NAVD88)',
    '10th Percentile (ft. NAVD88)',
    '25th Percentile (ft. NAVD88)',
    '75th Percentile (ft. NAVD88)',
    '90th Percentile (ft. NAVD88)',
    'Flood Category (NWS)',
    'Graph'
]

min_merged = min_merged.reindex(columns=ordered_cols)

# Save 6-min combined
out_6min = os.path.join(OUTPUT_DIR, f"{today_str}_ALL_STATIONS_COMBINED_6MIN.csv")
min_merged.to_csv(out_6min, index=False)
print(f"Saved 6-min combined to {out_6min}")
