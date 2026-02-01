import os
import platform
import datetime
import time
import requests

model = "GFS"
forecast_type = "DET"

# List of allowed hostnames
allowed_hostnames = [
    'Nates-MacBook-Pro.local', 'UM-C02RQ1S8FVH7', 'nates-mbp.lan',
    'Nates-MBP.lan', 'Nates-MBP.localdomain',
    'Nates-MBP.homenet.telecomitalia.it'
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

pwd = os.path.join(BASE_DIR, f'1_data/forecasts/{model}')
os.chdir(pwd)

# set days back window
db = 0

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    time.sleep(2)  # sleep for 10 seconds as in the original bash script

# Get today's date and time
today = datetime.date.today() - datetime.timedelta(days = db)
bdate2 = (today - datetime.timedelta(days=2)).strftime('%Y%m%d')
bdate1 = (today - datetime.timedelta(days=1)).strftime('%Y%m%d')
date = today.strftime('%Y%m%d')  # e.g., 20230927
time_cycle = 0  # 0, 6, 12, 18
cycle = f"{time_cycle:02d}"  # Convert to 2-digit string

# Latitude and longitude ranges
lat1, lat2, lon1, lon2 = 20.0, 30.0, 275.0, 285.0

for bdate in [bdate2, bdate1]:
    for hour in ['06', '12', '18']:
        url = (
              f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{bdate}%2F{hour}%2Fatmos&"
              f"file=gfs.t{hour}z.pgrb2.0p25.f000&var_PRMSL=on&var_TMP=on&var_UGRD=on&var_VGRD=on"
              f"&lev_10_m_above_ground=on&lev_surface=on&lev_mean_sea_level=on&subregion="
              f"&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
          )
        filename = f"gfsall.{bdate}.t{hour}z.f000.grib2"
        download_file(url, filename)
        
url = (
      f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{bdate1}%2F00%2Fatmos&"
      f"file=gfs.t00z.pgrb2.0p25.f000&var_PRMSL=on&var_TMP=on&var_UGRD=on&var_VGRD=on"
      f"&lev_10_m_above_ground=on&lev_surface=on&lev_mean_sea_level=on&subregion="
      f"&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
  )
filename = f"gfsall.{bdate1}.t00z.f000.grib2"
download_file(url, filename)

url = (
      f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{bdate}%2F00%2Fatmos&"
      f"file=gfs.t00z.pgrb2.0p25.f000&var_PRMSL=on&var_TMP=on&var_UGRD=on&var_VGRD=on"
      f"&lev_10_m_above_ground=on&lev_surface=on&lev_mean_sea_level=on&subregion="
      f"&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
  )
filename = f"gfsall.{date}.t00z.f000.grib2"
download_file(url, filename)

# Base URL for the data source
base_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{date}%2F{cycle}%2Fatmos&"

# Loop through forecast hours and download data
for i in range(0, 244, 3):
    fhr = f"{i:03d}"  # Convert to 3-digit string
    # Construct the URL for this forecast hour
    url = (
        f"{base_url}"
        f"file=gfs.t{cycle}z.pgrb2.0p25.f{fhr}&var_PRMSL=on&var_TMP=on&var_UGRD=on&var_VGRD=on"
        f"&lev_10_m_above_ground=on&lev_surface=on&lev_mean_sea_level=on&subregion="
        f"&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
    )
    
    # Construct the filename for this forecast hour
    filename = f"gfsall.{date}.t{cycle}z.f{fhr}.grib2"
    
    # Download the file
    download_file(url, filename)

