import os
import platform
import datetime
import time
import requests

# Define the base directory based on the hostname
hostname = platform.node()
if hostname == 'Nates-MacBook-Pro.local' or hostname == 'UM-C02RQ1S8FVH7':
    BASE_DIR = '/Users/nate/Documents/SFWMD_work/'
elif hostname == 'mango.rsmas.miami.edu':
    BASE_DIR = '/home/ntaminger/SFWMD_work/'
elif hostname == 'aztidal01p':
    BASE_DIR = '/home/ntaminge/SFWMD_work/'
else:
    raise ValueError(f"Unknown hostname: {hostname}")

pwd = os.path.join(BASE_DIR, 'predictions/')
os.chdir(pwd)

# set days back window
db = 0
# Get today's date and time
today = datetime.date.today() - datetime.timedelta(days = db)
bdate2 = (today - datetime.timedelta(days=2)).strftime('%Y%m%d')
bdate1 = (today - datetime.timedelta(days=1)).strftime('%Y%m%d')
date = today.strftime('%Y%m%d')  # e.g., 20230927
time_cycle = 0  # 0, 6, 12, 18
cycle = f"{time_cycle:02d}"  # Convert to 2-digit string
runs = 30

# Latitude and longitude ranges
lat1, lat2, lon1, lon2 = 20.0, 30.0, 275.0, 285.0

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    time.sleep(2)  # sleep for 10 seconds as in the original bash script

for bdate in [bdate2, bdate1, date]:
    if bdate == bdate2:
        for hour in ['06', '12', '18']:
            for i in range(1, runs+1):
                member = f"gep{i:02d}"
                url = (
                      f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?dir=%2Fgefs.{bdate}%2F{hour}%2Fatmos%2Fpgrb2sp25&"
                      f"file={member}.t{hour}z.pgrb2s.0p25.f000&var_PRMSL=on&var_TMP=on&var_UGRD=on&var_VGRD=on"
                      f"&lev_10_m_above_ground=on&lev_surface=on&lev_mean_sea_level=on&subregion="
                      f"&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
                      )
                filename = f"{member}.{bdate}.t{hour}z.f000.grib2"
                download_file(url, filename)
    elif bdate == bdate1:
        for hour in ['00', '06', '12', '18']:
            for i in range(1, runs+1):
                member = f"gep{i:02d}"
                url = (
                      f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?dir=%2Fgefs.{bdate}%2F{hour}%2Fatmos%2Fpgrb2sp25&"
                      f"file={member}.t{hour}z.pgrb2s.0p25.f000&var_PRMSL=on&var_TMP=on&var_UGRD=on&var_VGRD=on"
                      f"&lev_10_m_above_ground=on&lev_surface=on&lev_mean_sea_level=on&subregion="
                      f"&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
                      )
                filename = f"{member}.{bdate}.t{hour}z.f000.grib2"
                download_file(url, filename)
    else:
        for hour in ['00']:
            for i in range(1, runs+1):
                member = f"gep{i:02d}"
                url = (
                      f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?dir=%2Fgefs.{bdate}%2F{hour}%2Fatmos%2Fpgrb2sp25&"
                      f"file={member}.t{hour}z.pgrb2s.0p25.f000&var_PRMSL=on&var_TMP=on&var_UGRD=on&var_VGRD=on"
                      f"&lev_10_m_above_ground=on&lev_surface=on&lev_mean_sea_level=on&subregion="
                      f"&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
                      )
                filename = f"{member}.{bdate}.t{hour}z.f000.grib2"
                download_file(url, filename)

# Base URL for the data source
base_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?dir=%2Fgefs.{date}%2F{cycle}%2Fatmos%2Fpgrb2sp25&"
# Loop through forecast hours and download data
for i in range(0, 242, 3):
    fhr = f"{i:03d}"  # Convert to 3-digit string
    for j in range(1, runs+1):
        member = f"gep{j:02d}"
        # Construct the URL for this forecast hour
        url = (
            f"{base_url}"
            f"file={member}.t{cycle}z.pgrb2s.0p25.f{fhr}&var_PRMSL=on&var_TMP=on&var_UGRD=on&var_VGRD=on"
            f"&lev_10_m_above_ground=on&lev_surface=on&lev_mean_sea_level=on&subregion="
            f"&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
        )
        
        # Construct the filename for this forecast hour
        filename = f"{member}.{date}.t{cycle}z.f{fhr}.grib2"
        
        # Download the file
        download_file(url, filename)
    
