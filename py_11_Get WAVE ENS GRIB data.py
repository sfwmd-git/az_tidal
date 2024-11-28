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

url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_wave_0p25.pl?dir=%2Fgefs.20240820%2F00%2Fwave%2Fgridded&file=gefs.wave.t00z.p01.global.0p25.f000.grib2&var_HTSGW=on&all_lev=on&subregion=&toplat=30&leftlon=275&rightlon=285&bottomlat=20"

for bdate in [bdate2, bdate1, date]:
    if bdate == bdate2:
        for hour in ['06', '12', '18']:
            for i in range(1, runs+1):
                member = f"p{i:02d}"
                url = (
                      f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_wave_0p25.pl?dir=%2Fgefs.{bdate}%2F{hour}%2Fwave%2Fgridded&"
                      f"file=gefs.wave.t{hour}z.{member}.global.0p25.f000.grib2&var_HTSGW=on&"
                      f"all_lev=on&subregion=&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
                      )
                filename = f"gefs.wave.{member}.{bdate}.t{hour}z.f000.grib2"
                download_file(url, filename)
    elif bdate == bdate1:
        for hour in ['00', '06', '12', '18']:
            for i in range(1, runs+1):
                member = f"p{i:02d}"
                url = (
                      f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_wave_0p25.pl?dir=%2Fgefs.{bdate}%2F{hour}%2Fwave%2Fgridded&"
                      f"file=gefs.wave.t{hour}z.{member}.global.0p25.f000.grib2&var_HTSGW=on&"
                      f"all_lev=on&subregion=&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
                      )
                filename = f"gefs.wave.{member}.{bdate}.t{hour}z.f000.grib2"
                download_file(url, filename)
    else:
        for hour in ['00']:
            for i in range(1, runs+1):
                member = f"p{i:02d}"
                url = (
                      f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_wave_0p25.pl?dir=%2Fgefs.{bdate}%2F{hour}%2Fwave%2Fgridded&"
                      f"file=gefs.wave.t{hour}z.{member}.global.0p25.f000.grib2&var_HTSGW=on&"
                      f"all_lev=on&subregion=&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
                      )
                filename = f"gefs.wave.{member}.{bdate}.t{hour}z.f000.grib2"
                download_file(url, filename)

# Base URL for the data source
base_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_wave_0p25.pl?dir=%2Fgefs.{date}%2F{cycle}%2Fwave%2Fgridded&"
# Loop through forecast hours and download data
for i in range(0, 242, 3):
    fhr = f"{i:03d}"  # Convert to 3-digit string
    for i in range(1, runs+1):
        member = f"p{i:02d}"
        # Construct the URL for this forecast hour
        url = (
            f"{base_url}"
            f"file=gefs.wave.t{cycle}z.{member}.global.0p25.f{fhr}.grib2&var_HTSGW=on&"
            f"all_lev=on&subregion=&toplat={lat2}&leftlon={lon1}&rightlon={lon2}&bottomlat={lat1}"
        )
        
        # Construct the filename for this forecast hour
        filename = f"gefs.wave.{member}.{date}.t{cycle}z.f{fhr}.grib2"
        
        # Download the file
        download_file(url, filename)
    
