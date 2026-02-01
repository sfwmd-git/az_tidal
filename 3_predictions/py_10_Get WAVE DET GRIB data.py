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
domain = 'atlocn'

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    time.sleep(2)

today = datetime.date.today() - datetime.timedelta(days=db)
bdate2 = (today - datetime.timedelta(days=2)).strftime('%Y%m%d')
bdate1 = (today - datetime.timedelta(days=1)).strftime('%Y%m%d')
date = today.strftime('%Y%m%d')
time_cycle = 0  # 0, 6, 12, 18
cycle = f"{time_cycle:02d}"  # Convert to 2-digit string

base_url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.'

for bdate in [bdate2, bdate1]:
    for hour in ['06', '12', '18']:
        url = f'{base_url}{bdate}/{hour}/wave/gridded/gfswave.t{hour}z.{domain}.0p16.f000.grib2'
        filename = f'gfswave.{bdate}.t{hour}z.f000.grib2'
        download_file(url, filename)

url = f'{base_url}{bdate1}/00/wave/gridded/gfswave.t00z.{domain}.0p16.f000.grib2'
filename = f'gfswave.{bdate1}.t00z.f000.grib2'
download_file(url, filename)

url = f'{base_url}{date}/00/wave/gridded/gfswave.t00z.{domain}.0p16.f000.grib2'
filename = f'gfswave.{date}.t00z.f000.grib2'
download_file(url, filename)

base_url += f'{date}/{cycle}/wave/gridded/'
for idx, file_range in enumerate([range(0, 121, 3), range(123, 244, 3)]):
    for file_num in file_range:
        url = f'{base_url}gfswave.t{cycle}z.{domain}.0p16.f{file_num:03d}.grib2'
        filename = f'gfswave.{date}.t{cycle}z.f{file_num:03d}.grib2'
        download_file(url, filename)
