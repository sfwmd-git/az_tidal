# AZ_Tidal Prediction Project

This project automates the creation of 10-day tidal predictions for seven sites in South Florida using meteorological datasets, including the GFS and ECMWF deterministic and ensemble forecasts. The predictions are shared with the South Florida Water Management District (SFWMD) for daily updates on their website.

## Project Overview

The project uses a **multilinear regression model** trained on past environmental conditions and water level observations to create predictions. It:
- Adjusts NOAA's predictions for sea level rise.
- Incorporates upcoming 10-day forecasts.
- Generates maximum daily water level predictions.
- Interpolates values between NOAA's six-minute predictions and the calculated maximums to create a six-minute tide prediction.

The workflow automates:
1. Data retrieval and processing.
2. Prediction generation.
3. Visualization of forecasts.
4. Upload of forecast data to the SFWMD's SFTP server.

---

## Key Features

1. **Prediction Automation**: Automatically generates 10-day tidal forecasts for seven sites across South Florida.
2. **Data Consolidation**: Combines NOAA and GFS/ECMWF datasets into master CSVs and visual PNG outputs.
3. **SFTP Integration**: Uploads data to the SFWMD's server for visualization.
4. **Customizable**: Supports deterministic and ensemble forecasts for multiple meteorological datasets.

---

## Requirements

### Python Libraries
The following Python libraries are required:
- os
- platform
- pandas
- numpy
- datetime
- time
- pytz
- requests
- pygrib
- matplotlib
- argparse
- glob
- math

#### Install Required Libraries:
Run this command to install the required libraries:
```bash
pip install pandas numpy matplotlib pytz requests pygrib


## Additional System Requirements

### `pygrib`
Requires `eccodes` to handle GRIB files.
Install using:
```bash
conda install -c conda-forge pygrib
```

### SFTP Configuration
The system must have an SSH key configured to upload files to `sftp.sfwmd.gov` using the username `nht8`.

Test the connection with:
```bash
sftp nht8@sftp.sfwmd.gov
```

## Setup Instructions

1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/az_tidal.git
    ```
2. Navigate to the project directory:
    ```bash
    cd az_tidal
    ```
3. Install the required Python libraries (see above).
4. Ensure the SFTP configuration is complete for automated uploads.

---

## How to Run the Project

The project runs automatically via a `crontab` schedule. The crontab executes a series of shell scripts, which in turn call the Python scripts to process data, generate predictions, and upload results. The time is in UTC for all sites.

### Workflow Order
1. **`delete_forecasts.sh`** (12:00 AM): Deletes outdated forecast files.
7. **`run_gfs_det.sh`** (5:00 AM): Runs deterministic GFS forecasts for all sites.
8. **`run_gfs_ens.sh`** (5:10 AM): Runs GFS ensemble forecasts (~4 hours).
2. **`download_all_euro.sh`** (7:45 AM): Downloads ECMWF forecast data.
3. **`run_ecmwf_det.sh`** (7:50 AM): Runs deterministic ECMWF forecasts for all sites.
4. **`run_ecmwf_ens_eastfl.sh`** (8:00 AM): Processes ECMWF ensemble forecasts for East Florida sites.
5. **`run_ecmwf_ens_westfl.sh`** (8:00 AM): Processes ECMWF ensemble forecasts for West Florida sites.
6. **`run_ecmwf_ens_flkeys.sh`** (8:00 AM): Processes ECMWF ensemble forecasts for Florida Keys sites.
9. **`get_data_together.sh`** (10:53 AM): Consolidates forecast data into master CSVs.
10. **`put_aztidal.sh`** (10:55 AM): Uploads master CSVs and site-specific files to the SFWMD's SFTP server.

---

## Directory Structure. Ignore for noq

The project files are structured as follows:
```
az_tidal/
├── delete_forecasts.sh      # Deletes outdated forecast files
├── download_all_euro.sh     # Downloads ECMWF data
├── get_data_together.sh     # Combines forecast data into master CSVs
├── run_ecmwf_det.sh         # Runs deterministic ECMWF forecasts
├── run_ecmwf_ens_eastfl.sh  # Runs ECMWF ensemble forecasts for East Florida
├── run_ecmwf_ens_flkeys.sh  # Runs ECMWF ensemble forecasts for Florida Keys
├── run_ecmwf_ens_westfl.sh  # Runs ECMWF ensemble forecasts for West Florida
├── run_gfs_det.sh           # Runs deterministic GFS forecasts
├── run_gfs_ens.sh           # Runs ensemble GFS forecasts
├── put_aztidal.sh           # Uploads results to SFTP server
├── py_00_Get GFS DET GRIB data.py  # Pulls GFS deterministic data
├── py_01_Get GFS ENS GRIB data.py  # Pulls GFS ensemble data
├── py_10_Get WAVE DET GRIB data.py # Pulls wave height data (deterministic)
├── py_11_Get WAVE ENS GRIB data.py # Pulls wave height data (ensemble)
├── py_a0_Calc GFS DET.py           # Calculates GFS deterministic predictions
├── py_b0_Calc GFS ENS.py           # Calculates GFS ensemble predictions
├── py_c0_Calc ECMWF DET.py         # Calculates ECMWF deterministic predictions
├── py_d0_Calc ECMWF ENS.py         # Calculates ECMWF ensemble predictions
├── py_w_Get Data Together.py       # Combines forecasts into master CSV
└── py_x_Delete Grib.py             # Deletes old GRIB files
```

---

## Contributors

This project was developed by **[Nate Taminger/RSMAES/SFWMD]**. Please contact us for questions or support.

---
