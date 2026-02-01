# Tidal Wave Prediction System

A comprehensive system for predicting tidal water levels using meteorological and oceanographic data from multiple forecast models (GFS and ECMWF) with various model configurations.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Workflow Pipeline](#workflow-pipeline)
- [Setup](#setup)
- [Usage](#usage)
- [Key Concepts](#key-concepts)
- [Model Variants](#model-variants)
- [Forecast Types](#forecast-types)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This system predicts tidal water levels at multiple NOAA tide gauge stations along the Florida coast by:

1. **Collecting** historical tide data and meteorological reanalysis data (ERA5)
2. **Cleaning** and processing the raw data
3. **Training** regression models that relate atmospheric/oceanic conditions to water level departures
4. **Generating** predictions using operational forecast data (GFS and ECMWF)
5. **Validating** model performance

The system supports multiple model variants and forecast types, allowing for comparison and ensemble approaches.

## Project Structure

```
tidal_work/
├── 0_station_info/          # Station metadata and configuration
│   ├── station_data.csv      # Master station information file
│   └── logos/                # Station logos for plots
│
├── 1_data/                   # Data collection and cleaning
│   ├── raw_water_data/        # Raw tide gauge data from NOAA
│   ├── raw_era5_data/        # Raw ERA5 reanalysis data
│   ├── clean_water_data/     # Processed tide data
│   ├── clean_era5_data/      # Processed ERA5 data
│   ├── forecasts/            # Operational forecast data (GFS/ECMWF)
│   └── py_*.py               # Data collection/cleaning scripts
│
├── 2_regressions/            # Base regression models
│   ├── input_data/           # Prepared regression input data
│   ├── normalization_data/   # Normalization parameters
│   ├── regression_all_data/  # Full regression results
│   ├── regression_use_data/  # Regression coefficients for predictions
│   └── py_*.py               # Regression scripts
│
├── dep_2_regressions/        # DEP variant regression models
├── slp_2_regressions/         # SLP variant regression models
│
├── 3_predictions/            # Base model predictions
│   ├── GFS_DET_*/            # GFS deterministic forecasts
│   ├── GFS_ENS_*/            # GFS ensemble forecasts
│   ├── ECMWF_DET_*/          # ECMWF deterministic forecasts
│   ├── ECMWF_ENS_*/          # ECMWF ensemble forecasts
│   └── py_*.py               # Prediction scripts
│
├── dep_3_predictions/        # DEP variant predictions
├── slp_3_predictions/         # SLP variant predictions
│
├── 4_plots/                  # Generated plots and visualizations
│
├── 5_shell_scripts/          # Automation scripts
│   ├── 1_print_sites.sh      # List all stations
│   ├── 2_update_regression.sh # Update regression models
│   ├── 3_update_slr.sh       # Update sea level rise adjustments
│   ├── 4_run_gfs_deterministic.sh
│   ├── 5_run_gfs_ensemble.sh
│   ├── 6_run_ecmwf_determinstic.sh
│   ├── 7_run_ecmwf_ensemble.sh
│   └── x_run_validation_analysis.sh
│
└── x_validation_analysis/     # Model validation and analysis
```

## Workflow Pipeline

The system follows a numbered workflow:

### Step 0: Station Information
- Station metadata stored in `0_station_info/station_data.csv`
- Contains station IDs, coordinates, flood thresholds, and forecast grid indices

### Step 1: Data Collection & Cleaning
1. **Get Tide Data** (`py_1_Get Tide Data.py`)
   - Downloads historical water level observations and predictions from NOAA API
   - Stores in `1_data/raw_water_data/`

2. **Get ERA5 Data** (`py_2_Get ERA5 Data.py`)
   - Downloads ERA5 reanalysis data (wind, pressure, SST, waves)
   - Stores in `1_data/raw_era5_data/`

3. **Clean Tide Data** (`py_3_Clean Tide Data.py`)
   - Processes raw tide data, calculates departures from predictions
   - Outputs to `1_data/clean_water_data/`

4. **Clean ERA5 Data** (`py_4_Clean ERA5 Data.py`)
   - Processes ERA5 data, extracts relevant variables
   - Outputs to `1_data/clean_era5_data/`

### Step 2: Regression Modeling
1. **Clean Regression Data** (`py_1_Clean Regression Data.py`)
   - Combines tide and ERA5 data
   - Creates normalized input datasets
   - Outputs to `2_regressions/input_data/` and `normalization_data/`

2. **Run Regression** (`py_2_Run Regression.py`)
   - Trains regression models for 1-10 day lead times
   - Uses Monte Carlo cross-validation (10,000 iterations)
   - Outputs coefficients to `regression_use_data/` and full results to `regression_all_data/`

### Step 3: Generate Predictions
1. **Get Forecast Data** (`py_00_Get GFS DET GRIB data.py`, etc.)
   - Downloads operational forecast data (GRIB files)
   - Stores in `1_data/forecasts/`

2. **Calculate Predictions** (`py_a0_Calc GFS DET.py`, etc.)
   - Applies regression coefficients to forecast data
   - Generates water level predictions
   - Outputs organized by forecast type and station

3. **Combine Data** (`py_w0_Get Data Together.py`)
   - Aggregates predictions across stations
   - Creates master datasets

### Step 4: Visualization
- Plots generated in `4_plots/`
- Shows water level adjustments and predictions

## Setup

### Prerequisites

- Python 3.7+
- Required Python packages:
  - `pandas`
  - `numpy`
  - `netCDF4`
  - `xarray`
  - `cdsapi` (for ERA5 data)
  - `pygrib` (for GRIB file reading)
  - `matplotlib`
  - `scikit-learn`
  - `statsmodels`
  - `requests`
  - `pytz`

### Configuration

The system uses hostname-based path detection. To add a new machine:

1. Find your hostname: `python3 -c "import platform; print(platform.node())"`
2. Edit the relevant Python scripts to add your hostname and base directory path
3. Or set up a consistent directory structure across machines

**Current supported hostnames:**
- `Nates-MacBook-Pro.local`, `UM-C02RQ1S8FVH7`, `nates-mbp.lan`, `Nates-MBP.lan` → `/Users/nate/Documents/tidal_work/`
- `mango.rsmas.miami.edu` → `/home/ntaminger/tidal_work/`
- `aztidal01p` → `/home/ntaminge/tidal_work/`

### API Keys

Some data sources require API keys:

- **ERA5/CDS API**: Requires registration at https://cds.climate.copernicus.eu/
- **NOAA Tides API**: Public, no key required

## Usage

### Running Individual Scripts

All scripts use a consistent command-line interface with the `--site` argument:

```bash
# Get tide data for a specific station
python3 1_data/py_1_Get\ Tide\ Data.py --site virginia_key

# Clean regression data
python3 2_regressions/py_1_Clean\ Regression\ Data.py --site virginia_key

# Run regression
python3 2_regressions/py_2_Run\ Regression.py --site virginia_key

# Generate predictions
python3 3_predictions/py_a0_Calc\ GFS\ DET.py --site virginia_key
```

### Using Shell Scripts

Shell scripts automate workflows across all stations:

```bash
# Update regression models for all stations
cd 5_shell_scripts
./2_update_regression.sh

# Run GFS deterministic predictions for all stations
./4_run_gfs_deterministic.sh
```

**Note:** Before running shell scripts, update the `BASE_DIR` variable in each script to match your system.

### Available Stations

Current stations (from `station_data.csv`):
- `virginia_key` (8723214)
- `lake_worth` (8722670)
- `key_west` (8724580)
- `vaca_key` (8723970)
- `port_everglades` (8722956)
- `naples` (8725110)
- `naples_bay` (8725114)
- `fort_myers` (8725520)

## Key Concepts

### Water Level Departures

The system predicts **departures** from astronomical tide predictions:
- **Departure** = Observed water level - Predicted astronomical tide
- Departures are caused by meteorological and oceanographic forcing (wind, pressure, waves)

### Sea Level Rise

The system calculates **SLR** from past observations:
- **SLR** = Average departure of rolling_window set in py_3_Clean Tide Data in 1_data
- SLR is removed from Departure before building the regression

### Regression Models

The regression models relate atmospheric/oceanic variables to water level departures:
- **Predictors**: u10 (zonal wind), v10 (meridional wind), SST (sea surface temperature), SWH (significant wave height), previous departure
- **Target**: Water level departure (with pressure effects & SLR removed)
- **Lead times**: Models trained for 1-10 day forecasts

### Monte Carlo Cross-Validation

Regression models use Monte Carlo cross-validation:
- 10,000 random train/test splits (80/20)
- Provides robust coefficient estimates and uncertainty quantification

## Model Variants

The system supports multiple model variants:

### Base Model (`2_regressions/`, `3_predictions/`)
- Standard regression model
- Uses all available predictors

### DEP Variant (`dep_2_regressions/`, `dep_3_predictions/`)
- Modified regression approach
- Different treatment of departure variables

### SLP Variant (`slp_2_regressions/`, `slp_3_predictions/`)
- Sea level pressure-focused variant
- Alternative pressure handling

**Note:** Variants are implemented by duplicating directory structures with different prefixes. The `experiment` variable in prediction scripts controls which variant is used.

## Forecast Types

### GFS (Global Forecast System)
- **DET**: Deterministic forecast (single run)
- **ENS**: Ensemble forecast (multiple members)

### ECMWF (European Centre for Medium-Range Weather Forecasts)
- **DET**: Deterministic forecast
- **ENS**: Ensemble forecast

### Output Organization

For each forecast type, outputs are organized as:
- `*_all/`: Combined predictions across all stations
- `*_components/`: Individual component contributions
- `*_master/`: Master datasets with metadata
- `*_graphs/`: Visualization outputs

## Troubleshooting

### Common Issues

1. **"Unknown hostname" error**
   - Add your hostname to the allowed list in the Python script
   - Or modify the BASE_DIR logic to use environment variables

2. **Missing data files**
   - Ensure data collection scripts have been run
   - Check that API keys are configured (for ERA5)

3. **Path errors**
   - Verify BASE_DIR is set correctly in shell scripts
   - Ensure directory structure matches expected layout

4. **Import errors**
   - Install missing Python packages: `pip install <package_name>`
   - Check Python version compatibility

5. **GRIB file reading errors**
   - Ensure `pygrib` is properly installed
   - Check that forecast data files are valid GRIB format

^. **Timeout erros**
   - GFS data download endpoint can time out
   - Is rare, but prediction will not be made for that day
   - Just rerun the GFS scripts and the put data together script
   
### Getting Help

- Check script comments for configuration options
- Review error messages for specific file/path issues
- Verify station names match `station_data.csv` exactly

## Contributing

### Adding a New Station

1. Add station information to `0_station_info/station_data.csv`
2. Run data collection scripts for the new station
3. Run regression training
4. Update shell scripts if needed

### Adding a New Model Variant

1. Create new directories: `{variant}_2_regressions/` and `{variant}_3_predictions/`
2. Copy and modify regression scripts
3. Update prediction scripts with `experiment = "{variant}"`
4. Update shell scripts to include new variant

### Code Style

- Use descriptive variable names
- Include comments for complex logic
- Follow existing script structure and naming conventions
- Use `argparse` for command-line arguments

## Additional Resources

- **NOAA Tides API**: https://api.tidesandcurrents.noaa.gov/api/prod/
- **ERA5/CDS**: https://cds.climate.copernicus.eu/
- **GFS Data**: https://www.ncei.noaa.gov/ products/weather-climate-models/global-forecast
- **ECMWF Data**: https://www.ecmwf.int/

---

**Last Updated:** 2025
**Maintainers:** Nate

