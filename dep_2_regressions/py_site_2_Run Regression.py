import os
import platform
import numpy as np
import pandas as pd
import cdsapi
import requests
from datetime import datetime
from io import StringIO  # This is the correct import for StringIO
import calendar
import matplotlib.pyplot as plt
import argparse  # <-- Add this line'
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm

#------------------------------------------------------------------------------
#
#                 SET BASE DIRECTORY
#
#------------------------------------------------------------------------------

# Define the base directory based on the hostname
hostname = platform.node()
if hostname == 'Nates-MacBook-Pro.local' or hostname == 'UM-C02RQ1S8FVH7' or hostname == 'nates-mbp.lan' or hostname == 'Nates-MBP.lan':
    BASE_DIR = '/Users/nate/Documents/tidal_work/'
elif hostname == 'mango.rsmas.miami.edu':
    BASE_DIR = '/home/ntaminger/tidal_work/'
elif hostname == 'aztidal01p':
    BASE_DIR = '/home/ntaminge/tidal_work/'
else:
    raise ValueError(f"Unknown hostname: {hostname}")

os.chdir(BASE_DIR)
#------------------------------------------------------------------------------
#
#                 SET DIRECTORIES
#
#------------------------------------------------------------------------------

input_dir = os.path.join(BASE_DIR, '0_station_info')
norm_dir = os.path.join(BASE_DIR, '2_regressions', 'normalization_data')
data_dir = os.path.join(BASE_DIR, '2_regressions', 'input_data')
output_all = os.path.join(BASE_DIR, '2_regressions', 'regression_all_data')
output_use = os.path.join(BASE_DIR, '2_regressions', 'regression_use_data')
plot_dir = os.path.join(BASE_DIR, '4_plots')

#------------------------------------------------------------------------------
#
#                 LOAD STATION DATA FROM CSV
#
#------------------------------------------------------------------------------

# Load station-specific data from the site-specific CSV file
csv_file = os.path.join(input_dir, 'station_data.csv')
df = pd.read_csv(csv_file)

#------------------------------------------------------------------------------
#
#                 SET THE SITE
#
#------------------------------------------------------------------------------

site = 'virginia_key'  # Change this as needed for different sites

# # Argument parsing
# parser = argparse.ArgumentParser(description="Run the script for a specific site")
# parser.add_argument('--site', required=True, help='The site to run the script for (e.g., fort_myers)')
# args = parser.parse_args()
# 
# # Use the provided site argument
# site = args.site

site_data = df[df['site_name'] == site]
if site_data.empty:
    raise ValueError(f"Site '{site_name}' not found in the dataset.")

#------------------------------------------------------------------------------
#
#                 DEFINE VARIABLES
#
#------------------------------------------------------------------------------

site_name = site_data['site_name'].iloc[0]
station_ID = site_data['station_ID'].iloc[0]
station_abbr = site_data['station_abbr'].iloc[0]
station_name = site_data['station_name'].iloc[0]
wlon = site_data['wlon'].iloc[0]
elon = site_data['elon'].iloc[0]
slat = site_data['slat'].iloc[0]
nlat = site_data['nlat'].iloc[0]
startyear = site_data['start_year'].iloc[0]
endyear = datetime.now().year - 1

#------------------------------------------------------------------------------
#
#                 GET DATA AND ADD ROLLING AVERAGES
#
#------------------------------------------------------------------------------

# Get Data and Add Rolling Averages
df = pd.read_csv(f"{data_dir}/{station_ID}_regression_input.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.drop(columns=['slp_sd'])  # Assuming 'X' is an unnecessary column

# Calculate RMSE
dep_rmse = np.sqrt((df['departure_adj'] ** 2).mean())
# Print the result
print(f"Root Mean Squared Error (RMSE): {dep_rmse}")

# Rolling averages (using a simple moving average as an approximation)
df['u10_sd'] = df['u10_sd'].rolling(window=3, center=False, closed='right').mean()
df['v10_sd'] = df['v10_sd'].rolling(window=3, center=False, closed='right').mean()
df['sst_sd'] = df['sst_sd'].rolling(window=3, center=False, closed='right').mean()
df['swh_sd'] = df['swh_sd'].rolling(window=3, center=False, closed='right').mean()

all_model = pd.DataFrame()  # Initialize an empty DataFrame

i = 1
for i in range(1, 11):
    print(f"building {i} day model")

    # Lagged variables
    lag_df = df.copy()
    lag_df['departure_last'] = lag_df['departure_adj'].shift(i)
    lag_df['slp_lag'] = lag_df['slp'].shift(i)
    lag_df['pressure_diff'] = lag_df['slp'] - lag_df['slp_lag']
    lag_df['pressure_removed'] = lag_df['departure_adj'] - (lag_df['pressure_diff']) * -0.0328084

    # lag_df.to_csv(f"{pathMast}lag_{i}_day.csv", index=False)

    reg_df = lag_df.dropna(subset=['departure_last', 'slp_lag', 'pressure_diff', 'pressure_removed'])
    reg_df = reg_df[['departure_last', 'u10_sd', 'v10_sd', 'sst_sd', 'swh_sd', 'pressure_removed']]

    # Try 1: Single Linear Regression
    model = sm.ols(formula="pressure_removed ~ departure_last + u10_sd + v10_sd + sst_sd + swh_sd", data=reg_df).fit()
    coefs = model.params
    r_sq = model.rsquared_adj
    
    df_model_1 = pd.DataFrame({
        'dep_coef': [coefs['departure_last']],
        'dep_pval': [model.pvalues['departure_last']],
        'u10_coef': [coefs['u10_sd']],
        'u10_pval': [model.pvalues['u10_sd']],
        'v10_coef': [coefs['v10_sd']],
        'v10_pval': [model.pvalues['v10_sd']],
        'sst_coef': [coefs['sst_sd']],
        'sst_pval': [model.pvalues['sst_sd']],
        'swh_coef': [coefs['swh_sd']],
        'swh_pval': [model.pvalues['swh_sd']],
        'r2': [r_sq]
    })

    # Try 2: Monte Carlo Simulation
    dep_coef = []
    u10_coef = []
    v10_coef = []
    sst_coef = []
    swh_coef = []

    dep_pval = []
    u10_pval = []
    v10_pval = []
    sst_pval = []
    swh_pval = []

    rmse = []
    r2 = []

    for x in range(10000):
        if x % 1111 == 0:
            print(x)

        train, test = train_test_split(reg_df, test_size=0.2)
        model_all = sm.ols(formula="pressure_removed ~ departure_last + u10_sd + v10_sd + sst_sd + swh_sd", data=train).fit()

        dep_coef.append(model_all.params['departure_last'])
        u10_coef.append(model_all.params['u10_sd'])
        v10_coef.append(model_all.params['v10_sd'])
        sst_coef.append(model_all.params['sst_sd'])
        swh_coef.append(model_all.params['swh_sd'])

        dep_pval.append(model_all.pvalues['departure_last'])
        u10_pval.append(model_all.pvalues['u10_sd'])
        v10_pval.append(model_all.pvalues['v10_sd'])
        sst_pval.append(model_all.pvalues['sst_sd'])
        swh_pval.append(model_all.pvalues['swh_sd'])

        pred = model_all.predict(test)
        rmse_val = np.sqrt(np.mean((pred - test['pressure_removed']) ** 2))
        rmse.append(rmse_val)

        SSE = np.sum((pred - test['pressure_removed']) ** 2)
        SST = np.sum((test['pressure_removed'] - np.mean(test['pressure_removed'])) ** 2)
        r2_test = 1 - SSE / SST
        r2.append(r2_test)

    df_model_2 = pd.DataFrame({
        'dep_coef': dep_coef,
        'dep_pval': dep_pval,
        'u10_coef': u10_coef,
        'u10_pval': u10_pval,
        'v10_coef': v10_coef,
        'v10_pval': v10_pval,
        'sst_coef': sst_coef,
        'sst_pval': sst_pval,
        'swh_coef': swh_coef,
        'swh_pval': swh_pval,
        'rmse': rmse,
        'r2': r2
    })

    # Calculate means for Monte Carlo results
    df_model_3 = pd.DataFrame({
        'dep_coef': [np.mean(df_model_2['dep_coef'])],
        'dep_pval': [np.mean(df_model_2['dep_pval'])],
        'u10_coef': [np.mean(df_model_2['u10_coef'])],
        'u10_pval': [np.mean(df_model_2['u10_pval'])],
        'v10_coef': [np.mean(df_model_2['v10_coef'])],
        'v10_pval': [np.mean(df_model_2['v10_pval'])],
        'sst_coef': [np.mean(df_model_2['sst_coef'])],
        'sst_pval': [np.mean(df_model_2['sst_pval'])],
        'swh_coef': [np.mean(df_model_2['swh_coef'])],
        'swh_pval': [np.mean(df_model_2['swh_pval'])],
        'rmse': [np.mean(df_model_2['rmse'])],
        'r2': [np.mean(df_model_2['r2'])]
    })

    temp_model = df_model_3.copy()
    temp_model['day'] = i
    all_model = pd.concat([all_model, temp_model], ignore_index=True)

all_model["noaa_slr_rmse"] = dep_rmse
all_model["model_improvement"] = abs(( all_model["noaa_slr_rmse"] - all_model["rmse"]) / all_model["noaa_slr_rmse"])
round_model = all_model.round(4)
save_model = round_model[['day', 'rmse', 'r2', 'model_improvement', 'noaa_slr_rmse',
                        'dep_coef', 'u10_coef', 'v10_coef', 'sst_coef', 'swh_coef',
                        'dep_pval', 'u10_pval', 'v10_pval', 'sst_pval', 'swh_pval']]
use_model = round_model[['day', 'dep_coef', 'u10_coef', 'v10_coef', 'sst_coef', 'swh_coef']]

# Save results
save_model.to_csv(f"{output_all}/{station_ID}_all_data.csv", index=False)
use_model.to_csv(f"{output_use}/{station_ID}_use_data.csv", index=False)

