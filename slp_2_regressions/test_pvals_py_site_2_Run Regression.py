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
output_all = os.path.join(BASE_DIR, 'test_pvals_2_regressions', 'regression_all_data')
output_use = os.path.join(BASE_DIR, 'test_pvals_2_regressions', 'regression_use_data')
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
df = df.drop(columns=['slp'])  # Assuming 'X' is an unnecessary column

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
for i in range(1, 11):
    print(f"Building {i}-day model")
    
    lag_df = df.copy()
    lag_df['departure_last'] = lag_df['departure_adj'].shift(i)
    reg_df = lag_df.dropna(subset=['departure_adj', 'departure_last'])
    
    predictors = {
        'departure_last': 'dep',
        'slp_sd': 'slp',
        'u10_sd': 'u10',
        'v10_sd': 'v10',
        'sst_sd': 'sst',
        'swh_sd': 'swh'
    }
    removed_vars = []
    
    # Monte Carlo Simulation
    mc_results = {var: [] for var in predictors.keys()}
    mc_results.update({'rmse': [], 'r2': []})
    mc_pvals = {var: [] for var in predictors.keys()}
    
    for _ in range(100):
        train, test = train_test_split(reg_df, test_size=0.2)
        formula = "departure_adj ~ " + " + ".join(predictors.keys())
        mc_model = sm.ols(formula=formula, data=train).fit()
        
        for var in predictors.keys():
            mc_results[var].append(mc_model.params[var])
            mc_pvals[var].append(mc_model.pvalues[var])
        
        pred = mc_model.predict(test)
        rmse_val = np.sqrt(np.mean((pred - test['departure_adj']) ** 2))
        mc_results['rmse'].append(rmse_val)
        
        SSE = np.sum((pred - test['departure_adj']) ** 2)
        SST = np.sum((test['departure_adj'] - np.mean(test['departure_adj'])) ** 2)
        mc_results['r2'].append(1 - SSE / SST)
    
    # Compute means
    final_results = {predictors[var] + '_coef': np.mean(mc_results[var]) for var in predictors.keys()}
    final_pvals = {predictors[var] + '_pval': np.mean(mc_pvals[var]) for var in predictors.keys()}
    final_results.update({'rmse': np.mean(mc_results['rmse']), 'r2': np.mean(mc_results['r2']), 'day': i})
    final_results.update(final_pvals)
    
    # Iteratively remove insignificant variables
    while any(pval > 0.05 for pval in final_pvals.values()):
        max_pval_var = max(final_pvals, key=final_pvals.get)
        if final_pvals[max_pval_var] <= 0.05:
            break
        
        removed_var = [key for key, val in predictors.items() if val + '_pval' == max_pval_var][0]
        predictors.pop(removed_var)
        removed_vars.append(removed_var)
        print(f"Day {i}: Removing variable {removed_var} due to high p-value")
        
        # Re-run Monte Carlo with reduced predictors
        mc_results = {var: [] for var in predictors.keys()}
        mc_results.update({'rmse': [], 'r2': []})
        mc_pvals = {var: [] for var in predictors.keys()}
        
        for _ in range(100):
            train, test = train_test_split(reg_df, test_size=0.2)
            formula = "departure_adj ~ " + " + ".join(predictors.keys())
            mc_model = sm.ols(formula=formula, data=train).fit()
            
            for var in predictors.keys():
                mc_results[var].append(mc_model.params[var])
                mc_pvals[var].append(mc_model.pvalues[var])
            
            pred = mc_model.predict(test)
            rmse_val = np.sqrt(np.mean((pred - test['departure_adj']) ** 2))
            mc_results['rmse'].append(rmse_val)
            
            SSE = np.sum((pred - test['departure_adj']) ** 2)
            SST = np.sum((test['departure_adj'] - np.mean(test['departure_adj'])) ** 2)
            mc_results['r2'].append(1 - SSE / SST)
        
        final_results = {predictors[var] + '_coef': np.mean(mc_results[var]) for var in predictors.keys()}
        final_pvals = {predictors[var] + '_pval': np.mean(mc_pvals[var]) for var in predictors.keys()}
        final_results.update({'rmse': np.mean(mc_results['rmse']), 'r2': np.mean(mc_results['r2']), 'day': i})
        final_results.update(final_pvals)
    
    # Add back removed variables with 0 coef and 'N/A' p-value
    for var in removed_vars:
        var_name = predictors[var] if var in predictors else predictors.get(var, var.replace('_sd', ''))
        final_results[var_name + '_coef'] = 0
        final_results[var_name + '_pval'] = np.nan
    
    all_model = pd.concat([all_model, pd.DataFrame([final_results])], ignore_index=True)

# Reorder columns to match the required order
column_order = [
    'dep_coef', 'dep_pval', 'slp_coef', 'slp_pval', 'u10_coef', 'u10_pval',
    'v10_coef', 'v10_pval', 'sst_coef', 'sst_pval', 'swh_coef', 'swh_pval',
    'rmse', 'r2', 'day'
]
all_model = all_model[column_order]

print(f"Final Model for {station_name} Complete")


all_model["noaa_slr_rmse"] = dep_rmse
all_model["model_improvement"] = abs(( all_model["noaa_slr_rmse"] - all_model["rmse"]) / all_model["noaa_slr_rmse"])
round_model = all_model.round(4)
save_model = round_model[['day', 'rmse', 'r2', 'model_improvement', 'noaa_slr_rmse',
                        'dep_coef', 'slp_coef', 'u10_coef', 'v10_coef', 'sst_coef', 'swh_coef',
                        'dep_pval', 'slp_pval', 'u10_pval', 'v10_pval', 'sst_pval', 'swh_pval']]
use_model = round_model[['day', 'dep_coef', 'slp_coef', 'u10_coef', 'v10_coef', 'sst_coef', 'swh_coef']]

# Save results
save_model.to_csv(f"{output_all}/{station_ID}_all_data.csv", index=False)
use_model.to_csv(f"{output_use}/{station_ID}_use_data.csv", index=False)

