# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:53:31 2023

@author: carsk
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:53:31 2023

@author: carsk
"""

import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from tqdm import tqdm
import os
from rasterio.mask import mask
from datetime import datetime, timedelta

# Load the shapefile, rasters, and CSV file
basin_gdf = gpd.read_file(r"C:\Users\carsk\OneDrive - KU Leuven\Thesis\RESULTS\MOL5\basin.shp")
TIA_20m = rasterio.open(r"C:\Users\carsk\OneDrive - KU Leuven\Thesis\RESULTS\INPUTS\TIA_20mf.tif")
LAI_max_20m = rasterio.open(r"C:\Users\carsk\OneDrive - KU Leuven\Thesis\RESULTS\INPUTS\LAI_20mf.tif")
precip_df = pd.read_csv(r"C:\Users\carsk\OneDrive - KU Leuven\Thesis\RESULTS\INPUTS\9modeledStns.csv", parse_dates=['date'])


# Get the geometry of the shapefile in GeoJSON format and mask the TIA and LAI rasters
basin_geom = basin_gdf.geometry.to_crs(TIA_20m.crs)
TIA_array, _ = mask(TIA_20m, basin_geom, crop=True, nodata=TIA_20m.nodata)
LAI_array, _ = mask(LAI_max_20m, basin_geom, crop=True, nodata=LAI_max_20m.nodata)

# Squeeze the arrays to remove the unnecessary dimension and set nodata values to NaN
TIA_array = np.squeeze(TIA_array).astype('float32')
LAI_array = np.squeeze(LAI_array).astype('float32')
TIA_array[TIA_array == TIA_20m.nodata] = np.nan
LAI_array[LAI_array == LAI_max_20m.nodata] = np.nan


# User input for station and date range
station = input('Enter the station: ')
start_date = input('Enter the start date (MM/DD/YYYY): ')
end_date = input('Enter the end date (MM/DD/YYYY): ')

# Filter the precipitation data based on user input
precip_df = precip_df[(precip_df['date'] >= start_date) & (precip_df['date'] <= end_date) & (precip_df[station] > 0)][['date', station]]

# Function to calculate Pnet based on LAI and precipitation
def calculate_pnet(LAI, P):
    if np.isnan(LAI) or np.isnan(P):
        return np.nan

    if LAI < 1:
        Pnet = 0.04 + 0.99 * (P*12) - 0.09 * LAI
    else:
        if P < 0.6:
            Pnet = 0.03 + 0.72 * (P*12) - 0.05 * np.log(LAI)
        else:
            Pnet = -0.02 + 0.98 * (P*12) - 0.09 * LAI
    
    return min(Pnet, P*12)

# Read LAI raster as a numpy array
#LAI_array = LAI_max_20m.read(1)
#LAI_array[LAI_array < 0] = 0  # Set values less than 0 to 0

# Vectorize the Pnet calculation function
calculate_pnet_vec = np.vectorize(calculate_pnet)

# Initialize an empty list to store Pnet arrays
pnet_arrays = []

# Calculate Pnet for each date and precipitation value with a progress bar
for index, row in tqdm(precip_df.iterrows(), total=precip_df.shape[0]):
    P = row[station]
    Pnet_array = calculate_pnet_vec(LAI_array, P)
    pnet_arrays.append(Pnet_array)

# Save Pnet arrays to a list in the DataFrame
precip_df['Pnet_arrays'] = pnet_arrays

# TIA_array = TIA_20m.read(1)
# TIA_array[TIA_array < 0] = 0  # Set values less than 0 to 0

# Calculate RO for each Pnet array
ro_arrays = []
for pnet_array in pnet_arrays:
    ro_array = pnet_array * (0.01 * pnet_array + 0.47 * TIA_array + 0.02)
    ro_arrays.append(ro_array)

# Save RO arrays to a list in the DataFrame
precip_df['RO_arrays'] = ro_arrays

# Print the resulting DataFrame
print(precip_df)

# #CHANGE THE NAME OF THE BASIN
# # Output directory for RO raster files
# output_dir = "WMB" #CHANGE THE NAME OF THE BASIN
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)


# # Calculate RO for each Pnet array and save as raster files
# # Calculate RO for each Pnet array and save as raster files
# ro_arrays = []
# for index, (pnet_array, row) in enumerate(zip(pnet_arrays, precip_df.itertuples())):
#     ro_array = pnet_array * (0.01 * pnet_array + 0.47 * TIA_array + 0.02)
#     ro_arrays.append(ro_array)

#     # Save the RO array as a raster file
#     output_file = os.path.join(output_dir, f"WMB_{row.date.strftime('%Y_%m_%d_%H_%M')}_RO.tif")
#     print(f"Generating raster {index + 1}/{len(pnet_arrays)}: {output_file}")
#     with rasterio.open(
#         output_file,
#         'w',
#         driver='GTiff',
#         height=ro_array.shape[0],
#         width=ro_array.shape[1],
#         count=1,
#         dtype=ro_array.dtype,
#         crs=TIA_20m.crs,
#         transform=TIA_20m.transform,
#     ) as dst:
#         dst.write(ro_array, 1)


# # Save RO arrays to a list in the DataFrame
# precip_df['RO_arrays'] = ro_arrays

# # Print the resulting DataFrame
# print(precip_df)


###########################---------CREATING A TIME SERIES---------###################################
# Create an empty DataFrame to store the runoff time series
runoff_time_series = pd.DataFrame(columns=['date', 'runoff'])


for index, row in tqdm(precip_df.iterrows(), total=precip_df.shape[0]):
    date = row['date']
    
    # Get the runoff data from RO_arrays
    runoff_data = row['RO_arrays']

    # Calculate the sum of all pixels (ignoring non-positive values)
    runoff_sum = np.nansum(runoff_data) / 12

    # Append the date and runoff sum to the time series DataFrame
    runoff_time_series = runoff_time_series.append({'date': date, 'runoff': runoff_sum}, ignore_index=True)

# Set the date column as the index and sort the DataFrame by index
runoff_time_series['date'] = pd.to_datetime(runoff_time_series['date'])
runoff_time_series = runoff_time_series.set_index('date').sort_index()

# Create a complete DateTimeIndex for the desired date range with a 5-minute frequency
start_datetime = datetime.strptime(start_date, '%m/%d/%Y')
end_datetime = datetime.strptime(end_date, '%m/%d/%Y')
complete_index = pd.date_range(start=start_datetime, end=end_datetime, freq='5min')

# Reindex the DataFrame with the complete DateTimeIndex and fill missing values with 0
runoff_time_series = runoff_time_series.reindex(complete_index, fill_value=0)

# Filter the DataFrame based on the inputted date range
start_date_str = start_datetime.strftime('%Y-%m-%d')
end_date_str = end_datetime.strftime('%Y-%m-%d')
filtered_runoff_time_series = runoff_time_series.loc[start_date_str:end_date_str]

# Export the filtered_runoff_time_series DataFrame to a CSV file
filtered_runoff_time_series.to_csv("C:\\Users\\carsk\\OneDrive - KU Leuven\\Thesis\RESULTS\MOL5\RO_norouting\\runoff_time_series.csv")



import matplotlib.pyplot as plt

# Create a precipitation time series DataFrame from precip_df
precip_time_series = precip_df.set_index('date')[station]

# Create a new figure and axis for the plot
fig, ax1 = plt.subplots()

# Plot the precipitation time series as bars
ax1.bar(precip_time_series.index, precip_time_series, color='r', label='Precipitation')
ax1.set_xlabel('Date')
ax1.set_ylabel('Precipitation', color='r')
ax1.tick_params('y', colors='r')

# Invert the primary y-axis for the precipitation
ax1.invert_yaxis()

# Create a secondary y-axis for the runoff time series
ax2 = ax1.twinx()

# Plot the runoff time series on the secondary y-axis
ax2.plot(filtered_runoff_time_series.index, filtered_runoff_time_series['runoff'], 'b', label='Runoff')
ax2.set_ylabel('Runoff', color='b')
ax2.tick_params('y', colors='b')

# Set the plot title
plt.title('Precipitation and Runoff Time Series')

# Show the plot
plt.show()


# Optionally, you can save the plot as an image file
fig.savefig("C:\\Users\\carsk\\OneDrive - KU Leuven\\Thesis\RESULTS\MOL5\precipitation_runoff_time_series.png")

import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Create a subplots with shared x-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add Precipitation bar chart to the plot
fig.add_trace(go.Scatter(x=precip_time_series.index, y=precip_time_series, name="Precipitation", marker_color='red'), secondary_y=False)

# Add Runoff line chart to the plot
fig.add_trace(go.Scatter(x=filtered_runoff_time_series.index, y=filtered_runoff_time_series['runoff'], name="Runoff", line=dict(color='blue')), secondary_y=True)

# Set plot title
fig.update_layout(title="Precipitation and Runoff Time Series")

# Set y-axis titles and invert the primary y-axis
fig.update_yaxes(title_text="Precipitation", tickcolor='red', autorange="reversed", secondary_y=False)
fig.update_yaxes(title_text="Runoff", tickcolor='blue', secondary_y=True)

# Show the plot
fig.show()

# Optionally, you can save the plot as an HTML file
fig.write_html("C:\\Users\\carsk\\OneDrive - KU Leuven\\Thesis\RESULTS\MOL5\precipitation_runoff_time_series_plotly.html")
