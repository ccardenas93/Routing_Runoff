# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:59:01 2023

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

basin_station_map = {
    'WOL4': 'P11',
    # 'WOL5': 'P16',
    # 'WOL7': 'P13',
    # 'MAL1': 'P14',
    # 'ZEN6': 'P14',
    # 'ZEN3': 'P07',
    # 'BNV4': 'P07',
    # 'BNV8': 'P10',
    # 'MOL5': 'P02',# Replace 'Pxxx' with the actual station name
    # Add more basin-station mappings as required
}


for basin_name, station in basin_station_map.items():
    # Update input file paths
    basin_shp_path = f"C:\\Users\\carsk\\OneDrive - KU Leuven\\Thesis\\RESULTS\\{basin_name}\\basin.shp"
    # Load the shapefile, rasters, and CSV file
    basin_gdf = gpd.read_file(basin_shp_path)

    TIA_20m = rasterio.open(r"C:\Users\carsk\OneDrive - KU Leuven\Thesis\RESULTS\INPUTS\TIA_20mf.tif")
    LAI_max_20m = rasterio.open(r"C:\Users\carsk\OneDrive - KU Leuven\Thesis\RESULTS\INPUTS\LAI_20mf.tif")
    LAI_min_20m = rasterio.open(r"C:\Users\carsk\OneDrive - KU Leuven\Thesis\RESULTS\INPUTS\LAI_min_20mf.tif")
    precip_df = pd.read_csv(r"C:\Users\carsk\OneDrive - KU Leuven\Thesis\RESULTS\INPUTS\9modeledStns.csv", parse_dates=['date'])

    # Get the geometry of the shapefile in GeoJSON format and mask the TIA and LAI rasters
    basin_geom = basin_gdf.geometry.to_crs(TIA_20m.crs)
    TIA_array, TIA_transform = mask(TIA_20m, basin_geom, crop=True, nodata=TIA_20m.nodata)
    LAI_max_array, LAI_max_transform = mask(LAI_max_20m, basin_geom, crop=True, nodata=LAI_max_20m.nodata)
    LAI_min_array, _ = mask(LAI_min_20m, basin_geom, crop=True, nodata=LAI_min_20m.nodata)

    # Squeeze the arrays to remove the unnecesary dimension and set nodata values to NaN
    TIA_array = np.squeeze(TIA_array).astype('float32')
    LAI_max_array = np.squeeze(LAI_max_array).astype('float32')
    LAI_min_array = np.squeeze(LAI_min_array).astype('float32')
    TIA_array[TIA_array == TIA_20m.nodata] = np.nan
    LAI_max_array[LAI_max_array == LAI_max_20m.nodata] = np.nan
    LAI_min_array[LAI_min_array == LAI_min_20m.nodata] = np.nan

    print(f"Processing basin: {basin_name}")
    
    start_date = '01/01/2016'  # Replace with the desired start date 'MM/DD/YYYY'
    end_date = '01/02/2016'    # Replace with teh desired end date 'MM/DD/YYYY'

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
        
        return min(Pnet, P)

    # Vectorize the Pnet calculation function
    calculate_pnet_vec = np.vectorize(calculate_pnet)

    # Calculate Pnet for each date and precipitation value with a progress bar
    pnet_arrays = []
    for index, row in tqdm(precip_df.iterrows(), total=precip_df.shape[0]):
        P = row[station]
        month = row['date'].month

        # Select the appropriate LAI raster based on the month
        if month in [3, 4, 5, 6, 7, 8]:  # Spring and summer months
            LAI_array = LAI_max_array
        else:  # Autumn and winter months
            LAI_array = LAI_min_array

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

    # Read the TO raster and get the geometry of the shapefile in GeoJSON format
    TO = rasterio.open(fr"C:\Users\carsk\OneDrive - KU Leuven\Thesis\RESULTS\INPUTS\TO\{basin_name}TO.tif")
    basin_geom = basin_gdf.geometry.to_crs(TO.crs)
    TO_array, TO_transform = mask(TO, basin_geom, crop=True, nodata=TO.nodata)

    # Squeeze the arrays to remove the unnecessary dimension and set nodata values to NaN
    TO_array = np.squeeze(TO_array).astype('float32')
    TO_array[TO_array == TO.nodata] = np.nan

    # Convert TO_array from hours to minutes
    TO_array = TO_array * 60

    # Reclassify the TO_array to 5-minute intervals
    TO_array = np.ceil(TO_array / 5)

    # Initialize an empty DataFrame for the runoff time series
    runoff_time_series = pd.DataFrame(columns=['date', 'RO'])

    # Iterate through each RO_array in precip_df
    for _, row in precip_df.iterrows():
        RO_array = row['RO_arrays']
        date = row['date']

        # Get the unique values in the TO array
        unique_TO_values = np.unique(TO_array)

        # Iterate through the unique values in the TO array
        for TO_value in unique_TO_values:
            # Find the positions in the TO array where the value equals TO_value
            positions = np.where(TO_array == TO_value)

            # Sum the values in the RO_array at the matching positions
            RO_sum = (((np.sum(RO_array[positions])/12)*400)/(1000*300))

            # Increment the timestamp by 5 minutes for each unique TO value
            date += timedelta(minutes=5)

            # Append the sum and the corresponding timestamp to the runoff_time_series DataFrame
            runoff_time_series = runoff_time_series.append({'date': date, 'RO': RO_sum}, ignore_index=True)

 # Convert the date column to datetime, group by date, and sum the RO values for each group
    runoff_time_series['date'] = pd.to_datetime(runoff_time_series['date'])
    runoff_time_series = runoff_time_series.groupby('date').sum().reset_index()

    # Set the date column as the index and sort the DataFrame by index
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

    # Export filtered_runoff_time_series to a CSV file
    output_file_path = f"C:/Users/carsk/OneDrive - KU Leuven/Thesis/ROUTING/{basin_name}/RO_routed/{basin_name}_routed_RO_transform.csv"
    filtered_runoff_time_series.to_csv(output_file_path, index=True)

    print(f"Filtered runoff time series for {basin_name} has been saved as {output_file_path}")
    
    import matplotlib.pyplot as plt

    # Create a precipitation time series DataFrame from precip_df
    precip_time_series = precip_df.set_index('date')[station]
    precip_time_series = precip_time_series.reindex(filtered_runoff_time_series.index, fill_value=0)

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
    ax2.plot(filtered_runoff_time_series.index, filtered_runoff_time_series['RO'], 'b', label='Runoff')
    ax2.set_ylabel('Runoff', color='b')
    ax2.tick_params('y', colors='b')

    # Set the plot title
    plt.title(f'{basin_name} Precipitation and Runoff Time Series')

    # Create a scatter plot of precipitation vs. runoff
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(precip_time_series, filtered_runoff_time_series['RO'], c='g', marker='o', alpha=0.5)
    ax_scatter.set_xlabel('Precipitation')
    ax_scatter.set_ylabel('Runoff')
    ax_scatter.set_title(f'{basin_name} Precipitation vs. Runoff')
    plt.show()

    # Update the output file path for the scatter plot image
    output_scatter_image_path = f"C:/Users/carsk/OneDrive - KU Leuven/Thesis/ROUTING/{basin_name}/RO_routed/{basin_name}_P_vs_RO_scattertransform.png"
    fig_scatter.savefig(output_scatter_image_path)

    # Show the plot
    plt.show()


    # Update the output file path for the image
    output_image_path = f"C:/Users/carsk/OneDrive - KU Leuven/Thesis/ROUTING/{basin_name}/RO_routed/{basin_name}_P_RO_timeseriestransform.png"
    fig.savefig(output_image_path)
    
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    # Create a subplots with shared x-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Precipitation bar chart to the plot
    fig.add_trace(go.Scatter(x=precip_time_series.index, y=precip_time_series, name="Precipitation", marker_color='red'), secondary_y=False)

    # Add Runoff line chart to the plot
    fig.add_trace(go.Scatter(x=filtered_runoff_time_series.index, y=filtered_runoff_time_series['RO'], name="Runoff", line=dict(color='blue')), secondary_y=True)

    # Set plot title
    fig.update_layout(title="f{basin_name} Precipitation and Runoff Time Series")

    # Set y-axis titles and invert the primary y-axis
    fig.update_yaxes(title_text="Precipitation", tickcolor='red', autorange="reversed", secondary_y=False)
    fig.update_yaxes(title_text="Runoff", tickcolor='blue', secondary_y=True)

    # Show the plot
    fig.show()

    # Update the output file path for the Plotly HTML file
    output_plotly_html_path = f"C:/Users/carsk/OneDrive - KU Leuven/Thesis/ROUTING/{basin_name}/RO_routed/{basin_name}_P_RO_timeseriestransform.html"
    fig.write_html(output_plotly_html_path)

    # Create a scatter plot of precipitation vs. runoff using Plotly
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(x=precip_time_series, y=filtered_runoff_time_series['RO'], mode='markers', marker=dict(color='green', opacity=0.5), name=' {basin_name}Precipitation vs. Runoff'))
    scatter_fig.update_layout(title=f'{basin_name} Precipitation vs. Runoff', xaxis_title='Precipitation', yaxis_title='Runoff')
    scatter_fig.show()

    # Update the output file path for the Plotly scatter plot HTML file
    output_scatter_plotly_html_path = f"C:/Users/carsk/OneDrive - KU Leuven/Thesis/ROUTING/{basin_name}/RO_routed/{basin_name}_P_vs_RO_scattertransform.html"
    scatter_fig.write_html(output_scatter_plotly_html_path)
    

