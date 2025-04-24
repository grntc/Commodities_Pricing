import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import getpass
from datetime import datetime
from dateutil.relativedelta import relativedelta
import traceback
import math

COMMODITY_CONFIG = {
    "name": "Nat Gas (Henry Hub)",
    "series_id": "RNGWHHD",
    "resample_freq": 'W-FRI',
    "unit": "$/MMBtu",
    "date_format": "%Y-%m-%d",
    "start_date": "1999-01-01"
}

API_KEY = None

def get_api_key():
    global API_KEY
    if API_KEY is None:
        try:
            print("--- API Key will be visible during input ---")
            API_KEY = input("Please enter your EIA API Key: ")
            if not API_KEY or len(API_KEY.strip()) == 0:
                 print("API Key not provided or is empty. Exiting.")
                 API_KEY = None
                 return None
            API_KEY = API_KEY.strip()
            print("API Key stored.")
        except Exception as e:
            print(f"Could not get API key input: {e}")
            API_KEY = None
    return API_KEY

def fetch_eia_data_paginated(config):
    global API_KEY
    if not get_api_key():
        print("API Key is required.")
        return pd.DataFrame()

    commodity_name = config['name']
    start_date = config.get('start_date', None)
    base_url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
    page_size = 5000
    all_data_list = []
    current_offset = 0
    total_records = None

    print(f"\n--- Fetching for: {commodity_name} (Since {start_date or 'Beginning'}) ---")

    while True:
        params = {
            "frequency": "daily",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": current_offset,
            "length": page_size,
            "api_key": API_KEY
        }
        if start_date:
            params["start"] = start_date

        params_to_print = params.copy()
        params_to_print['api_key'] = '***HIDDEN***'
        print(f"Requesting records {current_offset + 1} - {current_offset + page_size}...")
        print(f"Params: {params_to_print}")

        try:
            response = requests.get(base_url, params=params, timeout=60)
            print(f"API Response Status Code: {response.status_code}")

            try:
                eia_data = response.json()
                print("API Response parsed.")
            except json.JSONDecodeError:
                eia_data = None
                print("Could not parse API response as JSON.")
                print("Raw Response Text:", response.text[:500] + "..." if len(response.text) > 500 else response.text)
                break

            if response.status_code != 200:
                 print(f"\n--- API Error ({response.status_code}) ---")
                 if eia_data and 'error' in eia_data:
                     print("Error details:", eia_data.get('error'))
                 elif eia_data:
                     print("Response Content (non-standard error):", json.dumps(eia_data, indent=2))
                 else:
                     print("Response Text:", response.text[:500] + "..." if len(response.text) > 500 else response.text)
                 response.raise_for_status()
                 break

            if eia_data and 'response' in eia_data and 'data' in eia_data['response']:
                current_data_list = eia_data['response']['data']
                num_fetched = len(current_data_list)
                print(f"Fetched {num_fetched} records in this request.")

                if total_records is None and 'total' in eia_data['response']:
                    total_records = int(eia_data['response']['total'])
                    print(f"Total records reported by API: {total_records}")

                if current_data_list:
                    all_data_list.extend(current_data_list)
                else:
                    print("Received empty data list, assuming end of records.")
                    break

                if num_fetched < page_size:
                    print("Fetched fewer records than page size, assuming end of records.")
                    break

                current_offset += num_fetched

                if total_records is not None and current_offset >= total_records:
                     print("Fetched records equal or exceed total reported, stopping.")
                     break

            else:
                print(f"Warning: Response OK, but no data array found or data array is empty.")
                if eia_data:
                     print("API Response Content:", json.dumps(eia_data, indent=2))
                break

        except requests.exceptions.Timeout:
            print(f"Error: The request to the EIA API timed out.")
            break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            if 'response' in locals() and response is not None:
                 print("Response Text:", response.text[:500] + "..." if len(response.text) > 500 else response.text)
            break
        except Exception as e:
            print(f"An unexpected error occurred fetching data: {e}")
            print(traceback.format_exc())
            break

    print(f"\nTotal records fetched across all pages: {len(all_data_list)}")
    if not all_data_list:
        return pd.DataFrame()
    else:
        return pd.DataFrame(all_data_list)


def process_raw_data(df, config):
    target_series_id = config['series_id']
    print(f"\n--- Processing raw daily data for {target_series_id} ---")
    print(f"Input data shape: {df.shape}")
    try:
        print(f"Filtering for series ID: {target_series_id}...")
        if 'series' not in df.columns:
            print(f"Error: 'series' column not found in fetched data. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        df_filtered = df[df['series'] == target_series_id].copy()
        print(f"Shape after filtering: {df_filtered.shape}")

        if df_filtered.empty:
            print(f"No data found for series ID {target_series_id} in the fetched data.")
            if 'series' in df.columns:
                 unique_series = df['series'].unique()
                 print(f"Unique series IDs in fetched data: {unique_series[:20]}..." if len(unique_series) > 20 else unique_series)
            return pd.DataFrame()

        print(f"Converting 'period' column using format: {config['date_format']}...")
        df_filtered['period'] = pd.to_datetime(df_filtered['period'], format=config['date_format'], errors='coerce')
        print(f"Null periods after conversion: {df_filtered['period'].isnull().sum()}")

        print("Converting 'value' column to numeric...")
        df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
        print(f"Null values after conversion: {df_filtered['value'].isnull().sum()}")

        print("Dropping rows with null period or value...")
        df_filtered.dropna(subset=['period', 'value'], inplace=True)
        print(f"Shape after dropping NaNs: {df_filtered.shape}")

        if df_filtered.empty:
            print("No valid data after initial cleaning and type conversion.")
            return pd.DataFrame()

        print("Setting 'period' as index...")
        df_filtered.set_index('period', inplace=True)
        print("Sorting index...")
        df_filtered.sort_index(inplace=True)

        print("Raw daily data processing complete.")
        return df_filtered[['value']]

    except Exception as e:
        print(f"An error occurred during raw data processing for {target_series_id}: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()

def resample_to_weekly(df_daily, config):
    print(f"\n--- Resampling daily data to weekly ({config['resample_freq']}) ---")
    if df_daily.empty:
        print("Input daily DataFrame is empty, cannot resample.")
        return pd.DataFrame()

    target_freq = config.get('resample_freq')
    if not target_freq:
        print("No resample frequency specified in config.")
        return df_daily

    try:
        print(f"Resampling data to {target_freq} frequency using mean...")
        df_resampled = df_daily['value'].resample(target_freq).mean()
        df_weekly = df_resampled.to_frame()
        print(f"Shape after resampling: {df_weekly.shape}")

        print("Dropping NaNs after resampling (if any)...")
        df_weekly.dropna(subset=['value'], inplace=True)
        print(f"Final weekly shape after dropping NaNs: {df_weekly.shape}")

        if df_weekly.empty:
            print("No data remaining after resampling.")
            return pd.DataFrame()

        print("Weekly resampling complete.")
        return df_weekly

    except Exception as e:
        print(f"An error occurred during weekly resampling: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()


def generate_summary(df_weekly, commodity_name, unit):
    print(f"\n--- Generating Summary for: {commodity_name} ---")
    if df_weekly.empty or len(df_weekly) < 2:
        print("Insufficient weekly data points for summary.")
        return f"Insufficient weekly data for {commodity_name} to generate a summary."

    try:
        latest_data = df_weekly.iloc[-1]
        previous_data = df_weekly.iloc[-2]

        latest_price = latest_data['value']
        previous_price = previous_data['value']
        change = ((latest_price - previous_price) / previous_price) * 100 if previous_price != 0 else 0

        latest_period_end_date = latest_data.name.strftime('%Y-%m-%d')
        previous_period_end_date = previous_data.name.strftime('%Y-%m-%d')
        direction = "increased" if change > 0 else "decreased" if change < 0 else "remained unchanged"

        summary = f"--- {commodity_name} Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---\n\n"
        summary += f"Data Source: EIA API (Processed to Weekly Avg.)\n"
        summary += f"Latest Available Week Ending: {latest_period_end_date}\n\n"
        summary += f" * Latest Weekly Avg Price: ${latest_price:.2f} {unit}\n"
        summary += f" * Previous Week Avg Price (Ending {previous_period_end_date}): ${previous_price:.2f} {unit}\n"
        summary += f" * Weekly Change: {direction} by {abs(change):.2f}%\n"

        print("Summary generated successfully.")
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        print(traceback.format_exc())
        return f"Error generating summary for {commodity_name}."

def plot_recent_trend(df, time_delta, time_unit, ax, data_label, title_suffix):
    if not pd.api.types.is_datetime64_any_dtype(df.index):
         print(f"Plotting Error ({title_suffix}): Index is not datetime.")
         ax.text(0.5, 0.5, f'Invalid index type for {title_suffix} plot', ha='center', va='center', transform=ax.transAxes)
         ax.set_title(f'{time_delta}{time_unit} Trend ({title_suffix})')
         return

    end_date = df.index.max()
    if pd.isna(end_date):
         print(f"Plotting Error ({title_suffix}): No valid end date.")
         ax.text(0.5, 0.5, f'No valid end date for {title_suffix} plot', ha='center', va='center', transform=ax.transAxes)
         ax.set_title(f'{time_delta}{time_unit} Trend ({title_suffix})')
         return

    print(f"DEBUG ({title_suffix}): Latest date in input data = {end_date}")

    if time_unit == 'Y':
        start_date = end_date - relativedelta(years=time_delta)
        title = f'{time_delta}-Year Trend ({title_suffix})'
    elif time_unit == 'M':
        start_date = end_date - relativedelta(months=time_delta)
        title = f'{time_delta}-Month Trend ({title_suffix})'
    elif time_unit == 'D':
         start_date = end_date - relativedelta(days=time_delta)
         title = f'{time_delta}-Day Trend ({title_suffix})'
    else:
        print(f"Plotting Error: Unsupported time unit '{time_unit}'.")
        ax.text(0.5, 0.5, f'Unsupported time unit: {time_unit}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{time_delta}{time_unit} Trend ({title_suffix})')
        return

    print(f"DEBUG ({title}): Calculated Start Date = {start_date}, End Date = {end_date}")
    period_df = df[(df.index >= start_date) & (df.index <= end_date)]
    print(f"DEBUG ({title}): Data points in filtered period_df = {len(period_df)}")
    if not period_df.empty:
        print(f"DEBUG ({title}): Filtered data range = {period_df.index.min()} to {period_df.index.max()}")


    if period_df.empty:
        print(f"No data available to plot for {title}")
        ax.text(0.5, 0.5, f'No data available for {title}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    marker = '.' if title_suffix == 'Weekly Avg' else ''
    color = 'darkcyan' if title_suffix == 'Weekly Avg' else 'purple'
    ax.plot(period_df.index, period_df['value'], label=data_label, color=color, marker=marker, markersize=3, linestyle='-')
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.set_xlim(start_date, end_date + relativedelta(days=1)) # Add a tiny buffer to end date for visibility
    print(f"DEBUG ({title}): Set x-axis limits from {start_date} to {end_date}")


    if time_unit == 'Y' and time_delta >= 5:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif time_unit == 'Y' and time_delta < 5 :
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif time_unit == 'M' and time_delta >= 6:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif time_unit == 'M' and time_delta < 6:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    elif time_unit == 'D':
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, time_delta // 7)))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)

def plot_all_trends(df_weekly, df_daily, commodity_name, unit):
    if df_weekly.empty and df_daily.empty:
        print(f"No weekly or daily data to plot for {commodity_name}.")
        return

    print(f"\n--- Generating Plots for: {commodity_name} ---")
    try:
        fig, axes = plt.subplots(5, 1, figsize=(9, 18), sharex=False)
        fig.suptitle(f'{commodity_name} Price Trends', fontsize=16, y=1.0)

        plot_recent_trend(df_weekly, 5, 'Y', axes[0], 'Weekly Avg Price', 'Weekly Avg')
        plot_recent_trend(df_weekly, 1, 'Y', axes[1], 'Weekly Avg Price', 'Weekly Avg')
        plot_recent_trend(df_weekly, 6, 'M', axes[2], 'Weekly Avg Price', 'Weekly Avg')
        plot_recent_trend(df_daily, 30, 'D', axes[3], 'Daily Price', 'Daily')
        plot_recent_trend(df_daily, 7, 'D', axes[4], 'Daily Price', 'Daily')

        for ax in axes:
            ax.set_ylabel(f'Price ({unit})', fontsize=9)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()
        print(f"Plots generated for {commodity_name}.")
    except Exception as e:
        print(f"An error occurred during plotting for {commodity_name}: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    raw_df = fetch_eia_data_paginated(COMMODITY_CONFIG)

    if not raw_df.empty:
        processed_df_daily = process_raw_data(raw_df, COMMODITY_CONFIG)
        processed_df_weekly = resample_to_weekly(processed_df_daily, COMMODITY_CONFIG)

        if not processed_df_weekly.empty:
            summary = generate_summary(processed_df_weekly, COMMODITY_CONFIG['name'], COMMODITY_CONFIG['unit'])
            print("\n" + summary)
        else:
            print("\nWeekly data processing failed. Cannot generate summary.")

        if not processed_df_weekly.empty or not processed_df_daily.empty:
             plot_all_trends(processed_df_weekly, processed_df_daily, COMMODITY_CONFIG['name'], COMMODITY_CONFIG['unit'])
        else:
             print("\nNo processed data available to plot.")

    else:
        print("\nData fetching failed. Cannot proceed.")
