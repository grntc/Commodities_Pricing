import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import traceback

COMMODITY_CONFIG = {
    "name": "Nat Gas (Henry Hub)",
    "series_id": "RNGWHHD",
    "resample_freq": 'W-FRI',
    "unit": "$/MMBtu",
    "date_format": "%Y-%m-%d"
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

def fetch_eia_data(config):
    global API_KEY
    if not get_api_key():
        print("API Key is required.")
        return pd.DataFrame()

    commodity_name = config['name']
    base_url_part = "https://api.eia.gov/v2/natural-gas/pri/fut/data/?frequency=daily&data[0]=value&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"
    full_url = f"{base_url_part}&api_key={API_KEY}"

    print(f"\n--- Fetching for: {commodity_name} ---")
    print(f"Using URL: {base_url_part}&api_key=***HIDDEN***")

    df_raw = pd.DataFrame()
    try:
        print("Making API request...")
        response = requests.get(full_url, timeout=45)
        print(f"API Response Status Code: {response.status_code}")

        try:
            eia_data = response.json()
            print("API Response parsed.")
        except json.JSONDecodeError:
            eia_data = None
            print("Could not parse API response as JSON.")
            print("Raw Response Text:", response.text[:500] + "..." if len(response.text) > 500 else response.text)

        if response.status_code != 200:
             print(f"\n--- API Error ({response.status_code}) ---")
             if eia_data and 'error' in eia_data:
                 print("Error details:", eia_data.get('error'))
             elif eia_data:
                 print("Response Content (non-standard error):", json.dumps(eia_data, indent=2))
             else:
                 print("Response Text:", response.text[:500] + "..." if len(response.text) > 500 else response.text)
             response.raise_for_status()

        if eia_data and 'response' in eia_data and 'data' in eia_data['response'] and eia_data['response']['data']:
            data_list = eia_data['response']['data']
            df_raw = pd.DataFrame(data_list)
            print(f"Successfully loaded {len(df_raw)} records (before processing).")
        else:
            print(f"Warning: Response OK, but no data array found or data array is empty.")
            if eia_data:
                 print("API Response Content:", json.dumps(eia_data, indent=2))

    except requests.exceptions.Timeout:
        print(f"Error: The request to the EIA API timed out.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        if 'response' in locals() and response is not None:
             print("Response Text:", response.text[:500] + "..." if len(response.text) > 500 else response.text)
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred fetching data: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()

    return df_raw

def process_data(df, config):
    target_series_id = config['series_id']
    print(f"\n--- Processing daily data for {target_series_id} ---")
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
                 print(f"Unique series IDs in fetched data: {df['series'].unique()}")
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

        target_freq = config.get('resample_freq')
        if target_freq:
            print(f"Resampling data to {target_freq} frequency using mean...")
            df_resampled = df_filtered['value'].resample(target_freq).mean()
            df_processed = df_resampled.to_frame()
            print(f"Shape after resampling: {df_processed.shape}")
        else:
             print("No resampling specified.")
             df_processed = df_filtered[['value']]

        print("Dropping NaNs after resampling (if any)...")
        df_processed.dropna(subset=['value'], inplace=True)
        print(f"Final shape after processing: {df_processed.shape}")

        if df_processed.empty:
            print("No data remaining after resampling/alignment.")
            return pd.DataFrame()

        print("Data processing complete.")
        return df_processed

    except Exception as e:
        print(f"An error occurred during data processing for {target_series_id}: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()

def generate_summary(df, commodity_name, unit):
    print(f"\n--- Generating Summary for: {commodity_name} ---")
    if df.empty or len(df) < 2:
        print("Insufficient data points for summary.")
        return f"Insufficient weekly data for {commodity_name} to generate a summary."

    try:
        latest_data = df.iloc[-1]
        previous_data = df.iloc[-2]

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

def plot_trend(df, period_str, ax):
    if not pd.api.types.is_datetime64_any_dtype(df.index):
         print(f"Plotting Error ({period_str}): Index is not datetime.")
         ax.text(0.5, 0.5, f'Invalid index type for {period_str}', ha='center', va='center', transform=ax.transAxes)
         ax.set_title(period_str)
         return

    end_date = df.index.max()
    if pd.isna(end_date):
         print(f"Plotting Error ({period_str}): No valid end date.")
         ax.text(0.5, 0.5, f'No valid end date for {period_str}', ha='center', va='center', transform=ax.transAxes)
         ax.set_title(period_str)
         return

    if period_str == '5Y':
        start_date = end_date - relativedelta(years=5)
        title = '5-Year Trend'
    elif period_str == '1Y':
        start_date = end_date - relativedelta(years=1)
        title = '1-Year Trend'
    elif period_str == '6M':
        start_date = end_date - relativedelta(months=6)
        title = '6-Month Trend'
    else:
        print(f"Plotting Error: Unsupported period string '{period_str}'.")
        ax.text(0.5, 0.5, f'Unsupported period: {period_str}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(period_str)
        return

    print(f"Filtering data for {title} (Start: {start_date}, End: {end_date})")
    period_df = df[(df.index >= start_date) & (df.index <= end_date)]
    print(f"Data points in period: {len(period_df)}")

    if period_df.empty:
        print(f"No data available to plot for {title}")
        ax.text(0.5, 0.5, f'No data available for {title}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    ax.plot(period_df.index, period_df['value'], label='Weekly Avg Price', color='darkcyan', marker='.', markersize=3, linestyle='-')
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    if period_str == '5Y':
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif period_str == '1Y':
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif period_str == '6M':
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

def plot_trends_for_commodity(df, commodity_name, unit):
    if df.empty:
        print(f"No data to plot for {commodity_name}.")
        return

    print(f"\n--- Generating Plots for: {commodity_name} ---")
    try:
        fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=False)

        print("Plotting 5Y trend...")
        plot_trend(df, '5Y', axes[0])
        print("Plotting 1Y trend...")
        plot_trend(df, '1Y', axes[1])
        print("Plotting 6M trend...")
        plot_trend(df, '6M', axes[2])

        for ax in axes:
            ax.set_ylabel(f'Price ({unit})', fontsize=10)

        fig.suptitle(f'{commodity_name} Price Trends (Weekly Avg)', fontsize=16, y=1.01)
        plt.tight_layout(rect=[0, 0.03, 1, 0.99])
        plt.show()
        print(f"Plots generated for {commodity_name}.")
    except Exception as e:
        print(f"An error occurred during plotting for {commodity_name}: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    raw_df = fetch_eia_data(COMMODITY_CONFIG)

    if not raw_df.empty:
        processed_df_weekly = process_data(raw_df, COMMODITY_CONFIG)

        if not processed_df_weekly.empty:
            summary = generate_summary(processed_df_weekly, COMMODITY_CONFIG['name'], COMMODITY_CONFIG['unit'])
            print("\n" + summary)
            plot_trends_for_commodity(processed_df_weekly, COMMODITY_CONFIG['name'], COMMODITY_CONFIG['unit'])
        else:
            print("\nData processing failed after filtering or resampling. Cannot generate summary or plots.")
    else:
        print("\nData fetching failed. Cannot proceed.")
