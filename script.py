import json
import pandas as pd
import requests
import traceback
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
try:
    from dash import dcc, html, Input, Output, State, callback, no_update
except ImportError:
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output, State
    from dash import no_update

from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import webbrowser
import threading
import time

COMMODITY_CONFIG = {
    "name": "Nat Gas (Henry Hub)",
    "series_id": "RNGWHHD",
    "resample_freq": 'W-FRI',
    "unit": "$/MMBtu",
    "date_format": "%Y-%m-%d",
    "start_date": "1999-01-01"
}

DATA_STORE = {'daily': pd.DataFrame(), 'weekly': pd.DataFrame(), 'summary': "", 'status': "API Key Needed"}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_eia_data_paginated(config, api_key_from_input):
    if not api_key_from_input:
        logging.error("API Key is required.")
        return pd.DataFrame(), "API Key Missing"

    commodity_name = config['name']
    series_id = config['series_id']
    start_date = config.get('start_date', None)
    base_url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
    page_size = 5000
    all_data_list = []
    current_offset = 0
    total_records = None
    fetch_status = "Fetching..."

    logging.info(f"\n--- Fetching for: {commodity_name} ({series_id}) (Since {start_date or 'Beginning'}) using {base_url} ---")

    while True:
        params = {
            "api_key": api_key_from_input,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": series_id,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": current_offset,
            "length": page_size
        }
        if start_date:
            params["start"] = start_date

        params = {k: v for k, v in params.items() if v is not None}

        params_to_print = params.copy()
        params_to_print['api_key'] = '***HIDDEN***'
        logging.info(f"Requesting records {current_offset + 1} - {current_offset + page_size}...")
        logging.debug(f"URL: {base_url}")
        logging.debug(f"Params: {params_to_print}")

        try:
            response = requests.get(base_url, params=params, timeout=60)
            logging.info(f"API Response Status Code: {response.status_code}")

            try:
                eia_data = response.json()
                logging.debug("API Response parsed.")
            except json.JSONDecodeError:
                eia_data = None
                logging.error("Could not parse API response as JSON.")
                raw_text = response.text[:500] + "..." if len(response.text) > 500 else response.text
                logging.error("Raw Response Text: %s", raw_text)
                fetch_status = f"Error: Could not parse API response. Text: {raw_text}"
                break

            if response.status_code != 200:
                 logging.error(f"\n--- API Error ({response.status_code}) ---")
                 error_detail = f"API Error {response.status_code}"
                 if eia_data and isinstance(eia_data, dict):
                     err_msg = eia_data.get('error', eia_data.get('errors', eia_data.get('message')))
                     if err_msg:
                          error_detail += f": {err_msg}"
                          logging.error("Error details: %s", err_msg)
                     elif 'request' in eia_data and isinstance(eia_data['request'], dict) and 'errors' in eia_data['request']:
                         nested_err = eia_data['request']['errors']
                         error_detail += f": {nested_err}"
                         logging.error("Error details: %s", nested_err)
                     else:
                         logging.error("Response Content (non-standard error): %s", json.dumps(eia_data, indent=2))

                 elif response.text:
                     raw_text = response.text[:500] + "..." if len(response.text) > 500 else response.text
                     error_detail += f" ({raw_text})"
                     logging.error("Response Text: %s", raw_text)
                 else:
                      error_detail += " (No further details)"

                 fetch_status = error_detail
                 break

            if eia_data and 'response' in eia_data and isinstance(eia_data['response'], dict) and 'data' in eia_data['response']:
                current_data_list = eia_data['response']['data']
                if total_records is None and 'total' in eia_data['response']:
                     total_records = int(eia_data['response']['total'])
                     logging.info(f"Total records reported by API: {total_records}")
            else:
                logging.warning(f"Warning: Response OK, but could not locate data array in response['data'].")
                logging.warning("API Response Content: %s", json.dumps(eia_data, indent=2))
                fetch_status = "Warning: Unexpected data format from API (expected response.data)"
                if current_offset == 0:
                     fetch_status = "Error: Unexpected data format on first fetch"
                else:
                     fetch_status = "Success: Reached end or encountered format change"
                break


            num_fetched = len(current_data_list)
            logging.info(f"Fetched {num_fetched} records in this request.")

            if current_data_list:
                processed_page_data = []
                if num_fetched > 0:
                    first_item = current_data_list[0]
                    if isinstance(first_item, dict) and 'period' in first_item and 'value' in first_item:
                         logging.debug("Detected dict-based data structure with 'period' and 'value'.")
                         for item in current_data_list:
                             processed_page_data.append({'period': item.get('period'), 'value': item.get('value')})
                    else:
                         logging.warning(f"Unrecognized data item format in response['data']: {first_item}")
                         fetch_status = "Error: Unrecognized data format in response['data']"
                         break

                all_data_list.extend(processed_page_data)
            else:
                logging.info("Received empty data list, assuming end of records.")
                fetch_status = "Success: Fetched all available data"
                break

            if num_fetched < page_size:
                logging.info("Fetched fewer records than page size, assuming end of records.")
                fetch_status = "Success: Fetched all available data"
                break

            current_offset += num_fetched

            if total_records is not None and current_offset >= total_records:
                 logging.info("Fetched records equal or exceed total reported, stopping.")
                 fetch_status = "Success: Fetched all available data"
                 break

        except requests.exceptions.Timeout:
            logging.error(f"Error: The request to the EIA API timed out.")
            fetch_status = "Error: Request Timeout"
            break
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            if 'response' in locals() and response is not None:
                 logging.error("Response Text: %s", response.text[:500] + "..." if len(response.text) > 500 else response.text)
            fetch_status = f"Error: Connection Error ({e})"
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred fetching data: {e}")
            logging.error(traceback.format_exc())
            fetch_status = f"Error: Unexpected error during fetch ({e})"
            break

    logging.info(f"\nTotal records fetched across all pages: {len(all_data_list)}")
    if not all_data_list:
         if fetch_status.startswith("Success"):
             fetch_status = "Success: No data found for the criteria"
         return pd.DataFrame(), fetch_status
    else:
        df = pd.DataFrame(all_data_list)
        if 'period' not in df.columns or 'value' not in df.columns:
             logging.error(f"Could not structure fetched data into 'period' and 'value' columns. Columns found: {df.columns.tolist()}")
             return pd.DataFrame(), "Error: Post-fetch data structuring failed"
        return df[['period', 'value']], fetch_status


def process_raw_data(df, config):
    logging.info(f"\n--- Processing raw daily data for {config['name']} ---")
    logging.info(f"Input data shape: {df.shape}")
    if df.empty:
        logging.warning("Input DataFrame is empty, cannot process.")
        return pd.DataFrame(), "Processing Error: Input empty"
    try:
        if 'period' not in df.columns or 'value' not in df.columns:
             logging.error(f"Error: 'period' or 'value' column not found. Available columns: {df.columns.tolist()}")
             return pd.DataFrame(), "Processing Error: Missing required columns"

        df_processed = df[['period', 'value']].copy()

        date_format_config = config.get('date_format')
        logging.info(f"Attempting to convert 'period' column to datetime...")
        try:
            if date_format_config:
                df_processed['period'] = pd.to_datetime(df_processed['period'], format=date_format_config, errors='coerce')
                logging.info(f"Used configured format '{date_format_config}'. Nulls after conversion: {df_processed['period'].isnull().sum()}")
            else:
                df_processed['period'] = pd.to_datetime(df_processed['period'], errors='coerce', infer_datetime_format=True)
                logging.info(f"Inferred datetime format. Nulls after conversion: {df_processed['period'].isnull().sum()}")

            if df_processed['period'].isnull().sum() > len(df_processed) * 0.8:
                 logging.warning("High number of null dates after initial conversion attempt. Trying alternative formats.")
                 df_processed['period_alt'] = pd.to_datetime(df['period'], format='%Y%m%d', errors='coerce')
                 df_processed['period'] = df_processed['period'].fillna(df_processed['period_alt'])
                 df_processed.drop(columns=['period_alt'], inplace=True)
                 logging.info(f"Nulls after trying YYYYMMDD format: {df_processed['period'].isnull().sum()}")

        except ValueError as ve:
            logging.error(f"Date conversion failed with specific format or inference: {ve}")
            df_processed['period'] = pd.NaT


        logging.info("Converting 'value' column to numeric...")
        df_processed['value'] = pd.to_numeric(df_processed['value'], errors='coerce')
        logging.info(f"Null values after conversion: {df_processed['value'].isnull().sum()}")

        logging.info("Dropping rows with null period or value...")
        original_count = len(df_processed)
        df_processed.dropna(subset=['period', 'value'], inplace=True)
        logging.info(f"Dropped {original_count - len(df_processed)} rows with NaNs.")
        logging.info(f"Shape after dropping NaNs: {df_processed.shape}")

        if df_processed.empty:
            logging.warning("No valid data after cleaning and type conversion.")
            return pd.DataFrame(), "Processing Warning: No valid data after cleaning"

        logging.info("Setting 'period' as index...")
        df_processed.set_index('period', inplace=True)
        logging.info("Sorting index...")
        df_processed.sort_index(inplace=True)

        logging.info("Raw daily data processing complete.")
        return df_processed[['value']], "Processing Success"

    except Exception as e:
        logging.error(f"An error occurred during raw data processing for {config['name']}: {e}")
        logging.error(traceback.format_exc())
        return pd.DataFrame(), f"Processing Error: {e}"

def resample_to_weekly(df_daily, config):
    logging.info(f"\n--- Resampling daily data to weekly ({config['resample_freq']}) ---")
    if df_daily.empty:
        logging.warning("Input daily DataFrame is empty, cannot resample.")
        return pd.DataFrame(), "Resampling Warning: Input empty"

    target_freq = config.get('resample_freq')
    if not target_freq:
        logging.warning("No resample frequency specified in config.")
        return df_daily, "Resampling Warning: No frequency specified"

    try:
        logging.info(f"Resampling data to {target_freq} frequency using mean...")
        if not isinstance(df_daily.index, pd.DatetimeIndex):
             logging.error("Cannot resample, index is not a DatetimeIndex.")
             return pd.DataFrame(), "Resampling Error: Index not DatetimeIndex"

        df_resampled = df_daily['value'].resample(target_freq).mean()
        df_weekly = df_resampled.to_frame()
        logging.info(f"Shape after resampling: {df_weekly.shape}")

        nan_count = df_weekly['value'].isnull().sum()
        if nan_count > 0:
            logging.info(f"Found {nan_count} NaN values after resampling (weeks with no daily data).")
            logging.info("Dropping NaNs...")
            df_weekly.dropna(subset=['value'], inplace=True)
        else:
            logging.info("No NaNs found after resampling.")

        logging.info(f"Final weekly shape after handling NaNs: {df_weekly.shape}")

        if df_weekly.empty:
            logging.warning("No data remaining after resampling and NaN removal.")
            return pd.DataFrame(), "Resampling Warning: No data after resampling"

        logging.info("Weekly resampling complete.")
        return df_weekly, "Resampling Success"

    except Exception as e:
        logging.error(f"An error occurred during weekly resampling: {e}")
        logging.error(traceback.format_exc())
        return pd.DataFrame(), f"Resampling Error: {e}"


def generate_summary_text(df_weekly, commodity_name, unit):
    logging.info(f"\n--- Generating Summary Text for: {commodity_name} ---")
    if df_weekly.empty or len(df_weekly) < 2:
        logging.warning("Insufficient weekly data points (< 2) for summary.")
        if df_weekly.empty:
            return f"No weekly data available for {commodity_name}."
        else:
             latest_data = df_weekly.iloc[-1]
             latest_price = latest_data['value']
             latest_period_end_date = latest_data.name.strftime('%Y-%m-%d')
             return (f"--- {commodity_name} Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---\n\n"
                     f"Data Source: EIA API (Processed to Weekly Avg.)\n"
                     f"Latest Available Week Ending: {latest_period_end_date}\n\n"
                     f"* Only one weekly data point available.\n"
                     f"* Latest Weekly Avg Price: ${latest_price:.2f} {unit}\n"
                     f"* Cannot calculate weekly change.")

    try:
        latest_data = df_weekly.iloc[-1]
        previous_data = df_weekly.iloc[-2]

        latest_price = latest_data['value']
        previous_price = previous_data['value']

        if pd.isna(latest_price) or pd.isna(previous_price):
             logging.warning("NaN values encountered in latest/previous price for summary.")
             return f"Could not calculate summary for {commodity_name} due to missing price data."

        change = 0.0
        direction = "remained unchanged"
        if previous_price != 0:
            change = ((latest_price - previous_price) / previous_price) * 100
            if change > 0.01:
                 direction = "increased"
            elif change < -0.01:
                 direction = "decreased"
        elif latest_price != 0:
             direction = "increased (from zero)"
             change = float('inf')

        latest_period_end_date = latest_data.name.strftime('%Y-%m-%d')
        previous_period_end_date = previous_data.name.strftime('%Y-%m-%d')

        summary = f"--- {commodity_name} Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---\n\n"
        summary += f"Data Source: EIA API (Processed to Weekly Avg.)\n"
        summary += f"Latest Available Week Ending: {latest_period_end_date}\n\n"
        summary += f"* Latest Weekly Avg Price: ${latest_price:.2f} {unit}\n"
        summary += f"* Previous Week Avg Price (Ending {previous_period_end_date}): ${previous_price:.2f} {unit}\n"
        if change == float('inf'):
             summary += f"* Weekly Change: {direction}\n"
        else:
             summary += f"* Weekly Change: {direction} by {abs(change):.2f}%\n"

        logging.info("Summary text generated successfully.")
        return summary
    except IndexError:
         logging.error("IndexError during summary generation - likely less than 2 data points despite initial check.")
         return f"Error generating summary for {commodity_name} (IndexError)."
    except Exception as e:
        logging.error(f"Error generating summary text: {e}")
        logging.error(traceback.format_exc())
        return f"Error generating summary for {commodity_name}."

def create_trend_figure(df, time_delta, time_unit, data_label, title_suffix, unit):
    fig = go.Figure()
    title = f'{time_delta}-{time_unit} Trend ({title_suffix})'

    if df is None or df.empty:
        logging.warning(f"Plotting Error ({title}): Input DataFrame is empty or None.")
        fig.add_annotation(text="No data available for this plot",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16)
        fig.update_layout(title=dict(text=title, x=0.5))
        return fig

    if not isinstance(df.index, pd.DatetimeIndex):
         logging.error(f"Plotting Error ({title}): Index is not a DatetimeIndex.")
         fig.add_annotation(text="Invalid data index for plotting (not datetime)",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16)
         fig.update_layout(title=dict(text=title, x=0.5))
         return fig

    if df.index.isna().all():
         logging.error(f"Plotting Error ({title}): Index contains only NaT values.")
         fig.add_annotation(text="No valid dates in data index",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False, font_size=16)
         fig.update_layout(title=dict(text=title, x=0.5))
         return fig

    end_date = df.index.max()

    try:
        if time_unit == 'Y':
            start_date = end_date - relativedelta(years=time_delta)
        elif time_unit == 'M':
            start_date = end_date - relativedelta(months=time_delta)
        elif time_unit == 'D':
             start_date = end_date - relativedelta(days=time_delta)
        else:
            logging.error(f"Plotting Error ({title}): Unsupported time unit '{time_unit}'.")
            fig.add_annotation(text=f"Unsupported time unit: {time_unit}",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font_size=16)
            fig.update_layout(title=dict(text=title, x=0.5))
            return fig
    except TypeError as te:
         logging.error(f"Plotting Error ({title}): Could not calculate start date. End date might be NaT or invalid. End date: {end_date}. Error: {te}")
         fig.add_annotation(text="Error calculating date range",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16)
         fig.update_layout(title=dict(text=title, x=0.5))
         return fig

    if pd.isna(start_date) or pd.isna(end_date):
         logging.error(f"Plotting Error ({title}): Invalid start or end date calculated ({start_date} to {end_date}).")
         fig.add_annotation(text="Invalid date range for plot",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False, font_size=16)
         fig.update_layout(title=dict(text=title, x=0.5))
         return fig

    period_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    period_df.dropna(subset=['value'], inplace=True)

    logging.debug(f"Plotting '{title}': Filtered data points = {len(period_df)}")

    if period_df.empty:
        try:
             date_range_str = f"in the calculated range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        except AttributeError:
             date_range_str = "(invalid date range)"
        logging.warning(f"No valid data available to plot for {title} {date_range_str}")
        fig.add_annotation(text=f"No data available for the {time_delta}-{time_unit} period",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16)
    else:
         period_df['value'] = pd.to_numeric(period_df['value'], errors='coerce')
         period_df.dropna(subset=['value'], inplace=True)

         if not period_df.empty:
             fig.add_trace(go.Scattergl(x=period_df.index, y=period_df['value'],
                                    mode='lines+markers' if title_suffix=='Weekly Avg' else 'lines',
                                    name=data_label,
                                    marker=dict(size=4 if title_suffix=='Weekly Avg' else 6),
                                    line=dict(color='darkcyan' if title_suffix == 'Weekly Avg' else 'purple')))
         else:
             logging.warning(f"No numeric data left to plot for {title} after final checks.")
             fig.add_annotation(text="No numeric data to plot for this period",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False, font_size=16)


    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title=f"Price ({unit})",
        xaxis_range=[start_date, end_date],
        margin=dict(l=50, r=20, t=50, b=40),
        hovermode="x unified"
    )
    if time_unit == 'Y' or (time_unit == 'M' and time_delta >= 6):
        fig.update_layout(xaxis_rangeslider_visible=True)

    return fig

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = f"{COMMODITY_CONFIG['name']} Pricing"

app.layout = html.Div([
    dcc.Store(id='api-key-store'),
    dcc.Store(id='data-status-store', data=DATA_STORE['status']),
    html.H1(id='main-title', children=f"{COMMODITY_CONFIG['name']} Price Trends", style={'textAlign': 'center'}),

    html.Div([
        html.Div(id='api-key-input-container', children=[
            dcc.Input(id='api-key-input', type='password', placeholder='Enter EIA API Key', style={'marginRight': '10px'}),
            html.Button('Fetch Data', id='fetch-button', n_clicks=0)
        ], style={'textAlign': 'center', 'padding': '20px'}),
        html.Div(id='status-message', style={'textAlign': 'center', 'padding': '10px', 'fontWeight': 'bold'})
    ]),

    html.Div(id='content-container', children=[
        dcc.Loading(
            id="loading-summary",
            type="default",
            children=html.Pre(id='summary-output', style={'border': '1px solid #ccc', 'padding': '10px', 'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word', 'fontFamily': 'monospace', 'margin': '10px'})
        ),
        dcc.Loading(
            id="loading-plots",
            type="default",
            children=html.Div(id='plots-container')
        )
    ], style={'display': 'none'})
])

@callback(
    [Output('api-key-store', 'data'),
     Output('data-status-store', 'data'),
     Output('status-message', 'children')],
    [Input('fetch-button', 'n_clicks')],
    [State('api-key-input', 'value')],
    prevent_initial_call=True
)
def store_api_key_and_fetch(n_clicks, api_key_value):
    if n_clicks > 0 and api_key_value:
        logging.info("Fetch button clicked, API key provided.")
        status_msg = "Fetching data..."
        return {'key': api_key_value}, "Fetching", status_msg
    elif n_clicks > 0 and not api_key_value:
        logging.warning("Fetch button clicked, but API key is missing.")
        status_msg = "Please enter an API Key."
        return no_update, "API Key Needed", status_msg
    return no_update, no_update, no_update


@callback(
    [Output('content-container', 'style'),
     Output('summary-output', 'children'),
     Output('plots-container', 'children'),
     Output('status-message', 'children', allow_duplicate=True)],
    [Input('data-status-store', 'data')],
    [State('api-key-store', 'data')],
    prevent_initial_call=True
)
def update_page_content(status, api_key_data):
    global DATA_STORE, COMMODITY_CONFIG

    if status != "Fetching" or not api_key_data:
        current_status = DATA_STORE.get('status', 'API Key Needed')
        status_msg = "Please enter an API Key." if current_status == "API Key Needed" else ""
        return {'display': 'none'}, "", html.Div(), status_msg

    api_key = api_key_data.get('key')
    if not api_key:
         return {'display': 'none'}, "", html.Div(), "API Key error."

    logging.info("Status is 'Fetching', proceeding with data pipeline.")
    run_data_pipeline(COMMODITY_CONFIG, api_key)

    df_weekly = DATA_STORE['weekly']
    df_daily = DATA_STORE['daily']
    summary_text = DATA_STORE['summary']
    final_status = DATA_STORE['status']
    commodity_name = COMMODITY_CONFIG['name']
    unit = COMMODITY_CONFIG['unit']

    is_error = "Error" in final_status or (df_daily.empty and df_weekly.empty and not final_status.startswith("Success"))

    if is_error:
         logging.error(f"Data pipeline finished with status: {final_status}")
         plots_layout = html.Div(f"Could not generate plots. Status: {final_status}", style={'textAlign': 'center', 'padding': '20px', 'color': 'red'})
         summary_text_display = summary_text if summary_text and "Error" not in summary_text else f"Status: {final_status}"
         content_style = {'display': 'block'}
         status_message = final_status
    else:
        logging.info("Generating Plotly figures for Dash...")
        fig_5y_w = create_trend_figure(df_weekly, 5, 'Y', 'Weekly Avg Price', 'Weekly Avg', unit)
        fig_1y_w = create_trend_figure(df_weekly, 1, 'Y', 'Weekly Avg Price', 'Weekly Avg', unit)
        fig_6m_w = create_trend_figure(df_weekly, 6, 'M', 'Weekly Avg Price', 'Weekly Avg', unit)
        fig_30d_d = create_trend_figure(df_daily, 30, 'D', 'Daily Price', 'Daily', unit)
        fig_7d_d = create_trend_figure(df_daily, 7, 'D', 'Daily Price', 'Daily', unit)
        logging.info("Figures generated.")

        plots_layout = html.Div([
            html.H3("5-Year Trend (Weekly Avg)", style={'marginTop': '20px'}),
            dcc.Graph(id='plot-5y-weekly', figure=fig_5y_w),
            html.H3("1-Year Trend (Weekly Avg)"),
            dcc.Graph(id='plot-1y-weekly', figure=fig_1y_w),
            html.H3("6-Month Trend (Weekly Avg)"),
            dcc.Graph(id='plot-6m-weekly', figure=fig_6m_w),
            html.H3("30-Day Trend (Daily)"),
            dcc.Graph(id='plot-30d-daily', figure=fig_30d_d),
            html.H3("7-Day Trend (Daily)"),
            dcc.Graph(id='plot-7d-daily', figure=fig_7d_d)
        ])
        content_style = {'display': 'block'}
        status_message = "Data loaded successfully."
        if "Warning" in final_status:
            status_message += f" (Note: {final_status})"
        elif final_status == "Success: No data found for the criteria":
            status_message = final_status


    return content_style, summary_text, plots_layout, status_message


def run_data_pipeline(config, api_key_from_input):
    global DATA_STORE
    logging.info("Starting data pipeline...")
    raw_df, fetch_status = fetch_eia_data_paginated(config, api_key_from_input)
    DATA_STORE['status'] = fetch_status

    if raw_df is not None and not fetch_status.startswith("Error"):
        processed_df_daily, process_status_d = process_raw_data(raw_df, config)
        processed_df_weekly, process_status_w = resample_to_weekly(processed_df_daily, config)

        DATA_STORE['daily'] = processed_df_daily if processed_df_daily is not None else pd.DataFrame()
        DATA_STORE['weekly'] = processed_df_weekly if processed_df_weekly is not None else pd.DataFrame()
        DATA_STORE['status'] += f" | Daily: {process_status_d} | Weekly: {process_status_w}"

        if DATA_STORE['weekly'] is not None and not DATA_STORE['weekly'].empty:
            summary = generate_summary_text(DATA_STORE['weekly'], config['name'], config['unit'])
            DATA_STORE['summary'] = summary
            logging.info("\n" + summary)
        else:
            daily_exists = DATA_STORE['daily'] is not None and not DATA_STORE['daily'].empty
            if daily_exists:
                logging.warning(f"Weekly data processing failed or resulted in empty DataFrame for {config['name']}. Cannot generate weekly summary.")
                DATA_STORE['summary'] = f"Weekly data unavailable for {config['name']}. Cannot generate weekly summary."
                try:
                    date_range_str = f"from {DATA_STORE['daily'].index.min().strftime('%Y-%m-%d')} to {DATA_STORE['daily'].index.max().strftime('%Y-%m-%d')}"
                except Exception:
                    date_range_str = "(dates unavailable)"
                DATA_STORE['summary'] += f"\nDaily data available {date_range_str}."
            else:
                 DATA_STORE['summary'] = f"No processed data available for {config['name']}."
                 if not DATA_STORE['status'].startswith("Error"):
                     DATA_STORE['status'] += " | Summary: Failed (No processed data)"


    else:
        logging.error(f"Data fetching failed. Status: {fetch_status}")
        DATA_STORE['summary'] = f"Data fetching failed for {config['name']}. Cannot generate plots or summary. Status: {fetch_status}"
        DATA_STORE['daily'] = pd.DataFrame()
        DATA_STORE['weekly'] = pd.DataFrame()
    logging.info("Data pipeline finished.")

def open_browser():
    try:
        webbrowser.open_new_tab("http://127.0.0.1:8050/")
    except Exception as e:
        logging.error(f"Could not open browser automatically: {e}")

if __name__ == "__main__":
    logging.info("Starting Dash server...")
    print("\n--- Dash App Initializing ---")
    print(f"Attempting to open dashboard at: http://127.0.0.1:8050/ in your browser.")
    print("If it doesn't open automatically, please navigate there manually.")
    print("Enter your API key in the browser window.")
    print("Press CTRL+C in this terminal to stop the server.")
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.start()
    app.run(debug=False, host='127.0.0.1', port=8050)