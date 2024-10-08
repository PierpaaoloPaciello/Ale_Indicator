import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import io
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Set page title and layout
st.set_page_config(page_title="Global Economic Research", layout="wide")    

# Set background color and add styling
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f9f9f9;
        padding-top: 20px;
    }
    .main {
        background-color: #f9f9f9;
        border-radius: 10px;
    }
    h1 {
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 32px;
        color: #333;
    }
    h2 {
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        color: #333;
    }
    p {
        text-align: justify;
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        color: #666;
    }
    .stPlotlyChart {
        display: flex;
        justify-content: center;
    }
    .stDataFrame {
        margin-left: auto;
        margin-right: auto;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description of the research
st.title("Global Growth Analysis Using OECD CLI Data")

st.markdown("""
### This research examines global growth cycles using the Composite Leading Indicators (CLI) data from the OECD and the S&P500 index.
The study investigates expansion and contraction signals, providing visual insights into the cycles using the Diffusion Index and market data.
""")

# Sidebar: Date range selection
st.sidebar.header("Date Range Selector")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("1990-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# Function to download CLI data from OECD
def get_oecd_data(countries, start_period):
    database = '@DF_CLI'
    frequency = 'M'
    indicator = 'LI..'
    unit_of_measure = 'AA...'
    
    # Join all country codes
    country_code = "+".join(countries)
    
    # Create the query URL
    query_text = f"{database}/{country_code}.{frequency}.{indicator}.{unit_of_measure}?startPeriod={start_period}&dimensionAtObservation=AllDimensions"
    url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES{query_text}"
    
    headers = { 
        'User-Agent': 'Mozilla/5.0', 
        'Accept': 'application/vnd.sdmx.data+csv; charset=utf-8' 
    }
    
    # Fetch the data
    download = requests.get(url=url, headers=headers)
    df = pd.read_csv(io.StringIO(download.text))
    
    return df

# List of countries for DI calculation
countries = ['AUS', 'AUT', 'BEL', 'CAN', 'CHL', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
             'DEU', 'GRC', 'HUN', 'ISL', 'IRL', 'ISR', 'ITA', 'JPN', 'KOR', 'LVA',
             'LTU', 'LUX', 'MEX', 'NLD', 'NZL', 'NOR', 'POL', 'PRT', 'SVK', 'SVN',
             'ESP', 'SWE', 'CHE', 'TUR', 'GBR', 'USA']

# Download the OECD CLI data
cli_data = get_oecd_data(countries, start_date.strftime('%Y-%m'))

# Process Data
pivot_data = cli_data.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
pivot_data.fillna(method='ffill', inplace=True)
pivot_data.index = pd.to_datetime(pivot_data.index, errors='coerce')

# Display processed Pivot Data instead of CLI
st.subheader("Processed OECD CLI Data")
st.markdown("The table below shows the processed OECD CLI data used for **Diffusion Index** calculations and trend identification.")
st.dataframe(pivot_data.tail(12), height=150, width=900)

# Diffusion Index calculation
pivot_data_change = pivot_data.diff()
diffusion_index = (pivot_data_change > 0).sum(axis=1) / len(pivot_data.columns)
pivot_data['DI'] = diffusion_index

# Add Global Mean CLI and smoothing
pivot_data['Global Mean CLI'] = pivot_data.mean(axis=1)
pivot_data['Global Mean CLI (Smoothed)'] = pivot_data['Global Mean CLI'].rolling(window=12, min_periods=1).mean()

# Fetch S&P500 Data from YFinance
spy_data = yf.download('^GSPC', start=start_date, end=end_date)
spy_data = spy_data[['Adj Close']]

# Function to find the nearest available dates in a given reference DataFrame (S&P500)
def find_nearest_dates(signals_df, reference_data):
    nearest_dates = reference_data.index.searchsorted(signals_df.index)
    nearest_dates = nearest_dates.clip(0, len(reference_data) - 1)  # Ensure no out-of-bounds indices
    return reference_data.index[nearest_dates]

# Plot Global Mean CLI and Diffusion Index with Plotly
st.subheader("Global Mean CLI and Diffusion Index")
cli_fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add Global Mean CLI (Smoothed) to primary y-axis (left)
cli_fig.add_trace(
    go.Scatter(x=pivot_data.index, y=pivot_data['Global Mean CLI (Smoothed)'], 
               mode='lines', name='Global Mean CLI (Smoothed)', 
               line=dict(color='black', width=2)),
    secondary_y=False,
)

# Add Diffusion Index to secondary y-axis (right)
cli_fig.add_trace(
    go.Scatter(x=pivot_data.index, y=pivot_data['DI'], 
               mode='lines', name='Diffusion Index', 
               line=dict(color='green', width=2)),
    secondary_y=True,
)

# Update layout with titles and axis labels
cli_fig.update_layout(
    title="Global Mean CLI (Smoothed) and Diffusion Index",
    template="plotly_white",
    height=600,  # Increased height for better visualization
    width=1200,  # Increased width for better visualization
    margin=dict(l=50, r=50, t=50, b=50),  # Better margins for spacing
)

# Update y-axes titles
cli_fig.update_yaxes(title_text="Global Mean CLI", secondary_y=False)
cli_fig.update_yaxes(title_text="Diffusion Index", secondary_y=True)

# Set the x-axis title
cli_fig.update_xaxes(title_text="Date")

# Show the plot
st.plotly_chart(cli_fig)

# Signal Detection (Expansion and Contraction phases)
st.subheader("Expansion and Contraction Signals")

diffusion_threshold = 0.5  # 50%
min_expansion_duration = 5  # Minimum of 5 consecutive months above threshold
blackout_period = pd.DateOffset(months=15)

# Expansion Signal Detection
pivot_data['expansion_flag'] = pivot_data['DI'] > diffusion_threshold
pivot_data['expansion_count'] = pivot_data['expansion_flag'].rolling(window=min_expansion_duration, min_periods=1).sum()
pivot_data['expansion_signal'] = (pivot_data['expansion_flag'] == False) & (pivot_data['expansion_count'].shift(1) >= min_expansion_duration)

# Contraction Signal Detection (Upturns)
pivot_data['contraction_flag'] = pivot_data['DI'] < diffusion_threshold
pivot_data['contraction_count'] = pivot_data['contraction_flag'].rolling(window=min_expansion_duration, min_periods=1).sum()
pivot_data['contraction_signal'] = (pivot_data['contraction_flag'] == False) & (pivot_data['contraction_count'].shift(1) >= min_expansion_duration)

# Filter signals with blackout period
filtered_signals = []
filtered_contraction_signals = []

last_signal_date = None
last_contraction_signal_date = None

for signal_date in pd.to_datetime(pivot_data[pivot_data['expansion_signal']].index):
    if last_signal_date is None or signal_date > last_signal_date + blackout_period:
        filtered_signals.append(signal_date)
        last_signal_date = signal_date

for contraction_date in pd.to_datetime(pivot_data[pivot_data['contraction_signal']].index):
    if last_contraction_signal_date is None or contraction_date > last_contraction_signal_date + blackout_period:
        filtered_contraction_signals.append(contraction_date)
        last_contraction_signal_date = contraction_date

filtered_signals_df = pd.DataFrame(index=filtered_signals, data={'signal': 'End of Expansion'})
filtered_contraction_signals_df = pd.DataFrame(index=filtered_contraction_signals, data={'signal': 'End of Contraction'})

# Use the nearest available S&P500 date for each expansion and contraction signal
nearest_expansion_dates = find_nearest_dates(filtered_signals_df, spy_data)
nearest_contraction_dates = find_nearest_dates(filtered_contraction_signals_df, spy_data)

# Plot Expansion/Contraction Signals and S&P 500 as Subplots with Plotly
st.subheader("Expansion and Contraction Signals with S&P 500")

exp_con_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, subplot_titles=("S&P500 with Expansion/Contraction Signals", "Diffusion Index with Expansion/Contraction Signals"))

# S&P500 price plot with signals
exp_con_fig.add_trace(go.Scatter(x=spy_data.index, y=spy_data['Adj Close'], mode='lines', name='S&P500', line=dict(color='gray', width=2)), row=1, col=1)
exp_con_fig.add_trace(go.Scatter(x=nearest_expansion_dates, y=spy_data.loc[nearest_expansion_dates, 'Adj Close'], mode='markers', name='Expansion Signals', marker=dict(color='red', size=10)), row=1, col=1)
exp_con_fig.add_trace(go.Scatter(x=nearest_contraction_dates, y=spy_data.loc[nearest_contraction_dates, 'Adj Close'], mode='markers', name='Contraction Signals', marker=dict(color='blue', size=10)), row=1, col=1)

# Diffusion Index with signals
exp_con_fig.add_trace(go.Scatter(x=pivot_data.index, y=pivot_data['DI'], mode='lines', name='Diffusion Index', line=dict(color='green', width=2)), row=2, col=1)
exp_con_fig.add_trace(go.Scatter(x=filtered_signals_df.index, y=pivot_data.loc[filtered_signals_df.index, 'DI'], mode='markers', name='Expansion Signals', marker=dict(color='red', size=10)), row=2, col=1)
exp_con_fig.add_trace(go.Scatter(x=filtered_contraction_signals_df.index, y=pivot_data.loc[filtered_contraction_signals_df.index, 'DI'], mode='markers', name='Contraction Signals', marker=dict(color='blue', size=10)), row=2, col=1)

# Update layout
exp_con_fig.update_layout(
    height=700,  # Increased height for better visualization
    width=1200,  # Increased width for better visualization
    title_text="Expansion and Contraction Signals with Diffusion Index",
    template="plotly_white",
)
exp_con_fig.update_xaxes(title_text="Date")
exp_con_fig.update_yaxes(title_text="S&P 500 Price", row=1, col=1)
exp_con_fig.update_yaxes(title_text="Diffusion Index", row=2, col=1)

# Show the plot
st.plotly_chart(exp_con_fig)

# Signal Table
st.subheader("Signal Table")
signals_combined = pd.concat([filtered_signals_df, filtered_contraction_signals_df]).sort_index()

# Signal table explanation
st.markdown("""
The table below shows the exact dates of the detected **Expansion** and **Contraction** signals. 
These signals are derived based on the OECD's CLI data and the corresponding Diffusion Index calculations.
""")

# Display the final table with improved styling
st.dataframe(signals_combined.style.set_table_attributes('style="width: 100%; margin: auto;"').set_properties(**{'text-align': 'center'}))

# Conclusion
st.markdown("## Conclusion\nThis research illustrates the detection of global growth expansion and contraction phases using OECD's CLI data. The integration with S&P500 performance highlights key turning points in the global economic cycle.")
