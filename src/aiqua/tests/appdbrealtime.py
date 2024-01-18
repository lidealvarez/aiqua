import datetime
import joblib
import mysql.connector
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import os
import uuid
import random
from apscheduler.schedulers.background import BackgroundScheduler
import json
import plotly
import threading
import requests
from collections import defaultdict

app = Flask(__name__)
app.secret_key = '6c061b2509dbc420431ad96a31042f4d'

global data_df
data_df = pd.DataFrame()


# Database configuration
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'admin',
    'password': 'admin@eskola',
    'database': 'aiqua'
}

# Global cache
cache = {}

def update_cache(key, value):
    cache[key] = value

def get_from_cache(key):
    return cache.get(key)

# Initialize the BackgroundScheduler
scheduler = BackgroundScheduler()

last_timestamp = None
last_alert_timestamp = None
global daily_anomaly_counts
# Global variable to keep track of anomalies per day
daily_anomaly_counts = defaultdict(int)

import numpy as np

def simulate_real_time_data(data_df):
    global last_timestamp
    # If last_timestamp is None, get the last timestamp from data_df
    if last_timestamp is None:
        if not data_df.empty:
            last_timestamp = data_df['Timestamp'].iloc[-1]
        else:
            # Set initial timestamp if data_df is empty
            last_timestamp = pd.Timestamp.now().floor('5T')  # Round down to nearest 5 minutes

    # Increment timestamp by 5 minutes
    last_timestamp += datetime.timedelta(minutes=5)

    # Convert 'Pressure' to numeric type for calculations
    if not data_df.empty:
        data_df['Pressure'] = pd.to_numeric(data_df['Pressure'], errors='coerce')
        normal_pressure_mean = data_df['Pressure'].mean()
        normal_pressure_std_dev = data_df['Pressure'].std()
    else:
        # Default values if no data is available
        normal_pressure_mean = 5
        normal_pressure_std_dev = 0.2

    # Define anomaly parameters for 'Pressure'
    anomaly_pressure_mean = normal_pressure_mean * 2  # For example, twice the normal mean
    anomaly_pressure_std_dev = normal_pressure_std_dev * 2  # For example, twice the normal std deviation

    # Probability of an anomaly occurring
    anomaly_probability = 0.10  # 10%

    # Determine if this data point is an anomaly
    if random.random() < anomaly_probability:
        pressure = np.random.normal(anomaly_pressure_mean, anomaly_pressure_std_dev)
    else:
        pressure = np.random.normal(normal_pressure_mean, normal_pressure_std_dev)

    new_data = {
        'Timestamp': last_timestamp,
        'Flow': np.random.normal(10, 0.5),
        'Pressure': pressure 
    }

    return new_data


# Initialize a lock
data_lock = threading.Lock()

def update_data(reductor_id):
    global data_df
    new_data = simulate_real_time_data(data_df)
        
    # Acquire lock before processing and updating data
    data_lock.acquire()

    try:
        # Make a copy of data_df for prediction
        data_for_prediction = data_df.copy()

        # Add new simulated data to the copy for prediction
        data_for_prediction = pd.concat([data_for_prediction, pd.DataFrame([new_data])], ignore_index=True)

        # Process the updated data for prediction
        sensitivity = get_sensitivity_for_reductor(reductor_id)
        update_cache(f'plot_data_{reductor_id}', (data_for_prediction, data_for_prediction['Timestamp'].min(), data_for_prediction['Timestamp'].max(), sensitivity))

        # Update the original data_df with new simulated data
        data_df = pd.concat([data_df, pd.DataFrame([new_data])], ignore_index=True)
        print(f"Updated data_df: {data_df.tail()}")

    finally:
        # Release the lock after processing is complete
        data_lock.release()


# Function to load scaler and model for a specific reductor
def load_reductor_assets(reductor_id):
    # Load the scaler
    scaler_filename = f'scaler_reductor{reductor_id}.save'
    scaler = joblib.load(scaler_filename)

    # Load the model
    model_filename = f'model_reductor{reductor_id}.h5'
    model = load_model(model_filename)

    return scaler, model

@app.route('/show_plot/<int:reductor_id>')
def show_plot(reductor_id):
    return render_template('plot.html', reductor_id=reductor_id)

@app.route('/get_plot/<int:reductor_id>')
def get_plot(reductor_id):
    # Load assets for the specified reductor
    try:
        scaler, model = load_reductor_assets(reductor_id)
    except FileNotFoundError:
        return "Scaler or model file not found for the specified reductor", 404

    # Retrieve data from cache
    cached_data = get_from_cache(f'plot_data_{reductor_id}')
    
    if cached_data is None:
        return "No data available for plotting", 404

    plot_data, start_date, end_date, sensitivity = cached_data

    # Assuming create_plotly_graph returns a Plotly figure
    fig, anomaly_count  = create_plotly_graph_full(plot_data, sensitivity, scaler, model, reductor_id)
    if fig is not None:
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON=graphJSON, anomaly_data=anomaly_count)
    else:
        return "No data available for plotting", 404

   
def get_data_from_db_for_reductor_date(start_date, end_date, reductor_id):
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor(dictionary=True) as cursor:
                query = """
                SELECT timestamp, flow, pressure
                FROM data
                WHERE timestamp BETWEEN %s AND %s AND reductorID = %s
                """
                cursor.execute(query, (start_date, end_date, reductor_id))
                rows = cursor.fetchall()
                if rows:
                    df2 = pd.DataFrame(rows)
                    return df2
                else:
                    return pd.DataFrame()  # Return an empty DataFrame if no data is found
    except Exception as err:  # Catching the database error
        print("Error in SQL operation:", err)
        return None  # Return None in case of an error

def get_data_from_db_for_reductor(reductor_id):
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor(dictionary=True) as cursor:
                # Updated query to fetch all data for a specific reductor
                query = """
                SELECT timestamp, flow, pressure
                FROM data
                WHERE reductorID = %s
                """
                cursor.execute(query, (reductor_id,))
                rows = cursor.fetchall()
                if rows:
                    df = pd.DataFrame(rows)
                    return df
                else:
                    return pd.DataFrame()  # Return an empty DataFrame if no data is found
    except Exception as err:  # Catching the database error
        print("Error in SQL operation:", err)
        return None  # Return None in case of an error
    
def fetch_reductor_name_and_town_id(reductor_id):
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                # Replace 'reductor_table' with your actual table name
                # and 'name', 'townID' with your actual column names
                query = """
                SELECT name, townID
                FROM reductor
                WHERE reductorID = %s
                """
                cursor.execute(query, (reductor_id,))
                row = cursor.fetchone()
                if row:
                    reductor_name, town_id = row
                    return reductor_name, town_id
                else:
                    return None, None  # No data found
    except mysql.connector.Error as err:
        print("Error in SQL operation:", err)
        return None  # Return a single None to indicate an error
    
def fetch_reductors():
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute("SELECT reductorID, name FROM reductor")
                return cursor.fetchall()
    except mysql.connector.Error as err:
        print("Database error:", err)
        return []
       
# Flask route handler for getting reductors
@app.route('/get_reductors', methods=['GET'])
def get_reductors():
    reductors = fetch_reductors()
    return jsonify(reductors)

def get_sensitivity_for_reductor(reductor_id):
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT sensibility FROM reductor WHERE reductorID = %s", (reductor_id,))
                row = cursor.fetchone()
                if row and row[0] is not None:
                    return row[0]
                else:
                    return None  # Sensitivity not found for reductor
    except mysql.connector.Error as err:
        print("Database error:", err)
        return None
    
def get_last_timestamp_from_db(reductor_id):
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                # Assuming the table name is 'data' and the timestamp column is 'timestamp'
                query = """
                SELECT timestamp
                FROM data
                WHERE reductorID = %s
                ORDER BY timestamp DESC
                LIMIT 1
                """
                cursor.execute(query, (reductor_id,))
                row = cursor.fetchone()
                if row:
                    return row[0]  # Return the timestamp
                else:
                    return None  # No data found
    except mysql.connector.Error as err:
        print("Error in SQL operation:", err)
        return None  # Return None to indicate an error

    
def send_alert_to_node_red(anomaly_data):
    url = "http://localhost:1880/anomaly-alert"
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=anomaly_data, headers=headers)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        
        # Debug: Print a message after sending the alert
        print("Alert sent to Node-RED.")
        
        if response.status_code == 200:
            print("Alert sent to Node-RED successfully")
        else:
            print("Failed to send alert")

        return response  # Return the response object
        
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that may occur during the request
        print("Error sending alert:", str(e))
        return None  # Return None to indicate an error


def create_plotly_graph_full(df, sensitivity, scaler, model, reductor_id):
    global last_alert_timestamp, daily_anomaly_counts
    df.columns = ['Timestamp', 'Flow', 'Pressure']
    # Ensure the 'Timestamp' column is in datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Separate numeric columns for resampling
    numeric_cols = df[['Flow', 'Pressure']]
    numeric_cols.set_index(df['Timestamp'], inplace=True)
    
    # Resample the numeric columns
    df_resampled = numeric_cols.resample('5T').mean()

    # Rejoin the Timestamp column
    df_resampled['Timestamp'] = df_resampled.index

    if df_resampled.empty:
        print("Resampled DataFrame is empty. Skipping processing.")
        return None

    df_resampled.dropna(inplace=True)
    if df_resampled.empty:
        print("DataFrame is empty after dropping NaNs.")
        return None
    
    # Scale the data
    pressure_scaled = scaler.transform(df_resampled[['Pressure']])
    predictions = model.predict(pressure_scaled)
    mse = np.mean(np.power(pressure_scaled - predictions, 2), axis=1)
    df_resampled['Reconstruction_Error'] = mse

    mse_threshold = np.quantile(df_resampled['Reconstruction_Error'], sensitivity)
    df_resampled['Predicted_Anomalies'] = df_resampled['Reconstruction_Error'] > mse_threshold
    
    # Track daily anomalies
    daily_anomaly_data = {}
    
    # Update daily anomaly counts and record days with more than 10 anomalies
    df_resampled['Date'] = df_resampled['Timestamp'].dt.date
    for date, group in df_resampled.groupby('Date'):
        daily_anomalies = group['Predicted_Anomalies'].sum()
        # Convert numpy.int64 to Python int
        daily_anomalies = int(daily_anomalies)
        daily_anomaly_counts[date] = daily_anomalies
        if daily_anomalies > 10:
            # Convert date to string
            date_str = date.strftime('%Y-%m-%d')
            daily_anomaly_data[date_str] = daily_anomalies

    print("Daily anomalies:", daily_anomaly_counts[date])
    anomaly_count = daily_anomaly_counts[date]
    print("Number of anomalies:", df_resampled['Predicted_Anomalies'].sum())

    # Define thresholds for severity levels
    severity_thresholds = {
        'mild': np.quantile(df_resampled[df_resampled['Predicted_Anomalies']]['Reconstruction_Error'], 0.75),
        'moderate': np.quantile(df_resampled[df_resampled['Predicted_Anomalies']]['Reconstruction_Error'], 0.90),
        'severe': np.quantile(df_resampled[df_resampled['Predicted_Anomalies']]['Reconstruction_Error'], 0.99)
    }

    # Function to classify anomaly severity
    def classify_anomaly_severity(row):
        if row['Predicted_Anomalies']:
            if row['Reconstruction_Error'] <= severity_thresholds['mild']:
                return 'Mild'
            elif row['Reconstruction_Error'] <= severity_thresholds['moderate']:
                return 'Moderate'
            else:
                return 'Severe'
        return 'Normal'

    # Apply the classification
    df_resampled['Anomaly_Severity'] = df_resampled.apply(classify_anomaly_severity, axis=1)

    # Function to get reductor name and town ID (assuming these are stored in your database)
    def get_reductor_details(reductor_id):
        reductor_name, town_id = fetch_reductor_name_and_town_id(reductor_id)
        return reductor_name, town_id

    # Fetch additional details
    reductor_name, town_id = get_reductor_details(reductor_id)
    # Check and send alerts for new anomalies
    for index, row in df_resampled.iterrows():
        if row['Predicted_Anomalies']:
            anomaly_timestamp = row['Timestamp']
            # Send an alert only if this anomaly is newer than the last alerted anomaly
            if last_alert_timestamp is None or anomaly_timestamp > last_alert_timestamp:
                # Prepare anomaly data
                anomaly_data = {
                    'timestamp': str(anomaly_timestamp),
                    'pressure': row['Pressure'],
                    'anomaly_severity': row['Anomaly_Severity'],
                    'reductorID': reductor_id,
                    'reductorName': reductor_name,
                    'townID': town_id
                }

                # Send alert
                send_alert_to_node_red(anomaly_data)

                # Update the last alert timestamp
                last_alert_timestamp = anomaly_timestamp


    # Create a Plotly figure
    fig = go.Figure()

    # Plot the normal pressure readings
    fig.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled['Pressure'], mode='lines', name='Pressure'))
    
    # Update line color for days with more than 10 anomalies
    for date, count in daily_anomaly_data.items():
        if count > 10:
            # Find all data points for the given date
            day_data = df_resampled[df_resampled['Date'] == pd.to_datetime(date)]
            fig.add_trace(go.Scatter(
                x=day_data.index,
                y=day_data['Pressure'],
                mode='lines',
                name=f'High Anomalies on {date} ({count})',
                line=dict(color='red', width=3)  # Highlight with a red, thicker line
            ))

    for severity, color in zip(['Mild', 'Moderate', 'Severe'], ['yellow', 'orange', 'red']):
        anomalies = df_resampled[df_resampled['Anomaly_Severity'] == severity]
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Pressure'], mode='markers', name=f'{severity} Anomalies', marker_color=color))

    fig.update_layout(title='Pressure Readings with Detected Anomalies', xaxis_title='Timestamp', yaxis_title='Scaled Pressure')

    # Set initial x-axis range to show the last day's data
    if not df_resampled.empty:
        max_date = df_resampled.index.max()
        one_day_ago = max_date - pd.Timedelta(days=1)
        fig.update_layout(xaxis_range=[one_day_ago, max_date])

    return fig, anomaly_count 

# Function to start data simulation for a reductor
def start_simulation_for_reductor(reductor_id, scheduler):
    global data_df
    global last_alert_timestamp
    data_df = get_data_from_db_for_reductor(reductor_id)
    if not data_df.empty:
        print(f"Loaded historical data for reductor {reductor_id}")
        last_alert_timestamp = get_last_timestamp_from_db(reductor_id)
        print(f"Last timestamp: {last_alert_timestamp}")
        try:
            scaler, model = load_reductor_assets(reductor_id)
        except FileNotFoundError:
            print(f"Scaler or model file not found for reductor {reductor_id}")
            return
        create_plotly_graph_full(data_df, get_sensitivity_for_reductor(reductor_id), scaler, model, reductor_id)
        # Cache the initial data
        #update_cache(f'plot_data_{reductor_id}', (data_df, data_df['Timestamp'].min(), data_df['Timestamp'].max()))

        # Schedule the update_data job
        scheduler.add_job(update_data, 'interval', seconds=5, args=[reductor_id])

    else:
        print(f"No historical data found for reductor {reductor_id}")
        
if __name__ == '__main__':



    # Start with reductor 6
    start_simulation_for_reductor(3, scheduler)

    # Start the scheduler
    scheduler.start()

    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)


