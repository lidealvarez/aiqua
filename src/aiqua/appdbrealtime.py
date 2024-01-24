import datetime
from numpy.random import Generator, MT19937
import joblib
import mysql.connector
from flask import Flask, request, jsonify, render_template, session
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

SQL_ERROR_MESSAGE = "Error in SQL operation:"

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

scheduler = BackgroundScheduler()

last_timestamp = None
last_alert_timestamp = None
global daily_anomaly_counts
# Global variable to keep track of anomalies per day
daily_anomaly_counts = defaultdict(int)

import numpy as np

# Create a random number generator using MT19937 algorithm
# Set the seed for reproducible results
seed_value = 42
rng = Generator(MT19937(seed_value))

def simulate_real_time_data(data_df):
    global last_timestamp
    # If last_timestamp is None, get the last timestamp from data_df
    if last_timestamp is None:
        if not data_df.empty:
            last_timestamp = data_df['Timestamp'].iloc[-1]
        else:
            # Set initial timestamp if data_df is empty
            last_timestamp = pd.Timestamp.now().floor('5T')

    # Increment timestamp by 5 minutes
    last_timestamp += pd.Timedelta(minutes=5)

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
    anomaly_pressure_mean = normal_pressure_mean * 2  
    anomaly_pressure_std_dev = normal_pressure_std_dev * 2  

    # Probability of an anomaly occurring
    anomaly_probability = 0.10 

    # Determine if this data point is an anomaly
    if rng.random() < anomaly_probability:
        pressure = rng.normal(anomaly_pressure_mean, anomaly_pressure_std_dev)
    else:
        pressure = rng.normal(normal_pressure_mean, normal_pressure_std_dev)

    new_data = {
        'Timestamp': last_timestamp,
        'Flow': rng.normal(10, 0.5),
        'Pressure': pressure 
    }

    return new_data


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

        sensitivity = get_sensitivity_for_reductor(reductor_id)
        update_cache(f'plot_data_{reductor_id}', (data_for_prediction, sensitivity))

        # Update the original data_df with new simulated data
        data_df = pd.concat([data_df, pd.DataFrame([new_data])], ignore_index=True)
        print(f"Updated data_df: {data_df.tail()}")

    finally:
        # Release the lock after processing is complete
        data_lock.release()


# Function to load scaler and model for a specific reductor
def load_reductor_assets(reductor_id):

    scaler_filename = f'scaler_reductor{reductor_id}.save'
    scaler = joblib.load(scaler_filename)


    model_filename = f'model_reductor{reductor_id}.h5'
    model = load_model(model_filename)

    return scaler, model

@app.route('/show_plot/<int:reductor_id>')
def show_plot(reductor_id):
    return render_template('plot.html', reductor_id=reductor_id)

@app.route('/get_plot/<int:reductor_id>')
def get_plot(reductor_id):

    try:
        scaler, model = load_reductor_assets(reductor_id)
    except FileNotFoundError:
        return "Scaler or model file not found for the specified reductor", 404


    cached_data = get_from_cache(f'plot_data_{reductor_id}')
    
    if cached_data is None:
        return "No data available for plotting", 404

    plot_data, sensitivity = cached_data


    fig, anomaly_count  = create_plotly_graph_full(plot_data, sensitivity, scaler, model)
    if fig is not None:
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON=graph_json , anomaly_data=anomaly_count)
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
                    return pd.DataFrame()  
    except Exception as err:  
        print(SQL_ERROR_MESSAGE , err)
        return None 

def get_data_from_db_for_reductor(reductor_id):
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor(dictionary=True) as cursor:

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
                    return pd.DataFrame()  
    except Exception as err:  
        print(SQL_ERROR_MESSAGE , err)
        return None 
    
def fetch_reductor_name_and_town_id(reductor_id):
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor() as cursor:
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
                    return None, None  
    except mysql.connector.Error as err:
        print(SQL_ERROR_MESSAGE , err)
        return None  
    
def fetch_reductors():
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute("SELECT reductorID, name FROM reductor")
                return cursor.fetchall()
    except mysql.connector.Error as err:
        print(SQL_ERROR_MESSAGE , err)
        return []
       
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
                    return None  
    except mysql.connector.Error as err:
        print(SQL_ERROR_MESSAGE , err)
        return None
    
def get_last_timestamp_from_db(reductor_id):
    try:
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor() as cursor:
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
                    return row[0]  
                else:
                    return None 
    except mysql.connector.Error as err:
        print(SQL_ERROR_MESSAGE , err)
        return None  

    
def send_alert_to_node_red(anomaly_data):
    url = "http://localhost:1880/anomaly-alert"
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=anomaly_data, headers=headers)
        response.raise_for_status()  
        
        print("Alert sent to Node-RED.")
        
        if response.status_code == 200:
            print("Alert sent to Node-RED successfully")
        else:
            print("Failed to send alert")

        return response  
        
    except requests.exceptions.RequestException as e:
        print("Error sending alert:", str(e))
        return None  


def preprocess_data(df):
    df.columns = ['Timestamp', 'Flow', 'Pressure']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    numeric_cols = df[['Flow', 'Pressure']]
    numeric_cols.set_index(df['Timestamp'], inplace=True)
    df_resampled = numeric_cols.resample('5T').mean()
    df_resampled['Timestamp'] = df_resampled.index
    df_resampled.dropna(inplace=True)
    return df_resampled

def scale_data(df_resampled, scaler, model):
    pressure_scaled = scaler.transform(df_resampled[['Pressure']])
    predictions = model.predict(pressure_scaled)
    mse = np.mean(np.power(pressure_scaled - predictions, 2), axis=1)
    df_resampled['Reconstruction_Error'] = mse
    return df_resampled

def detect_anomalies(df_resampled, sensitivity):
    mse_threshold = np.quantile(df_resampled['Reconstruction_Error'], sensitivity)
    df_resampled['Predicted_Anomalies'] = df_resampled['Reconstruction_Error'] > mse_threshold
    return df_resampled

def track_daily_anomalies(df_resampled):
    daily_anomaly_data = {}
    df_resampled['Date'] = df_resampled['Timestamp'].dt.date
    for date, group in df_resampled.groupby('Date'):
        daily_anomalies = group['Predicted_Anomalies'].sum()
        daily_anomaly_counts[date] = daily_anomalies
        if daily_anomalies > 10:
            date_str = date.strftime('%Y-%m-%d')
            daily_anomaly_data[date_str] = daily_anomalies
    return daily_anomaly_data


def create_plotly_figure(df_resampled):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled['Pressure'], mode='lines', name='Pressure'))

    for severity, color in zip(['Mild', 'Moderate', 'Severe'], ['yellow', 'orange', 'red']):
        anomalies = df_resampled[df_resampled['Anomaly_Severity'] == severity]
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Pressure'], mode='markers', name=f'{severity} Anomalies', marker_color=color))

    fig.update_layout(title='Pressure Readings with Detected Anomalies', xaxis_title='Timestamp', yaxis_title='Scaled Pressure')
    
    if not df_resampled.empty:
        max_date = df_resampled.index.max()
        one_day_ago = max_date - pd.Timedelta(days=1)
        fig.update_layout(xaxis_range=[one_day_ago, max_date])
    
    return fig

# Function to get reductor name and town ID
def get_reductor_details(reductor_id):
    reductor_name, town_id = fetch_reductor_name_and_town_id(reductor_id)
    return reductor_name, town_id
    
def check_and_send_alerts(df_resampled, reductor_id):
    global last_alert_timestamp
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

                send_alert_to_node_red(anomaly_data)

                last_alert_timestamp = anomaly_timestamp
    return last_alert_timestamp

def create_plotly_graph_full(df, sensitivity, scaler, model):
    global last_alert_timestamp
    daily_anomaly_data = {}
    
    df_resampled = preprocess_data(df)
    df_resampled = scale_data(df_resampled, scaler, model)
    df_resampled = detect_anomalies(df_resampled, sensitivity)
    
    # Update daily anomaly counts and record days with more than 10 anomalies
    df_resampled['Date'] = df_resampled['Timestamp'].dt.date
    for date, group in df_resampled.groupby('Date'):
        daily_anomalies = group['Predicted_Anomalies'].sum()
        daily_anomalies = int(daily_anomalies)
        daily_anomaly_counts[date] = daily_anomalies
        if daily_anomalies > 10:
            date_str = date.strftime('%Y-%m-%d')
            daily_anomaly_data[date_str] = daily_anomalies

    print("Daily anomalies:", daily_anomaly_counts[date])
    anomaly_count = daily_anomaly_counts[date]
    print("Number of anomalies:", df_resampled['Predicted_Anomalies'].sum())
    severity_thresholds = {
        'mild': np.quantile(df_resampled[df_resampled['Predicted_Anomalies']]['Reconstruction_Error'], 0.75),
        'moderate': np.quantile(df_resampled[df_resampled['Predicted_Anomalies']]['Reconstruction_Error'], 0.90),
        'severe': np.quantile(df_resampled[df_resampled['Predicted_Anomalies']]['Reconstruction_Error'], 0.99)
    }
    def classify_anomaly_severity(row):
        if row['Predicted_Anomalies']:
            if row['Reconstruction_Error'] <= severity_thresholds['mild']:
                return 'Mild'
            elif row['Reconstruction_Error'] <= severity_thresholds['moderate']:
                return 'Moderate'
            else:
                return 'Severe'
        return 'Normal'
    df_resampled['Anomaly_Severity'] = df_resampled.apply(classify_anomaly_severity, axis=1)
    last_alert_timestamp = check_and_send_alerts(df_resampled, current_simulation_reductor_id)
    fig = create_plotly_figure(df_resampled)
    
    return fig, anomaly_count




current_simulation_reductor_id = None

def start_simulation_for_reductor(reductor_id, scheduler):
    global data_df
    global last_alert_timestamp
    global current_simulation_reductor_id
    global last_timestamp
    

    # Stop the current simulation if it's for a different reductor
    if current_simulation_reductor_id is not None and current_simulation_reductor_id != reductor_id:
        scheduler.remove_job(f'update_data_job_{current_simulation_reductor_id}')
        print(f"Stopped simulation for reductor {current_simulation_reductor_id}")

    current_simulation_reductor_id = reductor_id

    # Clear the data for the previous reductor
    data_df = pd.DataFrame()
    last_alert_timestamp = None
    last_timestamp = None
    # Load historical data for the new reductor
    data_df = get_data_from_db_for_reductor(reductor_id)
    if not data_df.empty:
        print(f"Loaded historical data for reductor {reductor_id}")
        last_alert_timestamp = get_last_timestamp_from_db(reductor_id)
        print(f"Last alert timestamp: {last_alert_timestamp}")
        
        try:
            scaler, model = load_reductor_assets(reductor_id)
        except FileNotFoundError:
            print(f"Scaler or model file not found for reductor {reductor_id}")
            return

        create_plotly_graph_full(data_df, get_sensitivity_for_reductor(reductor_id), scaler, model)

        # Update the job in the scheduler for the new reductor
        job_id = f'update_data_job_{reductor_id}'
        if job_id not in [job.id for job in scheduler.get_jobs()]:
            scheduler.add_job(update_data, 'interval', seconds=5, id=job_id, args=[reductor_id])
            print(f"Started simulation for reductor {reductor_id}")
    else:
        print(f"No historical data found for reductor {reductor_id}")

        
@app.route('/node_red/start_simulation/<int:reductor_id>', methods=['POST'])
def start_simulation_from_node_red(reductor_id):
    try:
        start_simulation_for_reductor(reductor_id, scheduler)
        return jsonify({"status": "success", "message": f"Simulation started for reductor {reductor_id}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

        
if __name__ == '__main__':


    scheduler = BackgroundScheduler()
    scheduler.start()
    # Start with reductor 6
    start_simulation_for_reductor(5, scheduler)


    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)


