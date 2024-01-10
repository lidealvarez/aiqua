from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import io
import base64
from PIL import Image
import plotly.graph_objs as go
import plotly
import plotly.offline as pyo
import json

app = Flask(__name__)
app.secret_key = '6c061b2509dbc420431ad96a31042f4d'

# Load the trained model
model = load_model('anoeta_model.h5')

@app.route('/', methods=['GET'])
def index():
    plot_url = session.get('plot_url', None)
    return render_template('index.html', plot_url=plot_url)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if not file.filename.endswith('.csv'):
        return 'Invalid file format', 400

    df = pd.read_csv(file, delimiter=';')
    num_cols = df.shape[1]
    expected_num_cols = 5
    if num_cols != expected_num_cols:
        return f'Invalid file format: Expected {expected_num_cols} columns, found {num_cols}', 400
    df.columns = ['ID', 'Timestamp', 'Flow', 'Unknown', 'Pressure']

    # Get the date range or single month from the form
    detailed_month = request.form.get('detailed_month')
    start_month = request.form.get('start_month')
    end_month = request.form.get('end_month')

    if detailed_month:
        start_date = pd.to_datetime(detailed_month, format='%Y-%m').strftime("%Y-%m-01")
        end_date = pd.to_datetime(detailed_month, format='%Y-%m').to_period('M').end_time.strftime("%Y-%m-%d")
    elif start_month and end_month:
        start_date = pd.to_datetime(start_month, format='%Y-%m').strftime("%Y-%m-01")
        end_date = pd.to_datetime(end_month, format='%Y-%m').to_period('M').end_time.strftime("%Y-%m-%d")
    else:
        return "Please select either a single month or a start and end month", 400

    plot_url = create_plotly_graph(df, start_date, end_date)
    session['plot_url'] = plot_url
    print("plot_url", plot_url)
    return redirect(url_for('index'))

def create_plotly_graph(df, plot_start_date, plot_end_date):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.drop(['ID', 'Flow', 'Unknown'], axis=1)
    
    df.set_index('Timestamp', inplace=True)
    df_resampled = df.resample('5T').mean()
    df_resampled.dropna(inplace=True)

    # Fit and transform the data as in process_data
    scaler = StandardScaler()
    pressure_scaled = scaler.fit_transform(df_resampled[['Pressure']])
    
    predictions = model.predict(pressure_scaled)
    mse = np.mean(np.power(pressure_scaled - predictions, 2), axis=1)
    df_resampled['Reconstruction_Error'] = mse
    mse_threshold = np.quantile(df_resampled['Reconstruction_Error'], 0.9971)
    df_resampled['Predicted_Anomalies'] = df_resampled['Reconstruction_Error'] > mse_threshold

    # Filter data for the specified dates
    plot_data = df_resampled[plot_start_date:plot_end_date]

    # Plotly traces
    trace0 = go.Scatter(
        x=plot_data.index,
        y=plot_data['Pressure'],
        mode='lines',
        name='Pressure'
    )
    trace1 = go.Scatter(
        x=plot_data[plot_data['Predicted_Anomalies']].index,
        y=plot_data[plot_data['Predicted_Anomalies']]['Pressure'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10)
    )

    layout = go.Layout(
        title='Pressure Readings with Detected Anomalies',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Scaled Pressure')
    )

    fig = go.Figure(data=[trace0, trace1], layout=layout)

    # Save the figure as an HTML file
    static_dir = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    filename = f"plot_{uuid.uuid4()}.html"
    filepath = os.path.join(static_dir, filename)
    pyo.plot(fig, filename=filepath, auto_open=False)
    print("filepath",filename)
    return filename

import os
import uuid
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def process_data(df, plot_start_date, plot_end_date):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.drop(['ID', 'Flow', 'Unknown'], axis=1)
    
    df.set_index('Timestamp', inplace=True)
    df_resampled = df.resample('5T').mean()
    df_resampled.dropna(inplace=True)

    normal_data = df_resampled['2021-07-01':'2022-01-01']

    scaler = StandardScaler()
    normal_pressure_scaled = scaler.fit_transform(normal_data[['Pressure']])
    pressure_scaled = scaler.transform(df_resampled[['Pressure']])

    predictions = model.predict(pressure_scaled)
    mse = np.mean(np.power(pressure_scaled - predictions, 2), axis=1)
    df_resampled['Reconstruction_Error'] = mse
    mse_threshold = np.quantile(df_resampled['Reconstruction_Error'], 0.9971)
    df_resampled['Predicted_Anomalies'] = df_resampled['Reconstruction_Error'] > mse_threshold

    # Filter data for plotting
    plot_data = df_resampled[plot_start_date:plot_end_date]

    # Plot the filtered data
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(plot_data.index, plot_data['Pressure'], label='Pressure')
    anomalies = plot_data[plot_data['Predicted_Anomalies']]
    ax.scatter(anomalies.index, anomalies['Pressure'], color='r', label='Anomalies')
    ax.set_title('Pressure readings with detected anomalies')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Scaled Pressure')
    ax.legend()
    plt.close(fig)

    
    # Create a unique filename for the plot
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(app.root_path, 'static', filename)  # Adjusted filepath
    fig.savefig(filepath)

    # Resize and compress the image
    with Image.open(filepath) as image:
        # Convert to RGB mode
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Compress and save the image
        compressed_filename = f"{uuid.uuid4()}.jpg"
        compressed_filepath = os.path.join(app.root_path, 'static', compressed_filename)
        image.save(compressed_filepath, 'JPEG', quality=100)  # Adjust quality for your needs

    # Ensure to delete the original large image file to save space
    os.remove(filepath)

    plot_url = f'/static/{compressed_filename}'
    # Convert the 'Predicted_Anomalies' column to a list of booleans
    anomalies_list = df_resampled['Predicted_Anomalies'].tolist()
    print(plot_url)
    return plot_url, anomalies_list



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)