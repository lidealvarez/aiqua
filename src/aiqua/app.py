from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.offline as pyo
import os
import uuid

app = Flask(__name__)
app.secret_key = '6c061b2509dbc420431ad96a31042f4d'

# Load the trained model
model = load_model('anoeta_model.h5')


@app.route('/', methods=['GET'])
def index():
    plot_url = session.get('plot_url', 'default_value_if_not_set')
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)