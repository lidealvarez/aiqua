import mysql.connector
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

# Database configuration
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'admin',
    'password': 'admin@eskola',
    'database': 'aiqua'
}

def get_data_from_db_for_reductor(start_date, end_date, reductor_id):
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
    except mysql.connector.Error as err:
        print("Error in SQL operation:", err)
        return None  # Return None to indicate an error

        
@app.route('/get_reductors', methods=['GET'])
def get_reductors():
    with mysql.connector.connect(**db_config) as conn:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT reductorID, name FROM reductor")
            reductors = cursor.fetchall()
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



@app.route('/', methods=['GET'])
def index():
    plot_url = session.get('plot_url', 'default_value_if_not_set')
    return render_template('indexdb.html', plot_url=plot_url)
  
@app.route('/load_data', methods=['POST'])
def load_data():
    reductor_id = request.form.get('reductor_id')
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
    print(start_date, end_date)
    # Fetch data from database for selected reductor
    df = get_data_from_db_for_reductor(start_date, end_date, reductor_id)
    if df is not None:
        print(df.head())
        print(reductor_id)
    else:
        print("No data returned or an error occurred.")
    
    # Fetch sensitivity for the selected reductor
    sensitivity = get_sensitivity_for_reductor(reductor_id)
    if sensitivity is None:
        return "Sensitivity threshold not found for the selected reductor", 404
    # Process data with the machine learning model and create a plot
    plot_url = create_plotly_graph(df, start_date, end_date, sensitivity)
    session['plot_url'] = plot_url
    return redirect(url_for('index'))

def create_plotly_graph(df, plot_start_date, plot_end_date, sensitivity):
    df.columns = ['Timestamp', 'Flow', 'Pressure']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df_resampled = df.resample('5T').mean()
    df_resampled.dropna(inplace=True)

    # Fit and transform the data
    scaler = StandardScaler()
    pressure_scaled = scaler.fit_transform(df_resampled[['Pressure']])
    
    # Generate predictions and calculate reconstruction error
    predictions = model.predict(pressure_scaled)
    mse = np.mean(np.power(pressure_scaled - predictions, 2), axis=1)
    df_resampled['Reconstruction_Error'] = mse

    # Use the dynamic sensitivity value for mse_threshold
    mse_threshold = np.quantile(df_resampled['Reconstruction_Error'], sensitivity)
    df_resampled['Predicted_Anomalies'] = df_resampled['Reconstruction_Error'] > mse_threshold

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
        else:
            return 'Normal'

    # Apply the classification
    df_resampled['Anomaly_Severity'] = df_resampled.apply(classify_anomaly_severity, axis=1)

    # Filter data for the specified dates
    plot_data = df_resampled[plot_start_date:plot_end_date]

    # Create a Plotly figure
    fig = go.Figure()

    # Plot the normal pressure readings
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Pressure'], mode='lines', name='Pressure'))

    # Separate the anomalies based on severity
    for severity, color in zip(['Mild', 'Moderate', 'Severe'], ['yellow', 'orange', 'red']):
        anomalies = plot_data[plot_data['Anomaly_Severity'] == severity]
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Pressure'], mode='markers', name=f'{severity} Anomalies', marker_color=color))

    # Add title and labels
    fig.update_layout(title='Pressure Readings with Detected Anomalies', xaxis_title='Timestamp', yaxis_title='Scaled Pressure')

    # Save the figure as an HTML file
    static_dir = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    filename = f"plot_{uuid.uuid4()}.html"
    filepath = os.path.join(static_dir, filename)
    pyo.plot(fig, filename=filepath, auto_open=False)
    return filename

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
