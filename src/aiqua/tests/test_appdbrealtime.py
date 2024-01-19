import unittest
from unittest.mock import Mock, patch, MagicMock
import requests
import mysql.connector
import appdbrealtime
from appdbrealtime import load_reductor_assets, get_data_from_db_for_reductor_date, get_data_from_db_for_reductor, fetch_reductor_name_and_town_id, fetch_reductors, get_sensitivity_for_reductor, get_last_timestamp_from_db, send_alert_to_node_red, start_simulation_for_reductor, update_cache, get_from_cache, cache, simulate_real_time_data, update_data, start_simulation_for_reductor, scheduler
from appdbrealtime import app
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestCacheFunctions(unittest.TestCase):
    
    def test_update_cache(self):
        # Mock the global cache variable
        with patch.dict('appdbrealtime.cache', {}, clear=True):
            update_cache("test_key", "test_value")
            self.assertIn("test_key", cache)
            self.assertEqual(cache["test_key"], "test_value")

    def test_get_from_cache(self):
        # Mock the global cache variable
        with patch.dict('appdbrealtime.cache', {'another_key': 'another_value'}, clear=True):
            value = get_from_cache("another_key")
            self.assertEqual(value, "another_value")

    def test_get_from_cache_nonexistent_key(self):
        # Mock the global cache variable
        with patch.dict('appdbrealtime.cache', {}, clear=True):
            value = get_from_cache("nonexistent_key")
            self.assertIsNone(value)

class TestSimulateRealTimeData(unittest.TestCase):
    def setUp(self):
        # Setup a basic DataFrame for testing
        self.data_df = pd.DataFrame({
            'Timestamp': pd.date_range(start='2023-01-01', periods=3, freq='5T'),
            'Flow': [10, 10.5, 9.8],
            'Pressure': [5, 4.8, 5.2]
        })

    def test_simulate_with_non_empty_dataframe(self):
        # Mock the global last_timestamp
        with patch('appdbrealtime.last_timestamp', self.data_df['Timestamp'].iloc[-1]):
            new_data = simulate_real_time_data(self.data_df)
        
        # Assert that the function returns a dictionary
        self.assertIsInstance(new_data, dict)
        
        # Assert that the new data has the expected keys
        self.assertIn('Timestamp', new_data)
        self.assertIn('Flow', new_data)
        self.assertIn('Pressure', new_data)

        # Assert that the timestamp is correctly incremented
        expected_timestamp = self.data_df['Timestamp'].iloc[-1] + timedelta(minutes=5)
        self.assertEqual(new_data['Timestamp'], expected_timestamp)

    @patch('pandas.Timestamp.now')
    def test_simulate_with_empty_dataframe(self, mock_now):
        # Set a fixed current time as a Pandas Timestamp
        fixed_now = pd.Timestamp(datetime(2024, 1, 18, 17, 25, 0))
        mock_now.return_value = fixed_now

        # Reset last_timestamp before calling the function
        appdbrealtime.last_timestamp = None

        new_data = simulate_real_time_data(pd.DataFrame())

        # Similar assertions as above
        self.assertIsInstance(new_data, dict)
        self.assertIn('Timestamp', new_data)
        self.assertIn('Flow', new_data)
        self.assertIn('Pressure', new_data)

        # The timestamp is the fixed_now time rounded down and incremented by 5 minutes
        now_rounded_plus_5_minutes = fixed_now.floor('5T') + timedelta(minutes=5)
        self.assertEqual(new_data['Timestamp'], now_rounded_plus_5_minutes)

class TestUpdateData(unittest.TestCase):
    def setUp(self):
        # Setup a basic DataFrame for testing
        appdbrealtime.data_df = pd.DataFrame({
            'Timestamp': pd.date_range(start='2023-01-01', periods=3, freq='5T'),
            'Flow': [10, 10.5, 9.8],
            'Pressure': [5, 4.8, 5.2]
        })
        self.reductor_id = 1

    @patch('appdbrealtime.simulate_real_time_data')
    @patch('appdbrealtime.get_sensitivity_for_reductor')
    @patch('appdbrealtime.update_cache')
    def test_update_data(self, mock_update_cache, mock_get_sensitivity, mock_simulate_data):
        simulated_data = {
            'Timestamp': pd.Timestamp('2023-01-01 00:15:00'),
            'Flow': 10.2,
            'Pressure': 5.1
        }
        mock_simulate_data.return_value = simulated_data
        mock_get_sensitivity.return_value = 0.8  # Example sensitivity value
    
        update_data(self.reductor_id)
    
        # Test that simulate_real_time_data was called
        mock_simulate_data.assert_called_once()
    
        # Extract the arguments used in the last call to update_cache
        called_args, _ = mock_update_cache.call_args
        called_key, called_value = called_args

        # Check that the correct key was used in update_cache
        self.assertEqual(called_key, f'plot_data_{self.reductor_id}')

        # Check the contents of the value tuple passed to update_cache
        called_data_df, start_time, end_time, called_sensitivity = called_value
        self.assertEqual(len(called_data_df), 4)  # Initially 3 + 1 new data point
        self.assertTrue((called_data_df.iloc[-1] == pd.Series(simulated_data)).all())
        self.assertEqual(called_sensitivity, 0.8)

class TestLoadReductorAssets(unittest.TestCase):
    @patch('appdbrealtime.joblib.load')
    @patch('appdbrealtime.load_model')
    def test_load_reductor_assets(self, mock_load_model, mock_joblib_load):
        # Mock return values for joblib.load and load_model
        mock_scaler = MagicMock()
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_scaler
        mock_load_model.return_value = mock_model

        reductor_id = 1  # Example reductor ID
        scaler, model = load_reductor_assets(reductor_id)

        # Check if the functions were called with the correct filenames
        mock_joblib_load.assert_called_with(f'scaler_reductor{reductor_id}.save')
        mock_load_model.assert_called_with(f'model_reductor{reductor_id}.h5')

        # Check if the returned values are the mocked objects
        self.assertEqual(scaler, mock_scaler)
        self.assertEqual(model, mock_model)

class FlaskRoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        app.testing = True
        self.client = app.test_client()

    def test_show_plot(self):
        reductor_id = 1  # Example reductor ID
        response = self.client.get(f'/show_plot/{reductor_id}')

        self.assertEqual(response.status_code, 200)
        # Ensure that 'reductor_id' is correctly passed to the template
        # This check might be more complex depending on how your templates are set up
        self.assertIn(str(reductor_id), response.get_data(as_text=True))
    
    @patch('appdbrealtime.load_reductor_assets')
    @patch('appdbrealtime.get_from_cache')
    @patch('appdbrealtime.create_plotly_graph_full')
    def test_get_plot_success(self, mock_create_plotly_graph_full, mock_get_from_cache, mock_load_reductor_assets):
        # Mock dependencies
        mock_scaler = MagicMock()
        mock_model = MagicMock()
        mock_load_reductor_assets.return_value = (mock_scaler, mock_model)
        mock_get_from_cache.return_value = ('plot_data', 'start_date', 'end_date', 'sensitivity')
        mock_create_plotly_graph_full.return_value = ('fig', 'anomaly_count')

        reductor_id = 1  # Example reductor ID
        response = self.client.get(f'/get_plot/{reductor_id}')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('graphJSON', data)
        self.assertIn('anomaly_data', data)

    @patch('appdbrealtime.load_reductor_assets', side_effect=FileNotFoundError)
    def test_get_plot_file_not_found(self, mock_load_reductor_assets):
        reductor_id = 1  # Example reductor ID
        response = self.client.get(f'/get_plot/{reductor_id}')

        self.assertEqual(response.status_code, 404)
        self.assertIn("Scaler or model file not found for the specified reductor", response.get_data(as_text=True))

    @patch('appdbrealtime.load_reductor_assets')
    @patch('appdbrealtime.get_from_cache', return_value=None)
    def test_get_plote_no_data_available(self, mock_get_from_cache, mock_load_reductor_assets):
        mock_scaler = MagicMock()
        mock_model = MagicMock()
        mock_load_reductor_assets.return_value = (mock_scaler, mock_model)

        reductor_id = 1  # Example reductor ID
        response = self.client.get(f'/get_plot/{reductor_id}')

        self.assertEqual(response.status_code, 404)
        self.assertIn("No data available for plotting", response.get_data(as_text=True))
        
class TestGetDataFromDbForReductorDate(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_successful(self, mock_connect):
        # Mocking database response
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'timestamp': '2023-01-01 00:00:00', 'flow': 10, 'pressure': 5},
            {'timestamp': '2023-01-01 00:05:00', 'flow': 12, 'pressure': 6}
        ]

        # Test the function
        result = get_data_from_db_for_reductor_date('2023-01-01', '2023-01-02', 1)

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(column in result for column in ['timestamp', 'flow', 'pressure']))

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_no_rows_found(self, mock_connect):
        # Mocking empty response
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        result = get_data_from_db_for_reductor_date('2023-01-01', '2023-01-02', 1)

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_db_error(self, mock_connect):
        # Simulating a database error
        mock_connect.side_effect = Exception("Database error")

        result = get_data_from_db_for_reductor_date('2023-01-01', '2023-01-02', 1)

        # Assertions
        self.assertIsNone(result)
        
class TestGetDataFromDbForReductor(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_successful(self, mock_connect):
        # Mocking database response
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'timestamp': '2023-01-01 00:00:00', 'flow': 10, 'pressure': 5},
            {'timestamp': '2023-01-01 00:05:00', 'flow': 12, 'pressure': 6}
        ]

        # Test the function
        result = get_data_from_db_for_reductor(1)

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(column in result for column in ['timestamp', 'flow', 'pressure']))

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_no_rows_found(self, mock_connect):
        # Mocking empty response
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        result = get_data_from_db_for_reductor(1)

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_db_error(self, mock_connect):
        # Simulating a database error
        mock_connect.side_effect = Exception("Database error")

        result = get_data_from_db_for_reductor(1)

        # Assertions
        self.assertIsNone(result)
        
class TestFetchReductorNameAndTownId(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductor_name_and_town_id_success(self, mock_connect):
        # Mocking database response
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ('ReductorName', 'TownID')

        # Test the function
        reductor_name, town_id = fetch_reductor_name_and_town_id(1)

        # Assertions
        self.assertEqual(reductor_name, 'ReductorName')
        self.assertEqual(town_id, 'TownID')

    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductor_name_and_town_id_no_data(self, mock_connect):
        # Mocking no data found
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        reductor_name, town_id = fetch_reductor_name_and_town_id(1)

        # Assertions
        self.assertIsNone(reductor_name)
        self.assertIsNone(town_id)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductor_name_and_town_id_db_error(self, mock_connect):
        # Simulating a database error
        mock_connect.side_effect = mysql.connector.Error("Database error")

        result = fetch_reductor_name_and_town_id(1)

        # Assertions
        self.assertIsNone(result)

class TestFetchReductors(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductors_success(self, mock_connect):
        # Mocking database response
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'reductorID': 1, 'name': 'Reductor1'},
            {'reductorID': 2, 'name': 'Reductor2'}
        ]

        # Test the function
        result = fetch_reductors()

        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], 'Reductor1')
        self.assertEqual(result[1]['name'], 'Reductor2')

    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductors_db_error(self, mock_connect):
        # Simulating a database error
        mock_connect.side_effect = mysql.connector.Error("Database error")

        result = fetch_reductors()

        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

class FlaskGetReductorsTestCase(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    @patch('appdbrealtime.fetch_reductors')
    def test_get_reductors_success(self, mock_fetch_reductors):
        # Mocking the fetch_reductors function to return a successful response
        mock_fetch_reductors.return_value = [
            {'reductorID': 1, 'name': 'Reductor1'},
            {'reductorID': 2, 'name': 'Reductor2'}
        ]

        response = self.client.get('/get_reductors')
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['name'], 'Reductor1')
        self.assertEqual(data[1]['name'], 'Reductor2')

    @patch('appdbrealtime.fetch_reductors')
    def test_get_reductors_empty(self, mock_fetch_reductors):
        # Mocking the fetch_reductors function to return an empty list
        mock_fetch_reductors.return_value = []

        response = self.client.get('/get_reductors')
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data), 0)

class TestGetSensitivityForReductor(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_sensitivity_successful(self, mock_connect):
        # Mocking database response for successful retrieval
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (0.8,)

        # Test the function
        result = get_sensitivity_for_reductor(1)

        # Assertions
        self.assertEqual(result, 0.8)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_sensitivity_no_data_found(self, mock_connect):
        # Mocking no data found
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        result = get_sensitivity_for_reductor(1)

        # Assertions
        self.assertIsNone(result)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_sensitivity_db_error(self, mock_connect):
        # Simulating a database error
        mock_connect.side_effect = mysql.connector.Error("Database error")

        result = get_sensitivity_for_reductor(1)

        # Assertions
        self.assertIsNone(result)
        
class TestGetLastTimestampFromDb(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_last_timestamp_successful(self, mock_connect):
        # Mocking database response for successful retrieval
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        last_timestamp = datetime.now()
        mock_cursor.fetchone.return_value = (last_timestamp,)

        # Test the function
        result = get_last_timestamp_from_db(1)

        # Assertions
        self.assertEqual(result, last_timestamp)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_last_timestamp_no_data_found(self, mock_connect):
        # Mocking no data found
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        result = get_last_timestamp_from_db(1)

        # Assertions
        self.assertIsNone(result)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_last_timestamp_db_error(self, mock_connect):
        # Simulating a database error
        mock_connect.side_effect = mysql.connector.Error("Database error")

        result = get_last_timestamp_from_db(1)

        # Assertions
        self.assertIsNone(result)

class TestSendAlertToNodeRed(unittest.TestCase):
    @patch('appdbrealtime.requests.post')
    def test_send_alert_success(self, mock_post):
        # Mocking successful POST request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test the function
        anomaly_data = {'test': 'data'}
        response = send_alert_to_node_red(anomaly_data)

        # Assertions
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)

    @patch('appdbrealtime.requests.post')
    def test_send_alert_failure(self, mock_post):
        # Mocking failed POST request with status code error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Error")
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        response = send_alert_to_node_red({'test': 'data'})

        # Assertions
        self.assertIsNone(response)

    @patch('appdbrealtime.requests.post')
    def test_send_alert_exception(self, mock_post):
        # Simulating a request exception
        mock_post.side_effect = requests.exceptions.RequestException("Request Exception")

        response = send_alert_to_node_red({'test': 'data'})

        # Assertions
        self.assertIsNone(response)

class TestStartSimulationForReductor(unittest.TestCase):
    @patch('appdbrealtime.get_data_from_db_for_reductor')
    @patch('appdbrealtime.get_last_timestamp_from_db')
    @patch('appdbrealtime.load_reductor_assets')
    @patch('appdbrealtime.create_plotly_graph_full')
    @patch('appdbrealtime.scheduler.add_job')
    def test_start_simulation_success(self, mock_add_job, mock_create_plotly_graph, mock_load_reductor_assets, mock_get_last_timestamp, mock_get_data_from_db):
        # Mocking successful data retrieval and asset loading
        mock_get_data_from_db.return_value = MagicMock(empty=False)
        mock_get_last_timestamp.return_value = '2023-01-01 00:00:00'
        mock_load_reductor_assets.return_value = (MagicMock(), MagicMock())

        # Create a mock scheduler
        mock_scheduler = MagicMock()
        mock_scheduler.add_job = mock_add_job  # Assign the add_job mock to the mock_scheduler

        # Test the function
        start_simulation_for_reductor(1, mock_scheduler)

        # Assertions
        mock_get_data_from_db.assert_called_with(1)
        mock_get_last_timestamp.assert_called_with(1)
        mock_load_reductor_assets.assert_called_with(1)
        mock_create_plotly_graph.assert_called()

        # Check if add_job was called
        mock_add_job.assert_called()  # If exact arguments are not important


    @patch('appdbrealtime.get_data_from_db_for_reductor')
    def test_start_simulation_no_data(self, mock_get_data_from_db):
        # Mocking no historical data found
        mock_get_data_from_db.return_value = MagicMock(empty=True)

        # Create a mock scheduler
        mock_scheduler = MagicMock()

        start_simulation_for_reductor(1, mock_scheduler)

        # Assertions
        mock_get_data_from_db.assert_called_with(1)

    @patch('appdbrealtime.get_data_from_db_for_reductor')
    @patch('appdbrealtime.load_reductor_assets', side_effect=FileNotFoundError)
    def test_start_simulation_missing_assets(self, mock_load_reductor_assets, mock_get_data_from_db):
        # Mocking successful data retrieval but missing assets
        mock_get_data_from_db.return_value = MagicMock(empty=False)

        # Create a mock scheduler
        mock_scheduler = MagicMock()

        start_simulation_for_reductor(1, mock_scheduler)

        # Assertions
        mock_get_data_from_db.assert_called_with(1)
        mock_load_reductor_assets.assert_called_with(1)

        
class TestAppDbRealTime(unittest.TestCase):
    
    def setUp(self):
        app.config['DEBUG'] = True

    @patch('appdbrealtime.start_simulation_for_reductor')
    def test_start_scheduler(self, mock_scheduler):
        # Call your function which includes scheduler.start()
        start_simulation_for_reductor(6, mock_scheduler)
        
        # Assert that the start method of the mock_scheduler was called
        mock_scheduler.start.assert_called_once()

    def test_flask_app_config(self):
        # Testing the debug configuration
        self.assertTrue(app.debug)
        self.assertTrue(app.config['DEBUG'])
        
if __name__ == '__main__':
    unittest.main()
