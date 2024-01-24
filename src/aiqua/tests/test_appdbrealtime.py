from io import StringIO
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
from sklearn.preprocessing import StandardScaler

CACHE_KEY = 'appdbrealtime.cache'
DB_ERROR_MESSAGE = "Database error"
TEST_ERROR_MESSAGE = "Test error"

class TestCacheFunctions(unittest.TestCase):
    
    def test_update_cache(self):
        # Mock the global cache variable
        with patch.dict(CACHE_KEY, {}, clear=True):
            update_cache("test_key", "test_value")
            self.assertIn("test_key", cache)
            self.assertEqual(cache["test_key"], "test_value")

    def test_get_from_cache(self):
        with patch.dict(CACHE_KEY, {'another_key': 'another_value'}, clear=True):
            value = get_from_cache("another_key")
            self.assertEqual(value, "another_value")

    def test_get_from_cache_nonexistent_key(self):
        with patch.dict(CACHE_KEY, {}, clear=True):
            value = get_from_cache("nonexistent_key")
            self.assertIsNone(value)

class TestSimulateRealTimeData(unittest.TestCase):
    def setUp(self):
        self.data_df = pd.DataFrame({
            'Timestamp': pd.date_range(start='2023-01-01', periods=3, freq='5T'),
            'Flow': [10, 10.5, 9.8],
            'Pressure': [5, 4.8, 5.2]
        })

    def test_simulate_with_non_empty_dataframe(self):

        with patch('appdbrealtime.last_timestamp', self.data_df['Timestamp'].iloc[-1]):
            new_data = simulate_real_time_data(self.data_df)
        
        self.assertIsInstance(new_data, dict)
        
        self.assertIn('Timestamp', new_data)
        self.assertIn('Flow', new_data)
        self.assertIn('Pressure', new_data)

        expected_timestamp = self.data_df['Timestamp'].iloc[-1] + timedelta(minutes=5)
        self.assertEqual(new_data['Timestamp'], expected_timestamp)

    @patch('pandas.Timestamp.now')
    def test_simulate_with_empty_dataframe(self, mock_now):
        fixed_now = pd.Timestamp(datetime(2024, 1, 18, 17, 25, 0))
        mock_now.return_value = fixed_now

        appdbrealtime.last_timestamp = None

        new_data = simulate_real_time_data(pd.DataFrame())

        self.assertIsInstance(new_data, dict)
        self.assertIn('Timestamp', new_data)
        self.assertIn('Flow', new_data)
        self.assertIn('Pressure', new_data)

        now_rounded_plus_5_minutes = fixed_now.floor('5T') + timedelta(minutes=5)
        self.assertEqual(new_data['Timestamp'], now_rounded_plus_5_minutes)

class TestUpdateData(unittest.TestCase):
    def setUp(self):
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
    
        mock_simulate_data.assert_called_once()
    
        called_args, _ = mock_update_cache.call_args
        called_key, called_value = called_args

        self.assertEqual(called_key, f'plot_data_{self.reductor_id}')

        called_data_df, called_sensitivity = called_value
        self.assertEqual(len(called_data_df), 4)  
        self.assertTrue((called_data_df.iloc[-1] == pd.Series(simulated_data)).all())
        self.assertEqual(called_sensitivity, 0.8)

class TestLoadReductorAssets(unittest.TestCase):
    @patch('appdbrealtime.joblib.load')
    @patch('appdbrealtime.load_model')
    def test_load_reductor_assets(self, mock_load_model, mock_joblib_load):
        mock_scaler = MagicMock()
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_scaler
        mock_load_model.return_value = mock_model

        reductor_id = 1  # Example reductor ID
        scaler, model = load_reductor_assets(reductor_id)

        mock_joblib_load.assert_called_with(f'scaler_reductor{reductor_id}.save')
        mock_load_model.assert_called_with(f'model_reductor{reductor_id}.h5')

        self.assertEqual(scaler, mock_scaler)
        self.assertEqual(model, mock_model)

class FlaskRoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        app.testing = True
        self.client = app.test_client()

    def test_show_plot(self):
        reductor_id = 1  
        response = self.client.get(f'/show_plot/{reductor_id}')

        self.assertEqual(response.status_code, 200)
        self.assertIn(str(reductor_id), response.get_data(as_text=True))
    
    @patch('appdbrealtime.load_reductor_assets')
    @patch('appdbrealtime.get_from_cache')
    @patch('appdbrealtime.create_plotly_graph_full')
    def test_get_plot_success(self, mock_create_plotly_graph_full, mock_get_from_cache, mock_load_reductor_assets):
        mock_scaler = MagicMock()
        mock_model = MagicMock()
        mock_load_reductor_assets.return_value = (mock_scaler, mock_model)
        mock_get_from_cache.return_value = ('plot_data', 'sensitivity')
        mock_create_plotly_graph_full.return_value = ('fig', 'anomaly_count')

        reductor_id = 1  
        response = self.client.get(f'/get_plot/{reductor_id}')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('graphJSON', data)
        self.assertIn('anomaly_data', data)

    @patch('appdbrealtime.load_reductor_assets', side_effect=FileNotFoundError)
    def test_get_plot_file_not_found(self, mock_load_reductor_assets):
        reductor_id = 1  
        response = self.client.get(f'/get_plot/{reductor_id}')

        self.assertEqual(response.status_code, 404)
        self.assertIn("Scaler or model file not found for the specified reductor", response.get_data(as_text=True))

    @patch('appdbrealtime.load_reductor_assets')
    @patch('appdbrealtime.get_from_cache', return_value=None)
    def test_get_plote_no_data_available(self, mock_get_from_cache, mock_load_reductor_assets):
        mock_scaler = MagicMock()
        mock_model = MagicMock()
        mock_load_reductor_assets.return_value = (mock_scaler, mock_model)

        reductor_id = 1  
        response = self.client.get(f'/get_plot/{reductor_id}')

        self.assertEqual(response.status_code, 404)
        self.assertIn("No data available for plotting", response.get_data(as_text=True))
        
class TestGetDataFromDbForReductorDate(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_successful(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'timestamp': '2023-01-01 00:00:00', 'flow': 10, 'pressure': 5},
            {'timestamp': '2023-01-01 00:05:00', 'flow': 12, 'pressure': 6}
        ]

        result = get_data_from_db_for_reductor_date('2023-01-01', '2023-01-02', 1)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(column in result for column in ['timestamp', 'flow', 'pressure']))

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_no_rows_found(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        result = get_data_from_db_for_reductor_date('2023-01-01', '2023-01-02', 1)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_db_error(self, mock_connect):
        mock_connect.side_effect = Exception(DB_ERROR_MESSAGE)

        result = get_data_from_db_for_reductor_date('2023-01-01', '2023-01-02', 1)

        self.assertIsNone(result)
        
class TestGetDataFromDbForReductor(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_successful(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'timestamp': '2023-01-01 00:00:00', 'flow': 10, 'pressure': 5},
            {'timestamp': '2023-01-01 00:05:00', 'flow': 12, 'pressure': 6}
        ]

        result = get_data_from_db_for_reductor(1)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(column in result for column in ['timestamp', 'flow', 'pressure']))

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_no_rows_found(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        result = get_data_from_db_for_reductor(1)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_data_db_error(self, mock_connect):
        mock_connect.side_effect = Exception(DB_ERROR_MESSAGE)

        result = get_data_from_db_for_reductor(1)

        self.assertIsNone(result)
        
class TestFetchReductorNameAndTownId(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductor_name_and_town_id_success(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ('ReductorName', 'TownID')

        reductor_name, town_id = fetch_reductor_name_and_town_id(1)

        self.assertEqual(reductor_name, 'ReductorName')
        self.assertEqual(town_id, 'TownID')

    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductor_name_and_town_id_no_data(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        reductor_name, town_id = fetch_reductor_name_and_town_id(1)

        self.assertIsNone(reductor_name)
        self.assertIsNone(town_id)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductor_name_and_town_id_db_error(self, mock_connect):
        mock_connect.side_effect = mysql.connector.Error(DB_ERROR_MESSAGE)

        result = fetch_reductor_name_and_town_id(1)

        self.assertIsNone(result)

class TestFetchReductors(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductors_success(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'reductorID': 1, 'name': 'Reductor1'},
            {'reductorID': 2, 'name': 'Reductor2'}
        ]

        result = fetch_reductors()

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], 'Reductor1')
        self.assertEqual(result[1]['name'], 'Reductor2')

    @patch('appdbrealtime.mysql.connector.connect')
    def test_fetch_reductors_db_error(self, mock_connect):
        mock_connect.side_effect = mysql.connector.Error(DB_ERROR_MESSAGE)

        result = fetch_reductors()

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

class FlaskGetReductorsTestCase(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    @patch('appdbrealtime.fetch_reductors')
    def test_get_reductors_success(self, mock_fetch_reductors):
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
        mock_fetch_reductors.return_value = []

        response = self.client.get('/get_reductors')
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data), 0)

class TestGetSensitivityForReductor(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_sensitivity_successful(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (0.8,)

        result = get_sensitivity_for_reductor(1)

        self.assertEqual(result, 0.8)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_sensitivity_no_data_found(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        result = get_sensitivity_for_reductor(1)

        self.assertIsNone(result)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_sensitivity_db_error(self, mock_connect):
        mock_connect.side_effect = mysql.connector.Error(DB_ERROR_MESSAGE)

        result = get_sensitivity_for_reductor(1)

        self.assertIsNone(result)
        
class TestGetLastTimestampFromDb(unittest.TestCase):
    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_last_timestamp_successful(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        last_timestamp = datetime.now()
        mock_cursor.fetchone.return_value = (last_timestamp,)

        result = get_last_timestamp_from_db(1)

        self.assertEqual(result, last_timestamp)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_last_timestamp_no_data_found(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        result = get_last_timestamp_from_db(1)

        self.assertIsNone(result)

    @patch('appdbrealtime.mysql.connector.connect')
    def test_get_last_timestamp_db_error(self, mock_connect):
        mock_connect.side_effect = mysql.connector.Error(DB_ERROR_MESSAGE)

        result = get_last_timestamp_from_db(1)

        self.assertIsNone(result)

class TestSendAlertToNodeRed(unittest.TestCase):
    @patch('appdbrealtime.requests.post')
    def test_send_alert_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        anomaly_data = {'test': 'data'}
        response = send_alert_to_node_red(anomaly_data)

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)

    @patch('appdbrealtime.requests.post')
    def test_send_alert_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Error")
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        response = send_alert_to_node_red({'test': 'data'})

        self.assertIsNone(response)

    @patch('appdbrealtime.requests.post')
    def test_send_alert_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("Request Exception")

        response = send_alert_to_node_red({'test': 'data'})

        self.assertIsNone(response)

class TestStartSimulationForReductor(unittest.TestCase):
    @patch('appdbrealtime.get_data_from_db_for_reductor')
    @patch('appdbrealtime.get_last_timestamp_from_db')
    @patch('appdbrealtime.load_reductor_assets')
    @patch('appdbrealtime.create_plotly_graph_full')
    @patch('appdbrealtime.scheduler.add_job')
    @patch('appdbrealtime.scheduler.remove_job')
    def test_start_simulation_success(self, mock_remove_job, mock_add_job, mock_create_plotly_graph, mock_load_reductor_assets, mock_get_last_timestamp, mock_get_data_from_db):
        mock_get_data_from_db.return_value = MagicMock(empty=False)
        mock_get_last_timestamp.return_value = '2023-01-01 00:00:00'
        mock_load_reductor_assets.return_value = (MagicMock(), MagicMock())

        mock_scheduler = MagicMock()
        mock_scheduler.add_job = mock_add_job  
        mock_scheduler.remove_job = mock_remove_job  


        appdbrealtime.current_simulation_reductor_id = None
        appdbrealtime.last_timestamp = None

        start_simulation_for_reductor(1, mock_scheduler)

        mock_get_data_from_db.assert_called_with(1)
        mock_get_last_timestamp.assert_called_with(1)
        mock_load_reductor_assets.assert_called_with(1)
        mock_create_plotly_graph.assert_called()

        mock_add_job.assert_called()

    @patch('appdbrealtime.get_data_from_db_for_reductor')
    def test_start_simulation_no_data(self, mock_get_data_from_db):
        mock_get_data_from_db.return_value = MagicMock(empty=True)

        mock_scheduler = MagicMock()

        appdbrealtime.current_simulation_reductor_id = None
        appdbrealtime.last_timestamp = None

        start_simulation_for_reductor(1, mock_scheduler)

        mock_get_data_from_db.assert_called_with(1)

    @patch('appdbrealtime.get_data_from_db_for_reductor')
    @patch('appdbrealtime.load_reductor_assets', side_effect=FileNotFoundError)
    def test_start_simulation_missing_assets(self, mock_load_reductor_assets, mock_get_data_from_db):
        mock_get_data_from_db.return_value = MagicMock(empty=False)

        mock_scheduler = MagicMock()

        appdbrealtime.current_simulation_reductor_id = None
        appdbrealtime.last_timestamp = None

        start_simulation_for_reductor(1, mock_scheduler)

        mock_get_data_from_db.assert_called_with(1)
        mock_load_reductor_assets.assert_called_with(1)

    @patch('appdbrealtime.get_data_from_db_for_reductor')
    @patch('appdbrealtime.get_last_timestamp_from_db')
    @patch('appdbrealtime.load_reductor_assets')
    @patch('appdbrealtime.create_plotly_graph_full')
    @patch('appdbrealtime.scheduler.add_job')
    @patch('appdbrealtime.scheduler.remove_job')
    def test_stop_existing_simulation_for_different_reductor(self, mock_remove_job, mock_add_job, mock_create_plotly_graph, mock_load_reductor_assets, mock_get_last_timestamp, mock_get_data_from_db):
        mock_get_data_from_db.return_value = MagicMock(empty=False)
        mock_get_last_timestamp.return_value = '2023-01-01 00:00:00'
        mock_load_reductor_assets.return_value = (MagicMock(), MagicMock())

        mock_scheduler = MagicMock()
        mock_scheduler.add_job = mock_add_job
        mock_scheduler.remove_job = mock_remove_job

        appdbrealtime.current_simulation_reductor_id = 2

        start_simulation_for_reductor(1, mock_scheduler)

        mock_remove_job.assert_called_with('update_data_job_2')
        mock_get_data_from_db.assert_called_with(1)
        mock_get_last_timestamp.assert_called_with(1)
        mock_load_reductor_assets.assert_called_with(1)
        mock_create_plotly_graph.assert_called()
        mock_add_job.assert_called()
        
class FlaskNodeRedTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()

    @patch('appdbrealtime.start_simulation_for_reductor')
    def test_start_simulation_success(self, mock_start_simulation):
        mock_start_simulation.return_value = None

        reductor_id = 1 
        response = self.client.post(f'/node_red/start_simulation/{reductor_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'success')
        self.assertIn(f'Simulation started for reductor {reductor_id}', data['message'])

    @patch('appdbrealtime.start_simulation_for_reductor')
    def test_start_simulation_failure(self, mock_start_simulation):
        mock_start_simulation.side_effect = Exception(TEST_ERROR_MESSAGE)

        reductor_id = 1  # Example reductor ID
        response = self.client.post(f'/node_red/start_simulation/{reductor_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'error')
        self.assertIn(TEST_ERROR_MESSAGE, data['message'])
                  
class TestAppDbRealTime(unittest.TestCase):
    
    def setUp(self):
        app.config['DEBUG'] = True

    def test_flask_app_config(self):
        self.assertTrue(app.debug)
        self.assertTrue(app.config['DEBUG'])


class FlaskNodeRedTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()

    @patch('appdbrealtime.start_simulation_for_reductor')
    def test_start_simulation_success(self, mock_start_simulation):
        mock_start_simulation.return_value = None

        reductor_id = 1  
        response = self.client.post(f'/node_red/start_simulation/{reductor_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'success')
        self.assertIn(f'Simulation started for reductor {reductor_id}', data['message'])

    @patch('appdbrealtime.start_simulation_for_reductor')
    def test_start_simulation_failure(self, mock_start_simulation):
        mock_start_simulation.side_effect = Exception('Test error')

        reductor_id = 1 
        response = self.client.post(f'/node_red/start_simulation/{reductor_id}')
        self.assertEqual(response.status_code, 200) 
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'error')
        self.assertIn('Test error', data['message'])

                   
class TestAppDbRealTime(unittest.TestCase):
    
    def setUp(self):
        app.config['DEBUG'] = True

    def test_flask_app_config(self):
        self.assertTrue(app.debug)
        self.assertTrue(app.config['DEBUG'])
        
class TestPreprocessData(unittest.TestCase):

    def setUp(self):
        data = """Timestamp,Flow,Pressure
                  2024-01-01 00:00:00,10,100
                  2024-01-01 00:02:00,15,105
                  2024-01-01 00:06:00,20,110"""
        self.df = pd.read_csv(StringIO(data), parse_dates=['Timestamp'])

    def test_column_names(self):
        processed_df = appdbrealtime.preprocess_data(self.df)
        self.assertListEqual(processed_df.columns.tolist(), ['Flow', 'Pressure', 'Timestamp'])

    def test_timestamp_conversion(self):
        processed_df = appdbrealtime.preprocess_data(self.df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_df['Timestamp']))

    def test_resampling(self):
        processed_df = appdbrealtime.preprocess_data(self.df)
        expected_freq = pd.tseries.frequencies.to_offset('5T') 
        actual_freq = pd.tseries.frequencies.to_offset(processed_df.index.freqstr)

        self.assertEqual(actual_freq, expected_freq)


    def test_nan_handling(self):
        self.df.loc[3] = [pd.NaT, None, None]
        processed_df = appdbrealtime.preprocess_data(self.df)
        self.assertFalse(processed_df.isna().any().any())

    def test_output_structure(self):
        processed_df = appdbrealtime.preprocess_data(self.df)
        self.assertIsInstance(processed_df, pd.DataFrame)

class MockModel:
    """ Mock model for testing purposes. """
    def predict(self, input_data):
        # For testing, return a simple transformation of input_data
        return input_data * 0.5

class TestScaleData(unittest.TestCase):

    def setUp(self):

        data = {"Pressure": [100, 105, 110]}
        self.df_resampled = pd.DataFrame(data)

        self.scaler = StandardScaler()
        self.scaler.fit(self.df_resampled[['Pressure']])
        self.model = MockModel() 

    def test_scaling(self):
        scaled_df = appdbrealtime.scale_data(self.df_resampled, self.scaler, self.model)

        pressure_scaled_direct = self.scaler.transform(self.df_resampled[['Pressure']])

        predictions_direct = self.model.predict(pressure_scaled_direct)

        mse_direct = np.mean(np.power(pressure_scaled_direct - predictions_direct, 2), axis=1)

        mse_from_scaled_df = scaled_df['Reconstruction_Error'].values

        self.assertIn('Reconstruction_Error', scaled_df.columns)

        np.testing.assert_array_almost_equal(mse_direct, mse_from_scaled_df, decimal=7)



    def test_model_predictions(self):
        scaled_df = appdbrealtime.scale_data(self.df_resampled, self.scaler, self.model)
        self.assertIn('Reconstruction_Error', scaled_df.columns)
        self.assertTrue(all(scaled_df['Reconstruction_Error'] >= 0))  

    def test_output_structure(self):
        scaled_df = appdbrealtime.scale_data(self.df_resampled, self.scaler, self.model)
        self.assertIsInstance(scaled_df, pd.DataFrame)
        self.assertIn('Reconstruction_Error', scaled_df.columns)

class TestDetectAnomalies(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(seed=12345)  

        reconstruction_error = rng.random(100)  
        self.df_resampled = pd.DataFrame({"Reconstruction_Error": reconstruction_error})

    def test_mse_threshold_calculation(self):
        sensitivity = 0.95
        df_with_anomalies = appdbrealtime.detect_anomalies(self.df_resampled, sensitivity)
        calculated_threshold = np.quantile(self.df_resampled['Reconstruction_Error'], sensitivity)
        self.assertAlmostEqual(df_with_anomalies['Predicted_Anomalies'].sum(), 
                               sum(self.df_resampled['Reconstruction_Error'] > calculated_threshold))

    def test_anomaly_detection(self):
        sensitivity = 0.95
        df_with_anomalies = appdbrealtime.detect_anomalies(self.df_resampled, sensitivity)
        self.assertIn('Predicted_Anomalies', df_with_anomalies.columns)
        self.assertTrue(all(df_with_anomalies['Predicted_Anomalies'] == 
                            (df_with_anomalies['Reconstruction_Error'] > np.quantile(df_with_anomalies['Reconstruction_Error'], sensitivity))))

    def test_output_structure(self):
        sensitivity = 0.95
        df_with_anomalies = appdbrealtime.detect_anomalies(self.df_resampled, sensitivity)
        self.assertIsInstance(df_with_anomalies, pd.DataFrame)
        self.assertIn('Predicted_Anomalies', df_with_anomalies.columns)

class TestTrackDailyAnomalies(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(seed=12345)  

        dates = pd.date_range('2024-01-01', periods=40, freq='H')

        anomalies = rng.integers(low=0, high=2, size=40) 

        self.df_resampled = pd.DataFrame({'Timestamp': dates, 'Predicted_Anomalies': anomalies})

    def test_date_extraction(self):
        daily_data = appdbrealtime.track_daily_anomalies(self.df_resampled)
        self.assertTrue(all(isinstance(date, str) for date in daily_data.keys()))
        self.assertTrue(all(datetime.strptime(date, '%Y-%m-%d') for date in daily_data.keys()))

    def test_grouping_and_summation(self):
        daily_data = appdbrealtime.track_daily_anomalies(self.df_resampled)
        for date_str, anomaly_count in daily_data.items():
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            expected_count = self.df_resampled[self.df_resampled['Timestamp'].dt.date == date]['Predicted_Anomalies'].sum()
            self.assertEqual(anomaly_count, expected_count)

    def test_thresholding_anomalies(self):
        daily_data = appdbrealtime.track_daily_anomalies(self.df_resampled)
        self.assertTrue(all(count > 10 for count in daily_data.values()))

    def test_output_structure(self):
        daily_data = appdbrealtime.track_daily_anomalies(self.df_resampled)
        self.assertIsInstance(daily_data, dict)

class TestCreatePlotlyFigure(unittest.TestCase):

    def setUp(self):
        seed = 12345
        rng = np.random.default_rng(np.random.SeedSequence(seed))

        dates = pd.date_range('2024-01-01', periods=10, freq='H')

        pressure = rng.random(10) 

        severity = rng.choice(['Mild', 'Moderate', 'Severe', None], size=10)

        self.df_resampled = pd.DataFrame({'Timestamp': dates, 'Pressure': pressure, 'Anomaly_Severity': severity})
        self.df_resampled.set_index('Timestamp', inplace=True)

    def test_presence_of_elements(self):
        fig = appdbrealtime.create_plotly_figure(self.df_resampled)
        self.assertEqual(len(fig.data), 4) 
        self.assertIsNotNone(fig.layout.title)
        self.assertEqual(fig.layout.xaxis.title.text, 'Timestamp')
        self.assertEqual(fig.layout.yaxis.title.text, 'Scaled Pressure')

    def test_trace_data_correctness(self):
        fig = appdbrealtime.create_plotly_figure(self.df_resampled)

        expected_x_values = self.df_resampled.index.astype('int64') // 10**9
        plotly_x_values = pd.to_datetime(fig.data[0].x).astype('int64') // 10**9
        np.testing.assert_array_equal(plotly_x_values, expected_x_values)

        np.testing.assert_array_equal(fig.data[0].y, self.df_resampled['Pressure'].to_numpy())


    def test_color_and_style_settings(self):
        fig = appdbrealtime.create_plotly_figure(self.df_resampled)
        expected_colors = ['yellow', 'orange', 'red']
        for i, color in enumerate(expected_colors, start=1):
            self.assertEqual(fig.data[i].marker.color, color)

    def test_date_range_in_layout(self):
        fig = appdbrealtime.create_plotly_figure(self.df_resampled)

        if not self.df_resampled.empty:
            max_date = self.df_resampled.index.max()
            one_day_ago = max_date - pd.Timedelta(days=1)

            xaxis_range = list(fig.layout.xaxis.range)
            self.assertEqual(xaxis_range, [one_day_ago, max_date])


class TestGetReductorDetails(unittest.TestCase):

    @patch('appdbrealtime.fetch_reductor_name_and_town_id')
    def test_get_reductor_details(self, mock_fetch):
        mock_fetch.return_value = ("ReductorName", "TownID")

        reductor_id = 1  
        result = appdbrealtime.get_reductor_details(reductor_id)

        mock_fetch.assert_called_once_with(reductor_id)

        self.assertEqual(result, ("ReductorName", "TownID"))

class TestCheckAndSendAlerts(unittest.TestCase):

    @patch('appdbrealtime.get_reductor_details')
    @patch('appdbrealtime.send_alert_to_node_red')
    def test_check_and_send_alerts(self, mock_send_alert, mock_get_details):
        mock_get_details.return_value = ("ReductorName", "TownID")
        
        df_resampled = pd.DataFrame({
            'Timestamp': [pd.Timestamp('2021-01-01 00:00:00'), pd.Timestamp('2021-01-01 00:05:00')],
            'Pressure': [1.0, 1.5],
            'Predicted_Anomalies': [False, True],
            'Anomaly_Severity': ['Normal', 'Moderate']
        })

        appdbrealtime.last_alert_timestamp = None

        reductor_id = 1 
        last_alert_timestamp = appdbrealtime.check_and_send_alerts(df_resampled, reductor_id)

        mock_get_details.assert_called_once_with(reductor_id)

        self.assertEqual(mock_send_alert.call_count, 1)

        self.assertEqual(last_alert_timestamp, pd.Timestamp('2021-01-01 00:05:00'))

class TestCreatePlotlyGraphFull(unittest.TestCase):

    @patch('appdbrealtime.preprocess_data')
    @patch('appdbrealtime.scale_data')
    @patch('appdbrealtime.detect_anomalies')
    @patch('appdbrealtime.check_and_send_alerts')
    @patch('appdbrealtime.create_plotly_figure')
    def test_create_plotly_graph_full(self, mock_create_figure, mock_check_alerts, mock_detect_anomalies, mock_scale_data, mock_preprocess_data):
        rng = np.random.default_rng(42)
        mock_df = pd.DataFrame({
            'Timestamp': pd.date_range(start='2021-01-01', periods=5, freq='5T'),
            'Pressure': rng.random(5),
            'Predicted_Anomalies': [False, True, False, True, False],
            'Reconstruction_Error': rng.random(5)
        })
        mock_preprocess_data.return_value = mock_df
        mock_scale_data.return_value = mock_df
        mock_detect_anomalies.return_value = mock_df
        mock_check_alerts.return_value = pd.Timestamp('2021-01-01 00:20:00')

        mock_figure = MagicMock()
        mock_create_figure.return_value = mock_figure

        fig, anomaly_count = appdbrealtime.create_plotly_graph_full(mock_df, 0.95, None, None)

        mock_preprocess_data.assert_called_once_with(mock_df)
        mock_scale_data.assert_called_once()
        mock_detect_anomalies.assert_called_once()
        mock_check_alerts.assert_called_once()
        mock_create_figure.assert_called_once_with(mock_df)
        self.assertEqual(anomaly_count, 2)
        self.assertEqual(fig, mock_figure)

                        
if __name__ == '__main__':
    unittest.main()