import unittest
from unittest.mock import patch, MagicMock
import os
import json
from appdbrealtime import app
import pandas as pd
from appdbrealtime import get_data_from_db_for_reductor, create_plotly_graph_full

class BasicTests(unittest.TestCase):

    def setUp(self):
        app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
        self.app = app.test_client()
        self.app.testing = True

    # Test if the index page loads correctly
    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Reductor Data Analysis', response.data)  # Check for page content

    @patch('mysql.connector.connect')
    def test_get_reductors(self, mock_db_connect):
        # Mock the cursor's fetchall to return mock data
        mock_reductors_data = [{'reductorID': 1, 'name': 'Reductor1'}, {'reductorID': 2, 'name': 'Reductor2'}]
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = mock_reductors_data

        # Set the return_value of the mock connection's cursor method to our mock_cursor
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_db_connect.return_value.__enter__.return_value = mock_connection

        # Make a GET request to the /get_reductors endpoint
        response = self.app.get('/get_reductors')

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Load the response data as JSON
        data = json.loads(response.data)

        # Check if the response is a list and matches the mock data
        self.assertIsInstance(data, list)
        self.assertEqual(data, mock_reductors_data)
        
    def test_get_data_from_db_for_reductor(self):
        with patch('appdb.get_data_from_db_for_reductor') as mock_get_data:
            mock_get_data.return_value = pd.DataFrame({
                'date': ['2021-01-01', '2021-01-02'],
                'value': [123, 456]
            })

            # Known test data
            start_date = '2021-01-01'
            end_date = '2021-01-31'
            reductor_id = 4  

            # Import the function inside the context manager
            from appdbrealtime import get_data_from_db_for_reductor

            # Call the function
            result_df = get_data_from_db_for_reductor(start_date, end_date, reductor_id)

            # Check if the result is a DataFrame
            self.assertIsInstance(result_df, pd.DataFrame)

            # Check if DataFrame has 2 rows as mocked
            self.assertEqual(len(result_df), 2)
            self.assertTrue('date' in result_df.columns and 'value' in result_df.columns)
        
    @patch('appdb.get_data_from_db_for_reductor')  # Replace with the actual import path
    @patch('appdb.create_plotly_graph')  # Replace with the actual import path
    def test_load_data_valid_input(self, mock_create_plotly_graph, mock_get_data_from_db):
                # Create a mocked DataFrame
        mocked_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2021-01-01', periods=5, freq='D'),
            'flow': [100, 110, 105, 115, 120],
            'pressure': [10.5, 10.6, 10.4, 10.8, 10.7]
        })
        mock_get_data_from_db.return_value = mocked_df  # Provide a mocked DataFrame
        mock_create_plotly_graph.return_value = 'mock_plot_url'

        # Simulate valid form data
        response = self.app.post('/load_data', data={
            'reductor_id': '4',
            'detailed_month': '2021-01'
        }, follow_redirects=True)

        # Check the response
        self.assertEqual(response.status_code, 200)
        # Add more assertions here based on expected behavior

    def test_load_data_invalid_input(self):
        # Simulate invalid form data
        response = self.app.post('/load_data', data={}, follow_redirects=True)

        # Check the response for invalid input
        self.assertEqual(response.status_code, 400)
        self.assertIn("Please select either a single month or a start and end month", response.data.decode())
        
    @patch('appdb.pyo.plot', MagicMock())
    def test_create_plotly_graph(self):
        # Create a sample DataFrame for testing
        sample_df = pd.DataFrame({
            'Timestamp': pd.date_range(start='2021-01-01', periods=12, freq='H'),
            'Flow': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155],
            'Pressure': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2]
        })

        # Call the function with the sample DataFrame
        filename = create_plotly_graph(sample_df, '2021-01-01', '2021-01-02')

        # Check if the returned filename is as expected
        self.assertIn('plot_', filename)
        self.assertIn('.html', filename)
        
if __name__ == "__main__":
    unittest.main()
