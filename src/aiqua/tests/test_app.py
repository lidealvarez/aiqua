import unittest
from unittest.mock import patch, MagicMock
import os
import json
from appdb import app
import pandas as pd
from appdb import get_data_from_db_for_reductor, create_plotly_graph

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

    def test_get_reductors(self):
        # Make a GET request to the /get_reductors endpoint
        response = self.app.get('/get_reductors')
        
        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
        
        # Load the response data as JSON
        data = json.loads(response.data)
        
        # Check if the response is a list (as expected for reductors)
        self.assertIsInstance(data, list)
        
    def test_get_data_from_db_for_reductor(self):
        # Known test data
        start_date = '2021-01-01'
        end_date = '2021-01-31'
        reductor_id = 4  

        # Call the function
        result_df = get_data_from_db_for_reductor(start_date, end_date, reductor_id)
        
        # Check if the result is a DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)
        
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
