import unittest
import os
from appdb import app  # Adjust this import to match the actual file name where your Flask app is defined

class BasicTests(unittest.TestCase):

    def setUp(self):
        app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        response = self.app.get('/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
