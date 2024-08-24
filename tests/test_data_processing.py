import unittest
import pandas as pd
from src.data_processing.data_loader import load_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # This method will be called before each test
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'C', 'D', 'E'],
            'target': [0, 1, 0, 1, 1]
        })

    def test_load_data(self):
        # Test if data is loaded correctly
        loaded_data = load_data(self.sample_data)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 5)
        self.assertListEqual(list(loaded_data.columns), ['feature1', 'feature2', 'target'])

    def test_load_data_with_missing_values(self):
        # Test handling of missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'feature1'] = None
        loaded_data = load_data(data_with_missing)
        self.assertFalse(loaded_data['feature1'].isnull().any())

if __name__ == '__main__':
    unittest.main()