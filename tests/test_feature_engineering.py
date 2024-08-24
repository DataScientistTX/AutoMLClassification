import unittest
import pandas as pd
import numpy as np
from src.feature_engineering.feature_processor import preprocess_data

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'numeric_feature': [1, 2, 3, 4, 5],
            'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 1]
        })

    def test_preprocess_data(self):
        processed_data = preprocess_data(self.sample_data)
        
        # Check if categorical features are encoded
        self.assertIn('categorical_feature_A', processed_data.columns)
        self.assertIn('categorical_feature_B', processed_data.columns)
        self.assertIn('categorical_feature_C', processed_data.columns)
        
        # Check if numeric features are scaled
        self.assertTrue(np.all(processed_data['numeric_feature'] >= 0))
        self.assertTrue(np.all(processed_data['numeric_feature'] <= 1))

    def test_preprocess_data_with_missing_values(self):
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'numeric_feature'] = None
        processed_data = preprocess_data(data_with_missing)
        self.assertFalse(processed_data['numeric_feature'].isnull().any())

if __name__ == '__main__':
    unittest.main()