import unittest
import pandas as pd
import numpy as np
from src.models.model_trainer import train_models
from src.models.model_evaluator import evaluate_models

class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_train_models(self):
        models = train_models(self.sample_data)
        self.assertIn('Logistic Regression', models)
        self.assertIn('Random Forest', models)
        self.assertIn('SVM', models)

    def test_evaluate_models(self):
        models = train_models(self.sample_data)
        results = evaluate_models(models, self.sample_data)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('Model', results.columns)
        self.assertIn('Accuracy', results.columns)
        self.assertTrue(all(results['Accuracy'] >= 0) and all(results['Accuracy'] <= 1))

if __name__ == '__main__':
    unittest.main()