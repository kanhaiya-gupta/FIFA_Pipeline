import unittest
from src.models.model import KMeansModel

class TestKMeansModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            'pipeline': {
                'features': ['Overall', 'Potential'],
                'kmeans': {'k': 3, 'seed': 1}
            }
        }

    def test_build_pipeline(self):
        model = KMeansModel(self.config).build_pipeline()
        pipeline = model.get_pipeline()
        self.assertEqual(len(pipeline.getStages()), 3)  # Assembler, Scaler, KMeans

if __name__ == "__main__":
    unittest.main()
