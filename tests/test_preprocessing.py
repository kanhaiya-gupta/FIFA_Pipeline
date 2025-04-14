import unittest
from pyspark.sql import SparkSession
from src.preprocessing.preprocess import Preprocessor

class TestPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("TestPreprocessor").getOrCreate()
        cls.config = {
            'data': {'raw': 'data/raw/FIFA_2018.csv'},
            'pipeline': {'features': ['Overall', 'Potential', 'Acceleration', 'Agility', 'Value', 'Wage', 'NationalityIndex']}
        }

    def test_load_data(self):
        preprocessor = Preprocessor(self.spark, self.config)
        preprocessor.load_data()
        self.assertIsNotNone(preprocessor.get_data())

    def test_clean_data(self):
        preprocessor = Preprocessor(self.spark, self.config).load_data().clean_data()
        df = preprocessor.get_data()
        self.assertFalse(df.filter(col('Value').isNull()).count() > 0)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

if __name__ == "__main__":
    unittest.main()
