from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

class Preprocessor:
    def __init__(self, spark, config):
        self.spark = spark
        self.config = config
        self.df = None

    def load_data(self):
        """Load raw FIFA dataset."""
        self.df = self.spark.read.csv(self.config['data']['raw'], header=True, inferSchema=True)
        return self

    def clean_data(self):
        """Handle missing values and convert currency columns."""
        self.df = self.df.dropna(subset=self.config['pipeline']['features'][:-1])  # Exclude NationalityIndex
        self.df = self.df.withColumn('Value', col('Value').cast("string").replace('€', '').replace('M', 'e6').replace('K', 'e3').cast("float"))
        self.df = self.df.withColumn('Wage', col('Wage').cast("string").replace('€', '').replace('M', 'e6').replace('K', 'e3').cast("float"))
        return self

    def index_categorical(self):
        """Index categorical Nationality column and save indexer."""
        indexer = StringIndexer(inputCol="Nationality", outputCol="NationalityIndex")
        indexer_model = indexer.fit(self.df)
        self.df = indexer_model.transform(self.df)
        indexer_model.write().overwrite().save("models/nationality_indexer")
        return self

    def save_processed_data(self):
        """Save preprocessed data."""
        self.df.write.csv(self.config['data']['processed'], header=True, mode="overwrite")
        return self

    def get_data(self):
        """Return preprocessed DataFrame."""
        return self.df
