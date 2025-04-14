from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

class KMeansModel:
    def __init__(self, config):
        self.config = config
        self.pipeline = None

    def build_pipeline(self, k=None):
        """Build feature engineering and KMeans pipeline."""
        k = k or self.config['pipeline']['kmeans']['k']
        assembler = VectorAssembler(inputCols=self.config['pipeline']['features'], outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
        kmeans = KMeans(featuresCol="scaledFeatures", k=k, seed=self.config['pipeline']['kmeans']['seed'])
        self.pipeline = Pipeline(stages=[assembler, scaler, kmeans])
        return self

    def get_pipeline(self):
        """Return the pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        return self.pipeline
