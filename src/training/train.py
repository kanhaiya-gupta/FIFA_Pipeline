from pyspark.ml import PipelineModel
import os

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.trained_model = None

    def train(self, df):
        """Train the pipeline model."""
        pipeline = self.model.get_pipeline()
        self.trained_model = pipeline.fit(df)
        return self

    def save_model(self):
        """Save the trained model."""
        if self.trained_model is None:
            raise ValueError("Model not trained. Call train() first.")
        model_path = self.config['output']['model_path']
        self.trained_model.write.overwrite().save(model_path)
        return self

    def get_trained_model(self):
        """Return the trained model."""
        return self.trained_model
