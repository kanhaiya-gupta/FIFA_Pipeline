import json

class Evaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, model, df):
        """Compute WSSSE and save metrics."""
        transformed_df = model.transform(df)
        kmeans_stage = model.stages[-1]  # KMeans is the last stage
        wssse = kmeans_stage.computeCost(transformed_df)
        
        metrics = {"WSSSE": wssse}
        with open(self.config['output']['metrics_path'], 'w') as f:
            json.dump(metrics, f, indent=4)
        
        transformed_df.select("Name", "prediction").write.csv(
            self.config['output']['predictions_path'], header=True, mode="overwrite"
        )
        return transformed_df
