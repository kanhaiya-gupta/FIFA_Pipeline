import json
from src.models.model import KMeansModel
from src.training.train import Trainer
from src.evaluation.evaluate import Evaluator

def tune_k_values(df, config, hp_config):
    """Tune KMeans k parameter."""
    results = []
    k_values = hp_config['kmeans']['k_values']
    
    for k in k_values:
        model = KMeansModel(config).build_pipeline(k=k)
        trainer = Trainer(model, config).train(df)
        transformed_df = trainer.get_trained_model().transform(df)
        kmeans_stage = trainer.get_trained_model().stages[-1]
        wssse = kmeans_stage.computeCost(transformed_df)
        results.append({"k": k, "WSSSE": wssse})
    
    with open("reports/hp_tuning_results.json", 'w') as f:
        json.dump(results, f, indent=4)
