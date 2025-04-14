from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from src.utils.utils import load_config
from pyspark.sql.functions import lit
import pandas as pd

def create_app():
    """Create and configure FastAPI app."""
    app = FastAPI(title="FIFA Player Clustering API", version="1.0.0")
    config = load_config("config/config.yaml")
    spark = SparkSession.builder.appName("FIFA_API").getOrCreate()
    model = PipelineModel.load(config['output']['model_path'])

    class PlayerInput(BaseModel):
        Overall: float
        Potential: float
        Acceleration: float
        Agility: float
        Value: float
        Wage: float
        Nationality: str

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/predict")
    async def predict_cluster(player: PlayerInput):
        """Predict cluster for a player."""
        try:
            # Convert input to DataFrame
            data = {
                "Overall": player.Overall,
                "Potential": player.Potential,
                "Acceleration": player.Acceleration,
                "Agility": player.Agility,
                "Value": player.Value,
                "Wage": player.Wage,
                "Nationality": player.Nationality
            }
            pandas_df = pd.DataFrame([data])
            input_df = spark.createDataFrame(pandas_df)

            # Add NationalityIndex (mimicking preprocessing)
            from pyspark.ml.feature import StringIndexerModel
            indexer_path = "models/nationality_indexer"
            try:
                indexer = StringIndexerModel.load(indexer_path)
            except:
                # Fallback: fit indexer on the fly (for demo purposes)
                indexer = StringIndexer(inputCol="Nationality", outputCol="NationalityIndex").fit(input_df)
                indexer.write().overwrite().save(indexer_path)
            input_df = indexer.transform(input_df)

            # Predict cluster
            prediction = model.transform(input_df).select("prediction").collect()[0]["prediction"]
            return {"cluster": int(prediction)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    @app.on_event("shutdown")
    def shutdown_event():
        """Stop Spark session on shutdown."""
        spark.stop()

    return app
