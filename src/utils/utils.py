import yaml
from pyspark.sql import SparkSession

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_spark_session(app_name="FIFA_Pipeline"):
    """Initialize Spark session."""
    return SparkSession.builder.appName(app_name).getOrCreate()
