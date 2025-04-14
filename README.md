# PySpark FIFA Player Clustering Pipeline with FastAPI

A modular PySpark machine learning pipeline for clustering FIFA 2018 player data using KMeans, with a FastAPI server to deploy the model for real-time predictions.

## Overview

This project implements an end-to-end machine learning pipeline to cluster FIFA 2018 players based on attributes like `Overall`, `Potential`, `Acceleration`, and `Value`. Built with PySpark for scalable data processing and model training, it includes data preprocessing, feature engineering, KMeans clustering, evaluation, and visualization. A FastAPI server provides endpoints to predict player clusters, making the model accessible for real-time use.

Key features:
- **Preprocessing**: Cleans currency fields (`Value`, `Wage`) and indexes categorical columns (`Nationality`).
- **Clustering**: Uses KMeans to group players into clusters (e.g., elite, average).
- **Deployment**: FastAPI server with `/predict` endpoint for cluster predictions.
- **Reproducibility**: DVC for data and pipeline tracking.
- **Testing**: Unit tests for preprocessing and model components.

## Project Structure

```
FIFA_Pipeline/
├── config/                     # Configuration files
│   ├── config.yaml            # Dataset paths and settings
│   └── hp_config.json         # KMeans hyperparameter settings
├── data/                       # Datasets
│   ├── raw/                   # FIFA_2018.csv
│   └── processed/             # fifa_processed.csv
├── docs/                       # Documentation
│   ├── DVC_workflow_README.md # DVC pipeline guide
│   └── FastAPI_README.md      # API usage guide
├── models/                     # Trained models
│   ├── kmeans_model           # KMeans model
│   └── nationality_indexer    # StringIndexer for Nationality
├── notebooks/                  # Exploration notebooks
│   └── explore_fifa.ipynb
├── reports/                    # Outputs
│   ├── cluster_distribution.png
│   ├── metrics.json
│   └── predictions.csv
├── scripts/                    # Automation scripts
│   ├── hp_tuning.py
│   └── metrics_and_plots.py
├── src/                        # Source code
│   ├── api/                   # FastAPI routes
│   │   └── app.py
│   ├── evaluation/            # Model evaluation
│   │   └── evaluate.py
│   ├── models/                # Model definitions
│   │   └── model.py
│   ├── preprocessing/         # Data preprocessing
│   │   └── preprocess.py
│   ├── training/              # Model training
│   │   └── train.py
│   └── utils/                 # Utilities
│       └── utils.py
├── tests/                      # Unit tests
│   ├── test_model.py
│   └── test_preprocessing.py
├── .env                       # Environment variables
├── main.py                    # FastAPI server entry point
├── requirements.txt           # Dependencies
├── setup_pipeline.sh          # Setup script
├── dvc.yaml                   # DVC pipeline
└── README.md                  # This file
```

## Prerequisites

- Python 3.8+
- PySpark 3.5.0
- Java 8 or 11 (for PySpark)
- Git
- (Optional) DVC for data version control
- (Optional) Docker for containerized deployment

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fifa-pipeline.git
   cd fifa-pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**:
   - Copy `.env.example` to `.env` and configure if needed (e.g., Spark settings).
   - Run the setup script:
     ```bash
     bash setup_pipeline.sh
     ```

4. **Prepare data**:
   - Place `FIFA_2018.csv` in `data/raw/`. Example format:
     ```
     Name,Age,Overall,Potential,Value,Nationality,...
     Cristiano Ronaldo,32,94,94,€95.5M,Portugal,...
     ```

## Usage

### Train the Model

Run the pipeline to preprocess data, train the KMeans model, and generate outputs:

```bash
python src/preprocessing/preprocess.py
python src/training/train.py
python src/evaluation/evaluate.py
python scripts/metrics_and_plots.py
```

Or use DVC to run all stages:

```bash
dvc repro
```

Outputs:
- `models/kmeans_model`: Trained KMeans model.
- `models/nationality_indexer`: Nationality indexer.
- `reports/metrics.json`: WSSSE metric.
- `reports/predictions.csv`: Clustered players.
- `reports/cluster_distribution.png`: Cluster distribution plot.

### Run the FastAPI Server

Start the FastAPI server to serve predictions:

```bash
python main.py
```

The server runs at `http://localhost:8000`. Access the Swagger UI at `http://localhost:8000/docs` for interactive testing.

### API Endpoints

- **GET /health**:
  - Checks server status.
  - Response: `{"status": "healthy"}`

- **POST /predict**:
  - Predicts cluster for a player.
  - Request body:
    ```json
    {
      "Overall": 94.0,
      "Potential": 94.0,
      "Acceleration": 89.0,
      "Agility": 89.0,
      "Value": 95500000.0,
      "Wage": 565000.0,
      "Nationality": "Portugal"
    }
    ```
  - Response: `{"cluster": 0}`

Example - Example:
    ```bash
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"Overall": 94.0, "Potential": 94.0, "Acceleration": 89.0, "Agility": 89.0, "Value": 95500000.0, "Wage": 565000.0, "Nationality": "Portugal"}'
    ```

### Run Tests

Run unit tests for preprocessing and model components:

```bash
python -m unittest discover tests
```

## DVC Workflow

Track data, models, and pipeline stages with DVC:

```bash
dvc init
dvc add data/raw/FIFA_2018.csv
dvc repro
```

See `docs/DVC_workflow_README.md` for details.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [PySpark](https://spark.apache.org/docs/latest/api/python/) for scalable ML.
- [FastAPI](https://fastapi.tiangolo.com/) for API deployment.
- [DVC](https://dvc.org/) for data version control.
```

---

### Explanation

#### GitHub Project Title
- **"PySpark FIFA Player Clustering Pipeline with FastAPI"**:
  - **PySpark**: Highlights the core technology for data processing and ML.
  - **FIFA Player Clustering**: Specifies the domain and task.
  - **Pipeline**: Indicates a structured, end-to-end workflow.
  - **FastAPI**: Emphasizes the deployment mechanism.
  - Short and descriptive, suitable for GitHub visibility.

#### README.md
- **Overview**: Summarizes the project’s purpose, technologies, and features.
- **Structure**: Lists key directories for transparency.
- **Prerequisites**: Ensures users have the required tools.
- **Setup**: Guides users through cloning, installing dependencies, and preparing data.
- **Usage**: Provides clear instructions for training, running the API, testing, and using DVC.
- **API Section**: Details endpoints with examples, making it easy to interact with the server.
- **Contributing**: Encourages community contributions with a clear process.
- **License**: Specifies MIT for openness.
- **Acknowledgments**: Credits key libraries for transparency.
