## Heart Disease Prediction – End-to-End MLOps Pipeline

This project implements a complete end-to-end MLOps pipeline for predicting the presence of heart disease using machine learning.
It covers data processing, model training, experiment tracking, testing, CI/CD, and deployment as a REST API.

## Project Overview

The goal of this project is to:
- Train machine learning models to predict heart disease
- Track experiments using MLflow
- Package training and inference logic into reusable scripts
- Add automated testing and CI
- Serve the trained model via a REST API using FastAPI
- Enable reproducible and deployable ML workflows

## Dataset
**Source:** UCI Machine Learning Repository – Heart Disease Dataset
Access Method: Programmatic download using ucimlrepo

**Target Variable:**
Binary classification

0 → No heart disease

1 → Presence of heart disease

**Preprocessing**
- Missing values handled using median/mode imputation
- Target variable converted from multi-class to binary

Cleaned dataset saved as:

data/processed/heart_cleaned.csv

## Project Structure
heart-disease-mlops/
│
├── data/
│   └── processed/
│       └── heart_cleaned.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   ├── log_experiments.py
│   └── app.py
│
├── tests/
│   ├── test_training.py
│   └── test_prediction.py
│
├── mlruns/
├── pytest.ini
├── requirements.txt
└── README.md

## Models Used
**1. Logistic Regression**
- Used as a baseline model
- Trained on scaled numerical features

**2. Random Forest (Final Model)**
- Handles non-linear relationships
- No feature scaling required

Selected based on superior ROC-AUC performance

## Final Metrics (Test Set)
Model	ROC-AUC
Logistic Regression	~0.95
Random Forest	~0.95+
Experiment Tracking (MLflow)

**MLflow used to log:**
- Model parameters
- Evaluation metrics
- Trained model artifacts

## MLflow UI run locally to visualize experiments
Run API:
uvicorn src.app:app --reload

Start MLflow UI
python -m mlflow ui

Access at:

http://127.0.0.1:5000

## Reproducible Training

The entire training process is packaged into a script.

Run Training
python src/train.py

This will:

- Load processed data
- Train models
- Select the best model
- Save model artifacts to models/

## Model Inference

Inference logic is encapsulated in predict.py.

Example Usage
from src.predict import predict

sample_input = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

result = predict(sample_input)
print(result)

## Testing

Unit tests validate:

Model artifacts existence

Prediction output format and probability range

Run Tests
pytest

All tests must pass before deployment.

## CI/CD Pipeline

Implemented using GitHub Actions

Automatically:
- Installs dependencies
- Runs training script
- Executes unit tests
- Ensures reliability on every code push

## Model Serving (FastAPI)

The trained model is served as a REST API.

Start API Locally
uvicorn src.app:app --reload

## Access:

Health check: http://127.0.0.1:8000/

Swagger UI: http://127.0.0.1:8000/docs

## Prediction Endpoint

Method: POST

Endpoint: /predict

Input: JSON payload

Output: Prediction + probability

## Docker (Optional Deployment)

The application can be containerized using Docker for consistent deployment across environments.

docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api

## Key MLOps Concepts Demonstrated

- Reproducible data preprocessing
- Script-based training pipelines
- Experiment tracking
- Automated testing
- CI/CD integration
- REST API deployment
- Environment-independent execution

## Conclusion

This project demonstrates a full MLOps lifecycle, from data ingestion to deployment, following industry best practices.
It is designed to be modular, reproducible, testable, and deployable.
