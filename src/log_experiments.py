import mlflow
import mlflow.sklearn
import joblib
import os

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "../models/random_forest_model.pkl"
SCALER_PATH = "../models/scaler.pkl"

# ----------------------------
# Metrics from Colab (hardcode)
# ----------------------------
accuracy = 0.901639
roc_auc = 0.954545

# ----------------------------
# Load model
# ----------------------------
model = joblib.load(MODEL_PATH)

# ----------------------------
# MLflow setup
# ----------------------------
os.makedirs("../mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("Heart Disease Classification")

# ----------------------------
# Log experiment
# ----------------------------
with mlflow.start_run(run_name="Random Forest (Colab-trained)"):
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 200)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(model, artifact_path="model")

print("MLflow experiment logged successfully.")
