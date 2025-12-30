import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "heart_cleaned.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")



def train():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Ensure binary target (safety for reproducibility)
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    numerical_features = [
        "age", "trestbps", "chol", "thalach", "oldpeak", "ca"
    ]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_features] = scaler.fit_transform(
        X_train[numerical_features]
    )
    X_test_scaled[numerical_features] = scaler.transform(
        X_test[numerical_features]
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, y_prob_lr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, y_prob_rf)

    # Select best model
    best_model = rf if rf_auc >= lr_auc else lr

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/random_forest_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print("Training completed")
    print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
    print(f"Random Forest ROC-AUC: {rf_auc:.4f}")


if __name__ == "__main__":
    train()
