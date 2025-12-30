from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI(title="Heart Disease Prediction API")


class PatientInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict_heart_disease(data: PatientInput):
    result = predict(data.dict())
    return result
