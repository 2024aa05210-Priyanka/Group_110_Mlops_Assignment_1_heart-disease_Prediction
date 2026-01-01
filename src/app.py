import time
import logging

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest

from src.predict import predict


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------- Prometheus Metrics ----------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint"],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Request latency",
    ["endpoint"],
)


# ---------------- FastAPI App ----------------
app = FastAPI(title="Heart Disease Prediction API")


# ---------------- Request Monitoring Middleware ----------------
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time
    endpoint = request.url.path

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
    ).inc()

    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    logger.info(
        f"{request.method} {endpoint} "
        f"Status={response.status_code} "
        f"Time={latency:.4f}s"
    )

    return response


# ---------------- Data Model ----------------
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


# ---------------- Routes ----------------
@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict_heart_disease(data: PatientInput):
    result = predict(data.dict())

    logger.info(
        f"Prediction={result['prediction']}, "
        f"Probability={result['probability']}"
    )

    return result


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain",
    )
