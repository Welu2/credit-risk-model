import os

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI

from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk API", version="1.0.0")

# Read container environment variable first, 
# then fallback to local server registry
MODEL_URI = os.getenv("MODEL_URI", "models:/CreditRiskModel/latest")
model = mlflow.sklearn.load_model(MODEL_URI)


@app.get("/")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    data = pd.DataFrame([request.model_dump()])
    probability = float(model.predict_proba(data)[0][1])
    prediction = int(probability >= 0.5)

    return PredictionResponse(
        risk_probability=probability, prediction=prediction
    )
