import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse
)

app = FastAPI(
    title="Credit Risk API",
    version="1.0.0"
)

MODEL_NAME = "CreditRiskModel"

model = mlflow.sklearn.load_model(
    "models:/CreditRiskModel/latest"
)

@app.get("/")
def health():
    return {"status": "healthy"}

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    data = pd.DataFrame(
        [request.model_dump()]
    )

    probability = float(
        model.predict_proba(data)[0][1]
    )

    prediction = int(
        probability >= 0.5
    )

    return PredictionResponse(
        risk_probability=probability,
        prediction=prediction
    )