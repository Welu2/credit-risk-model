from pydantic import BaseModel


class PredictionRequest(BaseModel):
    num__Amount: float
    num__Value: float
    num__PricingStrategy: float

    num__TotalTransactionAmount: float
    num__AverageTransactionAmount: float
    num__TransactionCount: float
    num__StdTransactionAmount: float

    num__TransactionHour: float
    num__TransactionDay: float
    num__TransactionMonth: float
    num__TransactionYear: float

    num__ProviderId_WOE: float
    num__ProductId_WOE: float
    num__ProductCategory_WOE: float
    num__ChannelId_WOE: float
    num__CountryCode_WOE: float

    cat__CurrencyCode_UGX: int


class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int
