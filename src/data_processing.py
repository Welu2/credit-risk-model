import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)
# Ensure you run: pip install xverse
from xverse.transformer import WOE

# =====================================================
# CUSTOM TRANSFORMERS
# =====================================================

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Safely builds customer-level profiles during training (.fit)
    to prevent data leakage and support single-row inference.
    """
    def __init__(self):
        self.stats_ = None

    def fit(self, X, y=None):
        # Learn history exclusively from training slice
        self.stats_ = (
            X.groupby("CustomerId")["Amount"]
            .agg(
                TotalTransactionAmount="sum",
                AverageTransactionAmount="mean",
                TransactionCount="count",
                StdTransactionAmount="std"
            )
            .fillna(0)
            .reset_index()
        )
        return self

    def transform(self, X):
        X = X.copy()
        X = X.merge(self.stats_, on="CustomerId", how="left")
        
        # Protect against new customers during deployment
        fill_cols = ["TotalTransactionAmount", "AverageTransactionAmount", "TransactionCount", "StdTransactionAmount"]
        X[fill_cols] = X[fill_cols].fillna(0)
        return X


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract date and time features safely from TransactionStartTime.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        times = pd.to_datetime(X["TransactionStartTime"])
        
        X["TransactionHour"] = times.dt.hour
        X["TransactionDay"] = times.dt.day
        X["TransactionMonth"] = times.dt.month
        X["TransactionYear"] = times.dt.year
        X["DayOfWeek"] = times.dt.dayofweek
        X["IsWeekend"] = (X["DayOfWeek"] >= 5).astype(int)
        return X


class SklearnWoeTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn wrapper for xverse WoE implementation 
    targeting high-cardinality/important categorical bins.
    """
    def __init__(self, columns):
        self.columns = columns
        self.woe = WOE()

    def fit(self, X, y):
        if y is None:
            raise ValueError("WoE transformation requires a target array 'y'.")
        # Keep only specified target columns for computation
        X_target = X[self.columns].copy()
        self.woe.fit(X_target, y)
        return self

    def transform(self, X):
        X = X.copy()
        X_trans = self.woe.transform(X[self.columns])
        # Merge converted columns back over originals
        for col in self.columns:
            X[col] = X_trans[col]
        return X

# =====================================================
# BUILD PIPELINE
# =====================================================

def build_pipeline(woe_columns):
    numerical_features = [
        "Amount", "Value", "PricingStrategy",
        "TotalTransactionAmount", "AverageTransactionAmount",
        "TransactionCount", "StdTransactionAmount",
        "TransactionHour", "TransactionDay", "TransactionMonth",
        "TransactionYear", "DayOfWeek", "IsWeekend"
    ]

    # Split categoricals based on whether they go to WoE or One-Hot
    categorical_ohe = [
        "ProductCategory", "ChannelId"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(
            lambda x: np.sign(x) * np.log1p(np.abs(x)),
            validate=False
        )),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_ohe)
        ],
        remainder="drop" # Automatically eliminates raw identifiers not needed by final model
    )

    pipeline = Pipeline([
        ("aggregate_features", AggregateFeatures()),
        ("datetime_features", DateTimeFeatures()),
        # Transforms raw IDs into stable information metrics using the binary target matrix
        ("woe_transform", SklearnWoeTransformer(columns=woe_columns)),
        ("preprocessor", preprocessor)
    ])

    return pipeline

# =====================================================
# MAIN ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    DATA_PATH = "../data/raw/data.csv"
    df = pd.read_csv(DATA_PATH)

    target = "FraudResult"
    X = df.drop(columns=[target])
    y = df[target]

    # Select high-cardinality categorical metrics ideal for Weight of Evidence transformation
    woe_target_columns = ["ProviderId", "ProductId", "CustomerId", "AccountId", "SubscriptionId"]

    pipeline = build_pipeline(woe_columns=woe_target_columns)
    
    # WoE requires target values 'y' inside fit_transform block 
    X_processed = pipeline.fit_transform(X, y)

    print("=" * 50)
    print("Task 3 Completed Successfully")
    print("=" * 50)
    print(f"Original Row/Col Dimensions: {X.shape}")
    print(f"Model Ready Output Shape Matrix: {X_processed.shape}")

    joblib.dump(pipeline, "src/data_processing.py")
    print("Robust pipeline saved successfully inside src/ directory.")
