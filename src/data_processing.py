import os
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler



# =====================================================
# AGGREGATE FEATURES
# =====================================================

class AggregateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.customer_stats_ = (
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

        X = X.merge(
            self.customer_stats_,
            on="CustomerId",
            how="left"
        )

        cols = [
            "TotalTransactionAmount",
            "AverageTransactionAmount",
            "TransactionCount",
            "StdTransactionAmount"
        ]

        X[cols] = X[cols].fillna(0)

        return X


# =====================================================
# DATETIME FEATURES
# =====================================================

class DateTimeFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        dt = pd.to_datetime(X["TransactionStartTime"])

        X["TransactionHour"] = dt.dt.hour
        X["TransactionDay"] = dt.dt.day
        X["TransactionMonth"] = dt.dt.month
        X["TransactionYear"] = dt.dt.year

        return X


# =====================================================
# WOE TRANSFORMER
# =====================================================
class WoETransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns
        self.woe_maps_ = {}

    def fit(self, X, y):

        X = X.copy()

        for col in self.columns:

            temp = pd.DataFrame({
                "feature": X[col].astype(str),
                "target": y
            })

            grouped = temp.groupby("feature")["target"]

            good = (grouped.count() - grouped.sum()) + 0.5
            bad = grouped.sum() + 0.5

            woe = np.log(
                (good / good.sum()) /
                (bad / bad.sum())
            )

            self.woe_maps_[col] = woe.to_dict()

        return self

    def transform(self, X):

        X = X.copy()

        for col in self.columns:

            X[f"{col}_WOE"] = (
                X[col]
                .astype(str)
                .map(self.woe_maps_[col])
                .fillna(0)
            )

        return X

# =====================================================
# PIPELINE
# =====================================================

def build_pipeline():

    woe_columns = [
        "ProviderId",
        "ProductCategory",
        "ChannelId",
        "CountryCode"
    ]

    numeric_features = [
        "Amount",
        "Value",
        "PricingStrategy",

        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdTransactionAmount",

        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear",

        "ProviderId_WOE",
        "ProductCategory_WOE",
        "ChannelId_WOE",
        "CountryCode_WOE"
    ]

    categorical_features = [
        "CurrencyCode"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("aggregate_features", AggregateFeatures()),
            ("datetime_features", DateTimeFeatures()),
            ("woe_transform", WoETransformer(woe_columns)),
            ("preprocessor", preprocessor)
        ]
    )

    return pipeline


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_PATH = os.path.join(
        BASE_DIR,
        "..",
        "data",
        "raw",
        "data.csv"
    )

    MODEL_PATH = os.path.join(
        BASE_DIR,
        "..",
        "models",
        "data_pipeline.pkl"
    )

    df = pd.read_csv(DATA_PATH)

    target = "FraudResult"

    X = df.drop(columns=[target])
    y = df[target]

    pipeline = build_pipeline()

    X_processed = pipeline.fit_transform(X, y)

    print("=" * 50)
    print("TASK 3 COMPLETED")
    print("=" * 50)
    print(f"Original Shape : {X.shape}")
    print(f"Processed Shape: {X_processed.shape}")

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(f"Pipeline saved to: {MODEL_PATH}")