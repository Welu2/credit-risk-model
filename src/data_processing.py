import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =====================================================
# TASK 4 - RFM TARGET ENGINEERING
# =====================================================


def create_proxy_target(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (snapshot_date - x.max()).days,
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby("Cluster")[
        ["Recency", "Frequency", "Monetary"]
    ].mean()

    print("\nCluster Profiles")
    print(cluster_summary)

    # Least engaged cluster = high risk
    high_risk_cluster = cluster_summary["Recency"].idxmax()

    print(f"\nHigh Risk Cluster: " f"{high_risk_cluster}")

    rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

    df = df.merge(
        rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left"
    )

    print("\nTarget Distribution")
    print(df["is_high_risk"].value_counts())

    return df


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
                StdTransactionAmount="std",
            )
            .fillna(0)
        )

        return self

    def transform(self, X):

        X = X.copy()

        X = X.merge(self.customer_stats_, on="CustomerId", how="left")

        cols = [
            "TotalTransactionAmount",
            "AverageTransactionAmount",
            "TransactionCount",
            "StdTransactionAmount",
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

            temp = pd.DataFrame({"feature": X[col].astype(str), "target": y})

            grouped = temp.groupby("feature", observed=True)["target"]

            good = (grouped.count() - grouped.sum()) + 0.5

            bad = (grouped.sum()) + 0.5

            woe = np.log((good / good.sum()) / (bad / bad.sum()))

            self.woe_maps_[col] = woe.to_dict()

        return self

    def transform(self, X):

        X = X.copy()

        for col in self.columns:

            X[f"{col}_WOE"] = (
                X[col].astype(str).map(self.woe_maps_[col]).fillna(0)
            )

        return X


# =====================================================
# PIPELINE
# =====================================================


def build_pipeline():

    woe_columns = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "CountryCode",
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
        "ProductId_WOE",
        "ProductCategory_WOE",
        "ChannelId_WOE",
        "CountryCode_WOE",
    ]

    categorical_features = ["CurrencyCode"]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(
        [
            ("aggregate_features", AggregateFeatures()),
            ("datetime_features", DateTimeFeatures()),
            ("woe_transform", WoETransformer(woe_columns)),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "data.csv")

    MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "data_pipeline.pkl")

    PROCESSED_DATA_PATH = os.path.join(
        BASE_DIR, "..", "data", "processed", "processed_data.csv"
    )

    df = pd.read_csv(DATA_PATH)

    # TASK 4
    df = create_proxy_target(df)

    target = "is_high_risk"

    X = df.drop(columns=[target, "FraudResult"])

    y = df[target]

    pipeline = build_pipeline()

    X_processed = pipeline.fit_transform(X, y)

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    X_processed[target] = y.values

    print("\nFinal Shape")
    print(X_processed.shape)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    joblib.dump(
        {"pipeline": pipeline, "feature_names": feature_names}, MODEL_PATH
    )

    X_processed.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nSaved pipeline to: " f"{MODEL_PATH}")

    print(f"Saved processed dataset to: " f"{PROCESSED_DATA_PATH}")
