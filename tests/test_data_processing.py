import pandas as pd

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.data_processing import (
    DateTimeFeatures,
    create_proxy_target,AggregateFeatures 
)
def test_datetime_features():

    df = pd.DataFrame(
        {
            "TransactionStartTime": [
                "2024-01-15 10:30:00"
            ]
        }
    )

    transformer = DateTimeFeatures()

    result = transformer.transform(df)

    assert "TransactionHour" in result.columns
    assert "TransactionDay" in result.columns
    assert "TransactionMonth" in result.columns
    assert "TransactionYear" in result.columns


def test_aggregate_features():

    df = pd.DataFrame(
        {
            "CustomerId": [
                1,
                1,
                1
            ],
            "Amount": [
                100,
                200,
                300
            ]
        }
    )

    transformer = AggregateFeatures()

    transformer.fit(df)

    result = transformer.transform(df)

    assert (
        result[
            "TotalTransactionAmount"
        ].iloc[0]
        == 600
    )

    assert (
        result[
            "TransactionCount"
        ].iloc[0]
        == 3
    )