import pandas as pd
import pytest


def test_baseline_pipeline_setup():
    """Verify that pytest is configured correctly and pandas executes."""
    sample_data = {"account_id": [1, 2, 3], "missed_payments": [0, 1, 0]}
    df = pd.DataFrame(sample_data)

    assert not df.empty
    assert df.shape == (3, 2)
    assert "missed_payments" in df.columns
