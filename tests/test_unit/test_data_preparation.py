import numpy as np
import pandas as pd

from ml_app.main import fill_nan_with_mode

df = pd.DataFrame(
    [("bird", 2, 2), ("mammal", 4, np.nan), ("arthropod", 8, 0), ("bird", 2, np.nan)],
    index=("falcon", "horse", "spider", "ostrich"),
    columns=("species", "legs", "wings"),
)


def test_data_leaking():
    expected_df = df.copy()

    expected_df.at["horse", "wings"] = 0.0
    expected_df.at["ostrich", "wings"] = 0.0

    transformed_df = fill_nan_with_mode(df, ["wings"])

    assert transformed_df.equals(expected_df)
