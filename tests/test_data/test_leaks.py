import numpy as np
import pandas as pd

from ml_app.main import data_config

df = pd.DataFrame(
    [("bird", 2, 2), ("mammal", 4, np.nan), ("arthropod", 8, 0), ("bird", 2, np.nan)],
    index=("falcon", "horse", "spider", "ostrich"),
    columns=("species", "legs", "wings"),
)


def test_data_leaking():
    print(df)
