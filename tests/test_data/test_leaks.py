import numpy as np
import pandas as pd

from ml_app.main import data_loader


def test_data_leaking():
    train, test = data_loader()

    train_col_diff = list(set(train.columns) - set(test.columns))
    test_col_diff = list(set(test.columns) - set(train.columns))

    train = train.drop(train_col_diff, axis=1)
    test = test.drop(test_col_diff, axis=1)

    common = pd.merge(train, test, how="inner", on=["Name", "Ticket"])

    assert common.shape[0] == 0
