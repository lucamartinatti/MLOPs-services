import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "./data/titanic/"

data_config = {
    "drop": ["PassengerId", "Name", "Ticket", "Cabin"],
    "fill_nan": ["Embarked"],
    "label": "Survived",
}


def data_loader() -> list[pd.DataFrame]:
    train_set = pd.read_csv(DATA_PATH + "train.csv")
    test_set = pd.read_csv(DATA_PATH + "test.csv")
    return [train_set, test_set]


def fill_nan_with_mode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in columns:
        freq_port = df_copy[col].dropna().mode()[0]
        df_copy[col] = df_copy[col].fillna(freq_port)
    return df_copy


def categorical_features_encoder(df: pd.DataFrame) -> pd.DataFrame:
    encoder = LabelEncoder()
    categoricalFeatures = df.select_dtypes(include=["object"]).columns

    df[categoricalFeatures] = df[categoricalFeatures].astype(str)
    encoded = df[categoricalFeatures].apply(encoder.fit_transform)

    for j in categoricalFeatures:
        df[j] = encoded[j]

    return df


def data_prepartation(x: pd.DataFrame, config: dict) -> (pd.DataFrame, pd.DataFrame):
    x = fill_nan_with_mode(x, config["fill_nan"])

    x = categorical_features_encoder(x)

    x = x.drop(config["drop"], axis=1)

    y = x.pop(config["label"]) if config["label"] in x.columns else None

    return x, y


def main():
    train, test = data_loader()

    X, Y = data_prepartation(train, data_config)

    X_test, Y_test = data_prepartation(test, data_config)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20)


if __name__ == "__main__":
    main()
