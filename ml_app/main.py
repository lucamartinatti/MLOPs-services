import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "./data/titanic/"


def data_loader():
    train_set = pd.read_csv(DATA_PATH + "train.csv")
    test_set = pd.read_csv(DATA_PATH + "test.csv")
    return [train_set, test_set]


def data_prepartation(dataset: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    freq_port = dataset["Embarked"].dropna().mode()[0]
    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)

    encoder = LabelEncoder()
    categoricalFeatures = dataset.select_dtypes(include=["object"]).columns

    dataset[categoricalFeatures] = dataset[categoricalFeatures].astype(str)
    encoded = dataset[categoricalFeatures].apply(encoder.fit_transform)

    for j in categoricalFeatures:
        dataset[j] = encoded[j]

    dataset = dataset.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # TODO fill NaN

    Y = dataset.pop("Survived") if "Survived" in dataset.columns else None

    return dataset, Y


def main():
    train, test = data_loader()

    X, Y = data_prepartation(train)
    X_test, Y_test = data_prepartation(test)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20)


if __name__ == "__main__":
    main()
