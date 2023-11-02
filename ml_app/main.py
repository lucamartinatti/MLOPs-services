import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "./data/titanic/"


def data_prepartation(dataset):
    for i in range(len(dataset)):
        freq_port = dataset[i]["Embarked"].dropna().mode()[0]
        dataset[i]["Embarked"] = dataset[i]["Embarked"].fillna(freq_port)

    encoder = LabelEncoder()
    categoricalFeatures = dataset[0].select_dtypes(include=["object"]).columns
    for i, data in enumerate(dataset):
        data[categoricalFeatures] = data[categoricalFeatures].astype(str)
        encoded = data[categoricalFeatures].apply(encoder.fit_transform)
        for j in categoricalFeatures:
            dataset[i][j] = encoded[j]
    for i, data in enumerate(dataset):
        dataset[i] = dataset[i].drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # TODO fill NaN

    Y = dataset[0].pop("Survived")
    X = dataset[0]

    return X, Y, dataset[1]


if __name__ == "__main__":
    train_set = pd.read_csv(DATA_PATH + "train.csv")
    test_set = pd.read_csv(DATA_PATH + "test.csv")
    dataset = [train_set, test_set]

    X, Y, X_test = data_prepartation(dataset)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20)
    print("")
