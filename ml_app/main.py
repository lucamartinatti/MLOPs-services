import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from basic_net import TrainConfig, BasicNetwork, trainer

DATA_PATH = "./data/titanic/"

data_config = {
    "drop": ["PassengerId", "Name", "Ticket", "Cabin"],
    "fill_nan": ["Embarked"],
    "label": "Survived",
}

# TODO config.yaml file for data config and train config

input_size = 7
hidden_size = 128
output_size = 1
learning_rate = 0.001
epochs = 10


class TitanicDataset(Dataset):
    def __init__(self, dataframe, config):
        self.dataframe = dataframe
        self.config = config

    def __getitem__(self, index):
        features = self.dataframe.iloc[index].copy()
        label = (
            pd.Series([features.pop(self.config["label"])]).to_numpy()
            if self.config["label"] in features.index.to_list()
            else None
        )
        return features.to_numpy(), label

    def __len__(self):
        return len(self.dataframe)


def data_loader() -> list[pd.DataFrame]:
    train_set = pd.read_csv(DATA_PATH + "train.csv")
    test_set = pd.read_csv(DATA_PATH + "test.csv")
    return [train_set, test_set]


def fill_nan_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in df_copy.columns:
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


def dataset_creator(x: pd.DataFrame, config: dict) -> TitanicDataset:
    x = categorical_features_encoder(x)

    x = fill_nan_with_mode(x)

    x = x.drop(config["drop"], axis=1)

    x = x.astype("float32")

    train_dataset = TitanicDataset(x, config)

    return train_dataset


def train_function():
    config = TrainConfig(input_size, hidden_size, output_size, learning_rate, epochs)

    train, test = data_loader()

    train_dataset = dataset_creator(train, data_config)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)

    test_dataset = dataset_creator(test, data_config)

    network = BasicNetwork(config)

    trainer(network, train_loader, config)


def main():
    train_function()


if __name__ == "__main__":
    main()
