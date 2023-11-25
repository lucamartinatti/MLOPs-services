import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import mlflow.pytorch
from mlflow import MlflowClient


class TrainConfig:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs


class BasicNetwork(nn.Module):
    def __init__(self, config: TrainConfig):
        super(BasicNetwork, self).__init__()
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.output_size)

        self.accuracy = Accuracy("multiclass", num_classes=2)
        self.config = config

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

    def train_epoch(self, train_loader):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # acc = self.accuracy(inputs, labels)  # use validation data instead
        # Log performances
        # self.log("train_loss", loss, on_epoch=True)
        # self.log("acc", acc, on_epoch=True)
        return loss.item()


def trainer(network: BasicNetwork, train_loader, config: TrainConfig):
    with mlflow.start_run():
        running_loss = 0.0
        for epoch in range(config.epochs):
            loss = network.train_epoch(train_loader)
            running_loss += loss
            mlflow.log_metric(key="loss", value=loss, step=epoch)
            print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {running_loss}")
