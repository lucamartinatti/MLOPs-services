import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


def trainer(network: BasicNetwork, train_loader, config: TrainConfig):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {running_loss}")
