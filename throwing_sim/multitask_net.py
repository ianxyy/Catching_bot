import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from joblib import dump

class MultiTaskNet(nn.Module):
    def __init__(self, input_size):
        super(MultiTaskNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.5)  # Apply dropout
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.5)  # Apply dropout
        # Classification branch for touch prediction
        self.classifier = nn.Linear(64, 2)  # Two output neurons for binary classification
        # Regression branch for position prediction
        self.regressor = nn.Linear(64, 2)   # Two output neurons for regression

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout
        return self.classifier(x), self.regressor(x)


if __name__ == "__main__":
    data_1 = pd.read_csv('simulation_data2.csv')
    data_2 = pd.read_csv('simulation_data3.csv')
    data = pd.concat([data_1, data_2], axis=0)
    targets = data.iloc[:, -4:-2].copy()
    # Create binary classification targets
    # 1 if there's a touch, 0 if it's -1
    targets['LeftTouch'] = (targets['AverageLeftZ'] != -1.0).astype(int)
    targets['RightTouch'] = (targets['AverageRightZ'] != -1.0).astype(int)

    
    # Replace -1 with NaN
    targets.replace(-1.0, np.nan, inplace=True)

    # Prepare the features
    X = data.iloc[:, 1:-4].values
    y_class = targets[['LeftTouch', 'RightTouch']].values
    y_reg = targets[['AverageLeftZ', 'AverageRightZ']].values

    # Normalize the input features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # After fitting scaler to training data
    dump(scaler, 'scaler.joblib')
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y_class_tensor = torch.tensor(y_class, dtype=torch.float32)
    y_reg_tensor = torch.tensor(y_reg, dtype=torch.float32)

    # Create datasets, splitting the data for classification and regression
    mask_left_touch = ~torch.isnan(y_reg_tensor[:, 0])
    mask_right_touch = ~torch.isnan(y_reg_tensor[:, 1])

    train_X, val_X, train_y_class, val_y_class, train_y_reg, val_y_reg = train_test_split(
        X_tensor, y_class_tensor, y_reg_tensor, test_size=0.2, random_state=42
    )
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_X, train_y_class, train_y_reg)
    val_dataset = TensorDataset(val_X, val_y_class, val_y_reg)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    # Initialize the model, loss functions, and optimizer
    input_size = train_X.shape[1]
    model = MultiTaskNet(input_size)
    classification_criterion = nn.BCEWithLogitsLoss()
    regression_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training and evaluation
    train_losses = []
    val_losses = []
    for epoch in range(2000):
        model.train()  # Training mode
        train_loss = 0.0
        for inputs, class_targets, reg_targets in train_loader:
            optimizer.zero_grad()
            class_outputs, reg_outputs = model(inputs)
            class_loss = classification_criterion(class_outputs, class_targets)
            # Only compute regression loss where we have non-NaN values
            is_not_nan = ~torch.isnan(reg_targets).any(dim=1)
            reg_loss = regression_criterion(reg_outputs[is_not_nan], reg_targets[is_not_nan])
            loss = class_loss + reg_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)

        model.eval()  # Evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, class_targets, reg_targets in val_loader:
                class_outputs, reg_outputs = model(inputs)
                class_loss = classification_criterion(class_outputs, class_targets)
                is_not_nan = ~torch.isnan(reg_targets).any(dim=1)
                reg_loss = regression_criterion(reg_outputs[is_not_nan], reg_targets[is_not_nan])
                loss = class_loss + reg_loss
                val_loss += loss.item()
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}')

    # Saving model state dictionary
    torch.save(model.state_dict(), '/home/ece484/Catching_bot/throwing_sim/model_state_dict.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()