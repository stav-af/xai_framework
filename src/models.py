import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score


class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


def train_model(model, X_train, y_train, X_val, y_val, epochs=1000, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        val_inputs = torch.tensor(X_val, dtype=torch.float32)
        val_outputs = model(val_inputs)
        val_preds = (val_outputs.detach().numpy() > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_preds)
        
        if epoch % 40 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}')
    
    return model


def cross_val_train(model, X, y, n_splits=5, epochs=1000, lr=0.001):
    skf = StratifiedKFold(n_splits=n_splits)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr)
        
        val_inputs = torch.tensor(X_val, dtype=torch.float32)
        val_outputs = model(val_inputs)
        val_preds = (val_outputs.detach().numpy() > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_preds)
        fold_accuracies.append(val_acc)
    
    print(f"Average accuracy over {n_splits} folds: {np.mean(fold_accuracies):.4f}")
    return model

# Example usage (assuming input data X, labels y):
# X is an np.array of shape (n_samples, n_features)
# y is an np.array of shape (n_samples,)
# Replace this with actual Boolean data encoded as 1 for True and -1 for False

# X, y = load_your_data()

# input_size = X.shape[1]  # Assuming you have X and y loaded with proper shapes

# #model = cross_validate(X, y, input_size, n_splits=5, epochs=10000, lr=0.001)
# #torch.save(model.state_dict(), 'model.pth'# print(model.forward(torch.tensor([1,1,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)))

# 

# first_lines = X[3:4]
# print(first_lines)

# print()