import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score



class FeedForwardNN(nn.Module):
    def __init__(self, input_size, num_classes=3):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 20)
        self.fc6 = nn.Linear(20, 3)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()   

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.fc6(x)
        return x


def train_model(model, X_train, y_train, X_val, y_val, epochs=1000, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.long)
    val_inputs = torch.tensor(X_val, dtype=torch.float32)    
    for epoch in range(epochs):
        model.train()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad(): 
            val_outputs = model(val_inputs)
            val_preds = torch.argmax(val_outputs, dim=1)

            val_preds = val_preds.numpy()
            val_acc = accuracy_score(y_val, val_preds)
        
        if epoch % 5 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}')
    
    return model


def cross_val_train(model, X, y, n_splits=5, epochs=1000, lr=0.001):
    skf = StratifiedKFold(n_splits=n_splits)
    fold_accuracy = 0.0

    while fold_accuracy != 1.0:
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr)
            
            val_inputs = torch.tensor(X_val, dtype=torch.float32)
            val_outputs = model(val_inputs)

            val_preds = torch.argmax(val_outputs, dim=1)

            val_preds = val_preds.numpy()
            val_acc = accuracy_score(y_val, val_preds)

            fold_accuracy = val_acc

    
    return model


