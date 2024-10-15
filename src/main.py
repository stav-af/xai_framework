from pathlib import Path

import torch
import dataloader
from sklearn.cluster import KMeans

from models import cross_val_train, FeedForwardNN
from explainers import shap_explainer

def get_relative_filepaths(directory):
    directory_path = Path(directory)
    return [str(path.relative_to(directory_path)) for path in directory_path.rglob('*') if path.is_file()]


def train_models():
    data_filepaths = list(filter(lambda x: x.endswith(".csv"), get_relative_filepaths("./data")))
    for file in data_filepaths:
        print(f"Training on {file}")
        
        X, y, explanation = dataloader.load(f"./data/{file}")
        nn = FeedForwardNN(X.shape[1])

        model = cross_val_train(nn, X, y, 5, 200)
        torch.save(model.state_dict(), f'models/{file.replace("csv", "pth")}')


def main():
    model_path = "OR.pth"
    print(model_path)

    X, y, exp = dataloader.load(f'data/{model_path.replace("pth", "csv")}')

    model = FeedForwardNN(X.shape[1])
    model.load_state_dict(torch.load(f"models/{model_path}"))
    alpha = 0.1
    for i in range(1, X.size + 1):
        explanation = shap_explainer(model, X[i - 1: i])[0]
        # cutoff = torch.median(explanation)

        # extracted_explanation = (explanation > cutoff).nonzero(as_tuple=True)[0]

        # print(explanation)
        # print(extracted_explanation)
        abs_explanation = torch.abs(explanation)

        # Initialize EMA as the first element
        ema = abs_explanation[0]

        # Calculate EMA across the explanation values
        for value in abs_explanation[1:]:
            ema = alpha * value + (1 - alpha) * ema  # EMA update formula

        # Get the indices of the values that are larger than the EMA
        large_value_indices = (abs_explanation > ema).nonzero(as_tuple=True)[0]
        print(large_value_indices)
        print(explanation)
        print(y[i])

    print(avg)
    avg = [0 if val < 0.1 else 1 for val in avg]
    print(avg)

if __name__ == "__main__":
    main()