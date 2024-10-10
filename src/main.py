from pathlib import Path

import torch
import dataloader

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
    model_path = "XOR.pth"
    print(model_path)

    X, y, exp = dataloader.load(f'data/{model_path.replace("pth", "csv")}')

    model = FeedForwardNN(X.shape[1])
    model.load_state_dict(torch.load(f"models/{model_path}"))

    sample_size = 10
    avg = [0] * 12
    for explanation in shap_explainer(model, X[:sample_size]):
        for i, val in enumerate(explanation):
            avg[i] += val

    for i, val in enumerate(avg):
        avg[i] = val/sample_size

    print(avg)
    avg = [0 if val < 0.1 else 1 for val in avg]
    print(avg)
        



if __name__ == "__main__":
    main()