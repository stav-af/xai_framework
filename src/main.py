from pathlib import Path

import torch
import dataloader
from model import cross_val_train, FeedForwardNN

def get_relative_filepaths(directory):
    directory_path = Path(directory)
    return [str(path.relative_to(directory_path)) for path in directory_path.rglob('*') if path.is_file()]



def main():
    data_filepaths = list(filter(lambda x: x.endswith(".csv"), get_relative_filepaths("./data"))) 
    for file in data_filepaths:
        print(f"Training on {file}")
        X, y, explanation = dataloader.load(f"./data/{file}")
        nn = FeedForwardNN(X.shape[1])

        model = cross_val_train(nn, X, y, 5, 200)
        torch.save(model.state_dict(), f'models/{file.replace("csv", "pth")}')


if __name__ == "__main__":
    main()