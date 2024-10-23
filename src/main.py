from pathlib import Path

import torch
import numpy as np
import dataloader
import json
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from models import cross_val_train, FeedForwardNN
from explainers import kernel_shap, saliency, shapely_values, deeplift, lrp, deeplift, input_x_grad, integrated_grad
import permutations

explainers = {
    # "DeepLiftShap": deeplift_shap,
    "KernelShap": kernel_shap,
    "Saliency": saliency,
    "ShapleyValues": shapely_values,
    "DeepLift": deeplift,
    "lrp": lrp,
    "InputXGrad": input_x_grad,
    "IntegratedGrad": integrated_grad
}


def get_relative_filepaths(directory):
    directory_path = Path(directory)
    return [str(path.relative_to(directory_path)) for path in directory_path.rglob('*') if path.is_file()]


def cluster_explanation(vector):
    vector_np = vector.detach().numpy()
    abs_vector = np.abs(vector_np).reshape(-1, 1)  # Reshape for clustering
    
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(abs_vector)
    cluster_centers = kmeans.cluster_centers_
    
    relevant_cluster = np.argmax(cluster_centers)
    labels = kmeans.labels_
    
    relevant_indices = np.where(labels == relevant_cluster)[0]
    return relevant_indices.tolist()


def cluster_baselines(data):
    kmeans = KMeans(n_clusters=20, random_state=0)
    kmeans.fit(data)

    centroids = kmeans.cluster_centers_
    return centroids


def bool_train():
    for X, y, name in permutations.generate():
        nn = FeedForwardNN(12)
        nn = cross_val_train(nn, X, y, 5, 200)
        torch.save(nn.state_dict(), f"models_3val/{name}.pth")    


def bool_test():
    X, y, _ = next(permutations.generate())

    print(X[0], y[0])
    nn = FeedForwardNN(12)
    # nn = cross_val_train(nn, X, y, 5, 200)
    nn.load_state_dict(torch.load("models_3val/AND.pth"))
    # torch.save(nn.state_dict(), f"models_3val/AND.pth")
    
    results = {}
    for name in explainers.keys(): results[name] = 0
    
    X, y, exp = dataloader.load("data/AND.csv")
    for i in range(2**12):
        print("\n\n\n\n")
        for name, f_exp in explainers.items():
            ex = cluster_explanation(
                f_exp(nn, 
                      torch.tensor(X[i: i + 1], dtype=torch.float32, requires_grad=True),
                      int(y[i])
                )
            )
            if ex == exp[i]:
                results[name] += 1

        [print(exp, (n/(i + 1))) for exp, n in results.items()]   




def full_test():
    test_set_paths = list(filter(lambda x: x.endswith(".csv"), get_relative_filepaths("./data")))
    full_result = {"Headers": [dataset for dataset in test_set_paths]}
    for expname in explainers.keys(): full_result[expname] = []

    for path in test_set_paths:
        X, y, exp = dataloader.load(f"data/{path}")
        baselines = torch.tensor(cluster_baselines(X), dtype=torch.float32)

        nn = FeedForwardNN(X.shape[1])
        nn = cross_val_train(nn, X, y, 5, 200)

        torch.save(nn.state_dict(), f"models/{path}".replace(".csv", ".pth"))

        results = {}
        for name in explainers.keys(): results[name] = 0
        
        for i in range(1, 4067, 40):
            print("\n\n\n\n")
            for name, f_exp in explainers.items():
                ex = cluster_explanation(f_exp(nn, torch.tensor(X[i - 1: i], dtype=torch.float32, requires_grad=True)))
                if ex == exp[i - 1]:
                    results[name] += 1

            [print(exp, (n*40)/i) for exp, n in results.items()]   

        for name, result in results.items():
            full_result[name].append((result/4096) * 40)
        
    print(full_result)
    with open("initial_dump.pkl", 'w') as file:
        json.dump(full_result, file, indent=4)
        
if __name__ == "__main__":
    # main()
    # train_synth()
    # full_test()
    bool_train()