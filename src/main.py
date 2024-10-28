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
import types

model_testset = {
    "and.pth": "AND.csv",
    "or.pth": "OR.csv",
    "xor.pth": "XOR.csv",
    "xor_and_xor.pth": "XOR_AND_XOR.csv",
    "xor_and_3.pth": "XOR_AND/XOR_AND_3.csv",
    "xor_and_4.pth": "XOR_AND/XOR_AND_4.csv",
    "xor_and_5.pth": "XOR_AND/XOR_AND_5.csv",
    "xor_and_6.pth": "XOR_AND/XOR_AND_6.csv",
    "xor_and_7.pth": "XOR_AND/XOR_AND_7.csv",
    "xor_and_8.pth": "XOR_AND/XOR_AND_8.csv",
    "xor_and_9.pth": "XOR_AND/XOR_AND_9.csv",
    "xor_and_10.pth": "XOR_AND/XOR_AND_10.csv",
    "and_or_3.pth": "AND_OR/AND_OR_3.csv",
    "and_or_4.pth": "AND_OR/AND_OR_4.csv",
    "and_or_5.pth": "AND_OR/AND_OR_5.csv",
    "and_or_6.pth": "AND_OR/AND_OR_6.csv",
    "and_or_7.pth": "AND_OR/AND_OR_7.csv",
    "and_or_8.pth": "AND_OR/AND_OR_8.csv",
    "and_or_9.pth": "AND_OR/AND_OR_9.csv",
    "and_or_10.pth": "AND_OR/AND_OR_10.csv"
}

explainers = {
    # "DeepLiftShap": deeplift_shap,
    "ShapleyValues": shapely_values,
    "KernelShap": kernel_shap,
    "Saliency": saliency,
    "DeepLift": deeplift,
    "lrp": lrp,
    "InputXGrad": input_x_grad,
    "IntegratedGrad": integrated_grad
}

def hack_forward(nn):
    """
        Gradient-based explainers implicitly use a 0 baseline. 
        This hacks the forward function to output classes -1, 0, 1 instead of 0, 1, 2
        This gives us acceptable performance in the grad based explainers
    
        But we can still use standard classification loss functions (CE)
    """
    original_forward = nn.forward
    def new_forward(self, x):
        return original_forward(x) - 1
    
    nn.forward = types.MethodType(new_forward, nn)
    return nn


def get_relative_filepaths(directory):
    directory_path = Path(directory)
    return [str(path.relative_to(directory_path)) for path in directory_path.rglob('*') if path.is_file()]


def cluster_explanation(vector):
    vector_np = vector.detach().numpy()
    abs_vector = np.abs(vector_np).reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(abs_vector)
    cluster_centers = kmeans.cluster_centers_
    
    relevant_cluster = np.argmax(cluster_centers)
    labels = kmeans.labels_
    
    relevant_indices = np.where(labels == relevant_cluster)[0]
    return relevant_indices.tolist()


def top_n_features(vector, n):
    _, indices = torch.topk(vector, n)
    indices_list = indices.tolist()

    return set(indices_list[0])

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
    results = [["models", *[exp_name for exp_name in explainers.keys()]]]
    for model_name in model_testset.keys(): 
        results.append([model_name])
        
        X, y, exps_correct = dataloader.load(f"data/{model_testset[model_name]}")
        nn = FeedForwardNN(12)
        nn.load_state_dict(torch.load(f"models_3val/{model_name}"))
        
        # return -1, 1 instead of 0, 2
        # nn = hack_forward(nn)
        for exp_name in results[0][1:]:

            accuracy = 0
            exp_func = explainers[exp_name]
            for i in range(0, 4096):
                feature_importance = exp_func(
                        nn, 
                        torch.tensor(X[i: i+1], dtype=torch.float32, requires_grad=True),
                        target=int(y[i]))
                
                exp_extracted = top_n_features(feature_importance, len(exps_correct[i]))
                if exp_extracted == set(exps_correct[i]): accuracy += 1
                
                print(exp_name, model_name)
                print(exp_extracted, exps_correct[i], X[i])
                if i > 0: print((accuracy) / (i + 1)) 
            results[-1].append(accuracy / 4096)

    np.savetxt('text.txt', results, fmt='%s')


        
if __name__ == "__main__":
    # main()
    # train_synth()
    bool_test()
    # bool_train()