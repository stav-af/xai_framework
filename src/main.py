from pathlib import Path

import csv
import itertools
import torch
import numpy as np
import dataloader
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

from rex.rex import norm
from models import cross_val_train, FeedForwardNN
from explainers import kernel_shap, saliency, shapely_values, deeplift, lrp, deeplift, input_x_grad, integrated_grad, rexplain
from responsibility import *
from expression_bank import *
import permutations
import types
import math



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

formula_structures = {
    "and.pth": and_exp, 
    "or.pth": or_exp,
    "xor.pth": xor_exp(),
    "xor_and_xor.pth": xor_and(),
    "xor_and_3.pth": n_xor_and(3),
    "xor_and_4.pth": n_xor_and(4),
    "xor_and_5.pth": n_xor_and(5),
    "xor_and_6.pth": n_xor_and(6),
    "xor_and_7.pth": n_xor_and(7),
    "xor_and_8.pth": n_xor_and(8),
    "xor_and_9.pth": n_xor_and(9),
    "xor_and_10.pth": n_xor_and(10),
    "and_or_3.pth": n_and_or(3),
    "and_or_4.pth":n_and_or(4),
    "and_or_5.pth":n_and_or(5),
    "and_or_6.pth":n_and_or(6),
    "and_or_7.pth":n_and_or(7),
    "and_or_8.pth":n_and_or(8),
    "and_or_9.pth":n_and_or(9),
    "and_or_10.pth":n_and_or(10),
}


explainers = {
    # "DeepLiftShap": deeplift_shap,
    "ReX": rexplain,
    "ShapleyValues": shapely_values,
    "KernelShap": kernel_shap,
    "Saliency": saliency,
    "DeepLift": deeplift,
    "lrp": lrp,
    "InputXGrad": input_x_grad,
    "IntegratedGrad": integrated_grad,
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


def exp_mse(truth, result):
    norm(truth)
    norm(result)
    se = 0
    for lhs,rhs in zip(truth, result):
        se += (lhs - rhs) ** 2

    return se / 12


def js_divergence(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    p = np.abs(p)
    q = np.abs(q)

    p = p / np.sum(p)
    q = q / np.sum(q)
    
    m = 0.5 * (p + q)
    
    def kl_divergence(a, b):
        mask = (a != 0)
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))
    
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    
    if math.isnan(js_div): 
        print(f"nan on iuts \n{p}\n{q} ")
        exit(0)
    return abs(js_div)


def bool_test():
    test_size = 500

    results = [["models", *[exp_name for exp_name in explainers.keys()]]]
    for model_name in model_testset.keys(): 
        results.append([model_name])

        X = pow([[0], [1], [2]], 12) 
        random.shuffle(X)
        trees = [insert_values_arr(formula_structures[model_name], x) for x in X]
        y = [bool_to_int[eval_bool3(t)] for t in trees]


        torchX = torch.tensor(X, dtype=torch.float32)
        torchy = torch.tensor(y, dtype=torch.float32)

        nn = FeedForwardNN(12)
        nn.load_state_dict(torch.load(f"models_3val/{model_name}"))
        cross_val_train(nn, torchX, torchy, 5, 300)

        testX = pow([[0], [2]], 12)
        random.shuffle(testX)
        testtree = [insert_values_arr(formula_structures[model_name], x) for x in testX]

        print_expr_tree(testtree)
        testy = [bool_to_int[eval_bool3(t)] for t in testtree]
        exps = [causal_responsibility(t, 1.0) for t in testtree]
        for exp_name in results[0][1:]:

            loss = 0
            exp_func = explainers[exp_name]
            for i in range(test_size):
                tfeature_importance = exp_func(
                        nn, 
                        torch.tensor(testX[i: i+1], dtype=torch.float32, requires_grad=True),
                        target=int(testy[i]))
                
                feature_importance = tfeature_importance.cpu().tolist()
                ground_truth = exps[i]
                
                print(f"\n\n\n {exp_name}, {model_name}:\n")
                print(testX[i])
                print(norm(ground_truth))
                print(norm(feature_importance[0]))


                loss += js_divergence((ground_truth), (feature_importance[0])) / test_size
            results[-1].append(loss)

    np.savetxt('causal_resp.txt', results, fmt='%s', delimiter=",")

def cause_test():
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
            for i in range(0, 4095):
                feature_importance = exp_func(
                        nn, 
                        torch.tensor(X[i: i+1], dtype=torch.float32, requires_grad=True),
                        target=int(y[i]))
                
                # exp_extracted = top_n_features(feature_importance, len(exps_correct[i]))
                exp_extracted = cluster_explanation(feature_importance)
                if set(exp_extracted) == set(exps_correct[i]): accuracy += 1
                if i > 0: print(exp_name, ": ", (accuracy) / (i + 1)) 
            results[-1].append(accuracy / 500)

    np.savetxt('naive_resp.txt', results, fmt='%s', delimiter=",")


def rand_test():
    output_filename = "rand_test_results.csv"
    
    header = ["num_vars", "instance", *list(explainers.keys())]
    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    for num_vars in range(3, 10):
        print(f"Processing formulas with {num_vars} variables...")
        instance_losses = {exp_name: [] for exp_name in explainers.keys()}
        instance_counter = 0

        valid_instances = []
        while len(valid_instances) < 10:
            bool_structure = randomize(num_vars)
            # print_expr_tree(bool_structure)            


            X = pow([[0], [2]], 12)
            random.shuffle(X)
            
            testtree = [insert_values_arr(bool_structure, x) for x in X]

            prev = U
            for t in testtree:
                curr = eval_bool3(t)
                if curr != prev:
                    print(curr)
                    print_expr_tree(t)
                prev = curr


            y = [eval_bool3(t) for t in testtree]
            # print(y)
            print("\n\n\n\n")
            if len(set(y)) == 1:
                continue
            else:
                valid_instances.append((bool_structure, X, y, testtree))
            print("found!")
        for inst_idx, (bool_structure, X, y, testtree) in enumerate(valid_instances):
            nn = FeedForwardNN(12)
            rows = pow([[0], [1], [2]], 12)

            random.shuffle(rows)

            X_tensor = torch.tensor(rows, dtype=torch.float32)
            trees = [insert_values_arr(bool_structure, row) for row in rows]

            y_tensor = torch.tensor([bool_to_int[eval_bool3(t)] for t in trees], dtype=torch.float32)
            cross_val_train(nn, X_tensor, y_tensor, 5, 100)

            exps = [causal_responsibility(t, 1.0) for t in trees]

            losses_this_instance = {}
            trial_count = len(X)
            for exp_name, exp_func in explainers.items():
                loss_sum = 0
                for trial_idx in range(trial_count):
                    input_tensor = torch.tensor([X[trial_idx]], dtype=torch.float32, requires_grad=True)
                    target = int(bool_to_int[y[trial_idx]])
                    
                    explanation = exp_func(nn, input_tensor, target=target)
                    if isinstance(explanation, torch.Tensor) and explanation.ndim == 2:
                        explanation = explanation[0]
                    
                    loss_sum += js_divergence(explanation, exps[trial_idx])
                avg_loss = loss_sum / trial_count
                losses_this_instance[exp_name] = avg_loss
            
            for exp_name in explainers.keys():
                instance_losses[exp_name].append(losses_this_instance[exp_name])
            
            row = [num_vars, inst_idx]
            for exp_name in explainers.keys():

                row.append(losses_this_instance[exp_name])
            with open(output_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            instance_counter += 1
        
        avg_losses = {exp_name: np.mean(instance_losses[exp_name]) for exp_name in explainers.keys()}
        print(f"Average losses for {num_vars} variables: {avg_losses}")

        summary_row = [f"{num_vars}_avg", "all"]
        for exp_name in explainers.keys():
            summary_row.append(avg_losses[exp_name])
        with open(output_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(summary_row)

if __name__ == "__main__":
    # main()
    # train_synth()
    # bool_test()
    rand_test()
    # bool_train()
    # print_expr_tree(n_and_or(10))