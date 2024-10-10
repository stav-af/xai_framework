import numpy as np
import re

def extract_idx(s):
    pattern = r'f(\d+)'
    return int(re.search(pattern, s).group(1)) - 1


def load(DATASET_LOCATION):
    """
    Takes the location of a dataset with 12 binary input features, 1 label and a list of minimum explanations
    returns:
       X <- np arr,
       y <- np arr,
       label <- python array
    """
    with open(DATASET_LOCATION) as f:
        data = [x.strip().split(",") for x in f.readlines()][1:]
        X_raw = [[1 if val == '1' else -1 for val in row[:12]] for row in data]
        y_raw = [int(row[12]) for row in data]
        explanation_raw = [row[13:] for row in data]
    
    X = np.array(X_raw)
    y = np.array(y_raw)
    
    explanation = [[extract_idx(exp) for exp in explanations] for explanations in explanation_raw]
    return X, y, explanation
    