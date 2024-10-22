import numpy as np
import torch

truth_table_and = {
    (-1, -1): -1,
    (-1, 1): -1,    
    (-1, 0): -1,    
    (1, -1): -1,    
    (1, 1): 1,      
    (1, 0): 0,      
    (0, -1): -1,    
    (0, 1): 0,      
    (0, 0): 0       
}

def concat(base, add):
    retval = []
    for prefix in base:
        for suffix in add:
            retval.append([*prefix, *suffix])
    return retval

def pow(base, n):
    print(f"base: {base} n: {n}")
    accumulator = base
    for _ in range(n - 1):
        accumulator = concat(accumulator, base)
    return accumulator

def and_2(perms):
    for row in perms:
        row.append(truth_table_and[(row[0], row[1])])

    arr = np.array(perms)
    np.random.shuffle(arr)

    X = arr[:, :12]      
    y = arr[:, 12] 

    exps = []
    for i in range(X.shape[0]):  
        a, b = X[i][0], X[i][1]
        if a == b:
            exps.append([0, 1])
        elif a == 1:
            exps.append([1])
        elif b == 1:
            exps.append([0])
        elif a == -1:
            exps.append([0])
        elif b == -1:
            exps.append([1])

    return X, y, exps


def generate():
    result = pow([[-1], [0], [1]], 12)
    X, y, exp = and_2(result)
    print(X[0], y[0], exp[0])

    return and_2(result)
