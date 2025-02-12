
from itertools import combinations
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

import sys
import random

from responsibility import eval_bool3, randomize, insert_values_arr, print_expr_tree

"""CONSTANTS"""
N_PARTITIONS = 12
NEUTRAL_VALUE = 1
SERACH_DEPTH = 1

@dataclass
class Mutant:
    partitions: List[List[int]]
    data: List[int]


def wrap_eval(expr):
    def eval(data):
        populated = insert_values_arr(expr, data)
        return eval_bool3(populated)
    return eval


def cdf(arr):
    result = []
    cum = 0.0
    for x in arr:
        cum += x
        result.append(cum)
    return result

def norm(arr):
    s = sum(arr)
    if s == 0.0:
        n = len(arr)
        return [1.0 / n for _ in range(n)]
    else:
        return [x / s for x in arr]


def find_index(f, lst):
    for i, x in enumerate(lst):
        if f(x):
            return i
    return None


def powerset(iterable):
    s = list(iterable)
    return [list(combo) for r in range(len(s)+1) for combo in combinations(s, r)]


def partitions_combinations(partitions):
    return  list(filter(
        lambda x: len(x) < len(partitions),
        powerset(partitions)))


def occlude(data, partitions):
    acc = []
    for (idx, elem) in enumerate(data):
        if all((not idx in partition) for partition in partitions):
            acc.append(elem)
        else:
            acc.append(NEUTRAL_VALUE)

    return acc

def apply_responsibility(feature_importance, part, responsibility):
    distributed_resp = responsibility / len(part)
    ret = []
    for (idx, elem) in enumerate(feature_importance):
        if idx in part:
            ret.append(distributed_resp)
        else:
            ret.append(elem) 

    return ret


def partitions_equal(partitions_a, partitions_b):
    set_a = {tuple(p) if isinstance(p, list) else p for p in partitions_a}
    set_b = {tuple(p) if isinstance(p, list) else p for p in partitions_b}

    return set_a == set_b


def responsibility(part: List, consistent_set: List[List]) -> float:
    consistent_with_partition = [mut.partitions for mut in consistent_set if part not in mut.partitions]

    adding_part_changes_prediction = []
    for cst_part in consistent_with_partition:
        maybe_consistent = [part] + cst_part
        if not any(partitions_equal(maybe_consistent, cst.partitions) for cst in consistent_set):
            adding_part_changes_prediction.append(cst_part) 

    if not adding_part_changes_prediction:
        return 0.0

    min_mutant = min(adding_part_changes_prediction, key=len)
    minpart = len(min_mutant)

    return 1.0 / (1.0 + float(minpart))


def bf_resp(expr, feature_values):
    eval = wrap_eval(expr)
    initial_result = eval(feature_values)
    print(initial_result)
    
    feature_importance = [0 for _ in range(len(feature_values))]
    partitions = [[n] for n in range(len(feature_values))]

    mutants = []
    
    for combo in partitions_combinations(partitions):
        occluded_data = occlude(feature_values, combo)
        mut = Mutant(partitions=combo, data=occluded_data)
        mutants.append(mut)

    cst_set = list(filter(
        lambda mut: eval(mut.data) == initial_result,
        mutants
    ))

    for part in tqdm(partitions):
        resp = responsibility(part, cst_set)
        feature_importance = apply_responsibility(feature_importance, part, resp)
        

    return feature_importance


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 5 
    rand = randomize(size)

    print_expr_tree(rand)
    print("\n\n\n")

    result = bf_resp(rand, [2.0 for _ in range(12)])
    print(result)