
from collections import deque
from itertools import combinations
from dataclasses import dataclass
from typing import List

import random

import torch

"""CONSTANTS"""
N_PARTITIONS = 4
NEUTRAL_VALUE = 1
SERACH_DEPTH = 2

@dataclass
class Mutant:
    partitions: List[List[int]]
    data: List[int]


def wrap_forward(nn):
    def forward(data):
        tdata = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        res = nn.forward(tdata)
        pred = torch.argmax(res, dim=1)
        return pred.item()
    return forward

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

def partition(responsibility, choices, n_parts):
    if all(x == 0.0 for x in responsibility):
        n = len(responsibility)
        nonzero_resp = [1.0 for _ in range(n)]
    else:
        nonzero_resp = responsibility

    raw_options = norm(nonzero_resp)
    avg_resp = len(choices) / (float(n_parts) * len(responsibility))
    def aux(acc, curr, score, used):
        resp_options = [x if (i in choices and i not in used) else 0.0 
                        for i, x in enumerate(raw_options)]

        norm_resp_options = norm(resp_options)
        steps = cdf(norm_resp_options)

        if (set(used) == set(choices)) or all(x == 0 for x in resp_options):
            return acc

        rand = random.random()
        curr_choice = find_index(lambda x: x > rand, steps)
        if curr_choice is None:
            raise RuntimeError("Shouldn't really be here")

        curr_resp = resp_options[curr_choice]
        next_partition = curr + [curr_choice]
        next_score = score + curr_resp

        if next_score >= avg_resp:
            return aux(acc + [next_partition], [], 0.0, used + [curr_choice])
        else:
            return aux(acc, next_partition, next_score, used + [curr_choice])

    res = aux([], [], 0.0, [])
    return res


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


def rex(nn, feature_values):
    forward = wrap_forward(nn)
    initial_prediction = forward(feature_values)
    feature_importance = [0 for _ in range(len(feature_values))]
    # init partition search queue with feature indices
    for _ in range(25):
        part_q = deque()
        part_q.append((list(range(len(feature_values))), 1.0))

        while not len(part_q) == 0:
            indices, parent_resp = part_q.popleft()
            partitions = partition(feature_importance, indices, N_PARTITIONS)

            mutants = []
            for combo in partitions_combinations(partitions):
                occluded_data = occlude(feature_values, combo)
                mut = Mutant(partitions=combo, data=occluded_data)
                mutants.append(mut)

            cst_set = list(filter(
                lambda mut: forward(mut.data) == initial_prediction,
                mutants
            ))

            for part in partitions:
                resp = responsibility(part, cst_set)
                feature_importance = apply_responsibility(feature_importance, part, resp * parent_resp)
                
                if resp > 0 and set(part) != set(indices) and len(part) > 1:
                    part_q.append((part, resp))


    return feature_importance


if __name__ == "__main__":
    X, y, exps_correct = dataloader.load(f"data/AND.csv")

    nn = FeedForwardNN(12)
    nn.load_state_dict(torch.load(f"models_3val/and.pth"))

    inp = [0 for _ in range(12)]
    print(rex(inp, wrap_forward(nn)))