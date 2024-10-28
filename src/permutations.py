import numpy as np
import torch
import random

from bool3 import B
from functools import partial


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

def gen_and(row):
    return B(row[0]) & B(row[1])

def gen_or(row):
    return B(row[0]) | B(row[1])

def gen_xor(row):
    return B(row[0]) ^ B(row[1])

def gen_xor_and_xor(row):
    return ((B(row[0]) ^ B(row[1])) & (B(row[2]) ^ B(row[3])))

def gen_n_xor_and(n, row):
    result = B(row[0]) ^ B(row[1])
    for i in range(2, n):
        operand = B(row[i])
        if i % 2 == 0:
            result &= operand
        else:
            result ^= operand

    return result

def gen_n_and_or(n, row):
    result = B(row[0]) & B(row[1])
    for i in range(2, n):
        operand = B(row[i])
        if i % 2 == 0:
            result |= operand
        else:
            result &= operand

    return result


gen_funcs = {
    "and": gen_and,
    "or": gen_or,
    "xor": gen_xor,
    "xor_and_xor": gen_xor_and_xor,
    "xor_and_3": partial(gen_n_xor_and, 3),
    "xor_and_4": partial(gen_n_xor_and, 4),
    "xor_and_5": partial(gen_n_xor_and, 5),
    "xor_and_6": partial(gen_n_xor_and, 6),
    "xor_and_7": partial(gen_n_xor_and, 7),
    "xor_and_8": partial(gen_n_xor_and, 8),
    "xor_and_9": partial(gen_n_xor_and, 9),
    "xor_and_10": partial(gen_n_xor_and, 10),
    "and_or_3": partial(gen_n_and_or, 3),
    "and_or_4": partial(gen_n_and_or, 4),
    "and_or_5": partial(gen_n_and_or, 5),
    "and_or_6": partial(gen_n_and_or, 6),
    "and_or_7": partial(gen_n_and_or, 7),
    "and_or_8": partial(gen_n_and_or, 8),
    "and_or_9": partial(gen_n_and_or, 9),
    "and_or_10": partial(gen_n_and_or, 10),
}


def apply(func, base):
    result = []
    for row in base:
        result.append([*row, func(row).value])

    return result

def generate():
    base = pow([[2], [1], [0]], 12)
    for name, func in gen_funcs.items():
        result = apply(func, base)
        
        random.shuffle(result)
        yield torch.tensor([arr[:12] for arr in result], dtype=torch.float32), \
                torch.tensor([arr[12] for arr in result], dtype=torch.float32), \
                    name

if __name__ == "__main__":
    base = pow([[2], [1], [0]], 12)
    gen_n_xor_and(3, base[0])    




