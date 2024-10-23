import numpy as np
import torch
import random

from bool3 import B



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
    return ((B(row[0]) ^ B(row[1])) & (B(row[2]) ^ B(row[3]))).value

def gen_n_xor_and(n):
    n -= 1
    def gen_xor_and(row):
        if n < 3:
            return B(row[0]) ^ B(row[1])
        elif n % 2 == 1:
            return lambda row: gen_n_xor_and(row) & B(row[n])
        else:
            return lambda row: gen_xor_and(row) ^ B(row[n])
    
    return gen_xor_and


gen_funcs = {
    # gen_and,
    # gen_or,
    # gen_xor,
    # gen_xor_and_xor,
    "xor_and_3": gen_n_xor_and(3),
    "xor_and_4": gen_n_xor_and(4),
    "xor_and_5": gen_n_xor_and(5)
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
    generate()
