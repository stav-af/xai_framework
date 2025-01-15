
from responsibility import *

TOTAL_SIZE = 12

and_leaf = EX(Bop.AND, LEAF, LEAF)
or_leaf = EX(Bop.OR, LEAF, LEAF)
xor_leaf = EX(Bop.XOR, LEAF, LEAF)

and_exp = EX(Bop.NULL_RHS, and_leaf, Noise(10))
or_exp = EX(Bop.NULL_RHS, or_leaf, Noise(10))
xor_exp = EX(Bop.NULL_RHS, EX(Bop.XOR, LEAF, LEAF), Noise(10))

xor_and = EX(Bop.NULL_RHS, EX(Bop.AND, xor_leaf, xor_leaf), Noise(8))

print_expr_tree(xor_and)

def n_xor_and(n):
    if n < 3:
        raise ValueError("Works in range 3 - 12")
    
    parent = EX(Bop.XOR, LEAF, LEAF)
    for i in range(2, n):
        op = Bop.AND if not i % 2 else Bop.XOR
        parent = EX(op, parent, LEAF)
    
    res =  EX(Bop.NULL_RHS, parent, Noise(TOTAL_SIZE - n))
    print_expr_tree(res)
    return res

def n_and_or(n):
    if n < 3:
        raise ValueError("Works in range 3 - 12")
    
    parent = EX(Bop.AND, LEAF, LEAF)
    for i in range(2, n):
        op = Bop.OR if not i % 2 else Bop.AND
        parent = EX(op, parent, LEAF)

    res = EX(Bop.NULL_RHS, parent, Noise(TOTAL_SIZE - n))
    return res 

