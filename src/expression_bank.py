from responsibility import Bop, EX, LEAF


and_leaf = EX(Bop.AND, LEAF(1), LEAF(2))
or_leaf  = EX(Bop.OR, LEAF(1), LEAF(2))
xor_leaf = EX(Bop.XOR, LEAF(1), LEAF(2))

def and_exp():
    return EX(Bop.AND, and_leaf, LEAF(3))

def or_exp():
    return EX(Bop.OR, or_leaf, LEAF(3))

def xor_exp():
    return EX(Bop.XOR, xor_leaf, LEAF(3))

def xor_and():
    return EX(Bop.AND, xor_exp(), xor_exp())

def n_xor_and(n):
    if n < 3:
        raise ValueError("n_xor_and requires n >= 3")
    parent = EX(Bop.XOR, LEAF(0), LEAF(1))
    for i in range(2, n + 1):
        op = Bop.AND if (i % 2 == 0) else Bop.XOR
        parent = EX(op, parent, LEAF(str(i)))
    return parent

def n_and_or(n):
    if n < 3:
        raise ValueError("n_and_or requires n >= 3")
    parent = EX(Bop.AND, LEAF(0), LEAF(1))
    for i in range(2, n + 1):
        op = Bop.OR if (i % 2 == 0) else Bop.AND
        parent = EX(op, parent, LEAF(str(i)))
    return parent