from enum import Enum, auto
from collections import deque

import random

from permutations import pow

class Bop(Enum):
    AND = auto()
    OR = auto()
    XOR = auto()
    NULL_RHS = auto()
    NULL_LHS = auto()

T = "T"
F = "F"
U = "U"

LEAF = "PLACEHOLDER"

int_to_bool = {
    0.0 : F,
    1.0 : U,
    2.0 : T
}

bool_to_int = {value: key for key, value in int_to_bool.items()}


class EX:
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

class Noise: 
    def __init__(self, garbage_len):
        self.garbage_len = garbage_len

def print_expr_tree(expr, prefix="", is_left=True):
    """
    Recursively prints an expression tree in ASCII form, including:
      - EX nodes (showing Bop operator name)
      - Noise nodes (showing one label 'NOISE' plus 'garbage_len' children)
      - Atomic/leaf nodes: T, F, U, or LEAF
    """
    # Decide which branch symbol to use (├── for left, └── for right)
    branch_symbol = "├── " if is_left else "└── "

    # Handle atomic/leaf nodes (T, F, U, LEAF)
    if expr in [T, F, U, LEAF]:
        print(prefix + branch_symbol + str(expr))
        return

    # Handle Noise nodes
    if isinstance(expr, Noise):
        # Print the parent Noise node
        print(prefix + branch_symbol + f"NOISE({expr.garbage_len})")
        # Increase indentation for the Noise node's children
        child_prefix = prefix + ("│   " if is_left else "    ")
        # Print 'garbage_len' number of child leaves labeled "NOISE"
        for i in range(expr.garbage_len):
            # For intermediate children, use "├──", for last child use "└──"
            sub_is_left = (i < expr.garbage_len - 1)
            sub_branch_symbol = "├── " if sub_is_left else "└── "
            print(child_prefix + sub_branch_symbol + "NOISE")
        return

    # Handle EX nodes
    if isinstance(expr, EX):
        # Print the operator name (e.g. AND, OR, XOR, NULL_RHS, NULL_LHS)
        print(prefix + branch_symbol + expr.op.name)
        child_prefix = prefix + ("│   " if is_left else "    ")
        # Recursively print left subtree
        print_expr_tree(expr.lhs, prefix=child_prefix, is_left=True)
        # Recursively print right subtree
        print_expr_tree(expr.rhs, prefix=child_prefix, is_left=False)
        return

    # If there's an unknown type, just print it directly
    print("unknown type")
    print(prefix + branch_symbol + str(expr))

def invert(x):
    if isinstance(x, EX):
        raise ValueError("Cannot invert an expression; check precedence")
    elif x == T:
        return F
    elif x == F:
        return T
    else:
        return x


def eval_bool3(expr):
    if not isinstance(expr, EX):
        return expr
    
    else:
        op = expr.op
        lhs = expr.lhs
        rhs = expr.rhs

        elhs = eval_bool3(lhs)
        erhs = eval_bool3(rhs)

        if op == Bop.AND: 
            if elhs == F or erhs == F:
                return F
            elif elhs == T and erhs == T:
                return T
            else:
                return U

        elif op == Bop.OR:
            if elhs == T or erhs == T:
                return T
            elif elhs == F and erhs == F:
                return F
            else:
                return U

        elif op == Bop.XOR:
            if elhs == U or erhs == U:
                return U
            elif elhs == erhs:
                return F
            else:
                return T

        elif op == Bop.NULL_RHS:
            return elhs 
        elif op == Bop.NULL_LHS:
            return erhs

        else:
            raise ValueError("Unknown operator")


def causal_responsibility(expr, resp_parent):
    orig = eval_bool3(expr)
    if isinstance(expr, EX):
        op = expr.op
        lhs = expr.lhs
        rhs = expr.rhs

        e_lhs = eval_bool3(lhs)
        e_rhs = eval_bool3(rhs)

        expr_lhs_flipped = EX(op, invert(e_lhs), e_rhs)
        r_lhs = 1.0 if eval_bool3(expr_lhs_flipped) != orig else 0.0

        expr_rhs_flipped = EX(op, e_lhs, invert(e_rhs))
        r_rhs = 1.0 if eval_bool3(expr_rhs_flipped) != orig else 0.0

        expr_both_flipped = EX(op, invert(e_lhs), invert(e_rhs))
        r_tot = 0.5 if eval_bool3(expr_both_flipped) != orig else 0.0

        lhs_responsibility = max(r_lhs, r_tot) * resp_parent
        rhs_responsibility = max(r_rhs, r_tot) * resp_parent

        return (causal_responsibility(lhs, lhs_responsibility) +
                causal_responsibility(rhs, rhs_responsibility))
    
    if isinstance(expr, Noise):
        return [0.0 for _ in range(expr.garbage_len)]
    else:
        return [resp_parent]


def size(expr):
    if expr == LEAF:
        return 1
    elif isinstance(expr, Noise):
        return expr.garbage_len
    else:
        return size(expr.lhs) + size(expr.rhs)


def insert_values(expr, vals, idx):
    if expr == LEAF:
        return int_to_bool[vals[idx]]

    elif isinstance(expr, EX):
        l_size = size(expr.lhs)
        return EX(expr.op, insert_values(expr.lhs, vals, idx), 
            insert_values(expr.rhs, vals, idx + l_size))

    else:
        return expr

def randomize():
    used = 0
    tot = 12

    to_consolidate = deque([LEAF for _ in range(tot)])
    while len(to_consolidate) > 1:
        choice = random.choices([
            Bop.AND,
            Bop.OR,
            Bop.XOR,
            Bop.NULL_LHS,
            Bop.NULL_RHS
        ])[0]
        match choice:
            case Bop.NULL_LHS:
                maxr = int((tot - used)/3) + 1
                if maxr < 2: continue

                null_size = random.randint(1, maxr)
                if not all(c == LEAF for c in list(to_consolidate)[:maxr]): continue

                used += null_size
                for _ in range(null_size): to_consolidate.popleft()

                rhs = to_consolidate.popleft()
                expr = EX(choice, Noise(null_size), rhs)
                to_consolidate.append(expr)
            case Bop.NULL_RHS:
                maxr = int((tot - used)/3) + 1
                if maxr < 2: continue

                null_size = random.randint(1, maxr)
                if not all(c == LEAF for c in list(to_consolidate)[:maxr]): continue

                used += null_size
                for _ in range(null_size): to_consolidate.popleft()

                lhs = to_consolidate.popleft()
                expr = EX(choice, lhs, Noise(null_size))
                to_consolidate.append(expr)

            case op:
                lhs = to_consolidate.popleft()
                rhs = to_consolidate.popleft()

                expr = EX(op, lhs, rhs)
                to_consolidate.append(expr)
    
    return to_consolidate.popleft()
