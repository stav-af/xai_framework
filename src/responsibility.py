from enum import Enum, auto
from collections import defaultdict, deque

import random

from permutations import pow

def print_expr_tree(expr, prefix="", is_left=True):
    """
    Recursively pretty prints an expression tree in ASCII form.
    Handles EX nodes (with operators) and LEAF nodes (with names and values).
    """
    # Decide branch symbol (├── for left child, └── for right child)
    branch_symbol = "├── " if is_left else "└── "

    # Handle LEAF nodes
    if isinstance(expr, LEAF):
        value_str = f" ({expr.val})" if expr.val is not None else ""
        print(prefix + branch_symbol + f"LEAF({expr.name}){value_str}")
        return

    # Handle EX nodes
    if isinstance(expr, EX):
        # Print the operator name (e.g., AND, OR, XOR, NOT, NULL_RHS, NULL_LHS)
        print(prefix + branch_symbol + f"{expr.op.name}")

        # Update the prefix for children
        child_prefix = prefix + ("│   " if is_left else "    ")

        # Handle unary operator (NOT)
        if expr.op == Bop.NOT:
            print_expr_tree(expr.lhs, prefix=child_prefix, is_left=True)
        else:
            # Recursively print left subtree
            print_expr_tree(expr.lhs, prefix=child_prefix, is_left=True)
            # Recursively print right subtree
            if expr.rhs is not None:
                print_expr_tree(expr.rhs, prefix=child_prefix, is_left=False)

        return

    # Catch-all for unknown types (not expected in normal operation)
    print(prefix + branch_symbol + str(expr))

def merge_dicts(lhs, rhs):
    for key, val in rhs.items():
        lhs[key].extend(val)

    return lhs
    

def causal_responsibility(expr, resp_parent):
    orig = eval_bool3(expr)
    if isinstance(expr, EX):
        op = expr.op
        if op == Bop.NOT:
            return causal_responsibility(expr.lhs, resp_parent)

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

        return merge_dicts(causal_responsibility(lhs, lhs_responsibility),
                            causal_responsibility(rhs, rhs_responsibility))
    
    ret = defaultdict(list)
    ret[expr.name].append(resp_parent)
    return ret


class Bop(Enum):
    AND = auto()
    OR = auto()
    XOR = auto()
    NULL_RHS = auto()
    NULL_LHS = auto()
    NOT = auto()

class EX:
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

class LEAF:
    def __init__(self, name, val=None):
        self.name = name
        self.val = val

    def __repr__(self):
        return f"{self.name}: {self.val}"

T = "T"
F = "F"
U = "U"

int_to_bool = {
    0.0 : F,
    1.0 : U,
    2.0 : T
}

bool_to_int = {value: key for key, value in int_to_bool.items()}

def invert(leaf):
    if leaf == U: return U
    elif leaf == F: return T
    elif leaf == T: return F

    raise ValueError("whatthefuckkingrn")

def eval_bool3(expr):
    if isinstance(expr, LEAF):
        return int_to_bool[expr.val]
    elif isinstance(expr, str):
        return expr
    else:
        op = expr.op
        if op == Bop.NOT:
            lhs_val = eval_bool3(expr.lhs)
            if lhs_val == U:
                return U
            else:
                return T if lhs_val == F else F

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

        raise ValueError("Unknown operator")


def randomize(complexity):
    # Create a pool of LEAF nodes with unique names
    leaves = [LEAF(str(i)) for i in range(complexity)] * 2
    random.shuffle(leaves)
    leaf_pool = deque(leaves)

    nulls_left = 2
    complexity_left = complexity
    while len(leaf_pool) > 1:
        if complexity_left == 0:
            leaf_pool = deque(filter(lambda x: not isinstance(x, LEAF), leaf_pool))

        op_nulls = [Bop.AND, Bop.OR, Bop.XOR, Bop.NULL_LHS, Bop.NULL_RHS, Bop.NOT]
        op_nonull = [Bop.AND, Bop.OR, Bop.XOR, Bop.NOT]

        choice = random.choices(
            op_nulls if nulls_left > 1 else op_nonull
        )[0]

        match choice:
            case Bop.NOT:
                # Choose a random leaf and wrap it in a NOT operator
                leaf = leaf_pool.popleft()
                expr = EX(Bop.NOT, leaf, None)  # NOT only needs one operand (lhs)
                leaf_pool.append(expr)
            case Bop.NULL_LHS:
                unimportant = leaf_pool.popleft()
                important = leaf_pool.popleft()
                expr = EX(Bop.NULL_LHS, unimportant, important)
                leaf_pool.append(expr)
                nulls_left -= 1
            case Bop.NULL_RHS:
                unimportant = leaf_pool.popleft()
                important = leaf_pool.popleft()
                expr = EX(Bop.NULL_RHS, important, unimportant)
                leaf_pool.append(expr)
                nulls_left -= 1
            case op:
                lhs = leaf_pool.popleft()
                rhs = leaf_pool.popleft()

                expr = EX(op, lhs, rhs)
                leaf_pool.append(expr)
                complexity_left -= 2

    # Return the root of the randomly constructed formula
    return leaf_pool.popleft()

def insert_values(expr, vals):
    if isinstance(expr, LEAF):
        expr.val = vals[expr.name]
        return expr

    elif isinstance(expr, EX):
        return EX(expr.op, 
                  insert_values(expr.lhs, vals), 
                    insert_values(expr.rhs, vals))

    else:
        return expr


def insert_values_arr(expr, arr):
    vald = {str(i): arr[i] for i in range(len(arr))}
    return insert_values(expr, vald)



if __name__ == "__main__":
    randomformula = randomize(10)
    print_expr_tree(randomformula)
    print("\n\n\n\n")

    value_dict = {str(i): 2.0 for i in range(12)}

    populated = insert_values(randomformula, value_dict)
    print_expr_tree(populated)
    print("\n\n\n\n")
    print(causal_responsibility(populated, 1.0))