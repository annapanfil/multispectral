from dataclasses import dataclass
import math
from typing import Dict, Union, List
import numpy as np
import random

# PERMUTATIONS
def _prune(tree, constants):
    """Randomly remove one node (not leaf) from the tree and replace it with a value."""
    op_nodes = [i for i, node in enumerate(tree) if isinstance(node, str)]

    if len(op_nodes) == 0: return False
    
    # replace operator with a value
    node_to_change = random.choice(op_nodes)
    tree[node_to_change] = random.choice(constants.VALUES)
    make_not_trivial(tree, parent(node_to_change), constants.FORBIDEN_IDENTICAL, constants.VALUES)

    # prune whole tree
    l = left_child(node_to_change)
    r = right_child(node_to_change)
    size = 1
    while l < len(tree):
        tree[l:l+size] = [None] * size
        tree[r:r+size] = [None] * size
        l = left_child(l)
        r = left_child(r)
        size *= 2

    return True


def _add(tree, constants):
    """Replace a leaf node with random operation"""
    # Randomly decide whether to go to the left or right node

    leaf_nodes = [i for i, node in enumerate(tree) if not isinstance(node, str) and not node is None]
    node_to_change = random.choice(leaf_nodes)

    # add operation
    tree[node_to_change] = random.choice(list(constants.OPERATIONS.keys()))
    
    # add operands 
    left_ch = left_child(node_to_change)
    right_ch = right_child(node_to_change)

    # make the list bigger if necessary
    if right_ch > len(tree):
        for _ in range(2**(depth(tree)+1)): tree.append(None)

    tree[left_ch] = random.choice(constants.VALUES)
    tree[right_ch] = random.choice(constants.VALUES)

    make_not_trivial(tree, node_to_change, constants.FORBIDEN_IDENTICAL, constants.VALUES)

    return True

def _swap(tree, constants):
    """Randomly remove one node (not leaf) from the tree and replace it with a value. 
    Not possible if the tree has only one operation (depth == 1)."""
    op_nodes = [i for i, node in enumerate(tree) if isinstance(node, str)]
    if len(op_nodes) == 0:
        return False

    node_to_change = random.choice(op_nodes)

    while tree[node_to_change] in constants.ALTERNATE_OPERATORS:
        op_nodes.remove(node_to_change)
        if len(op_nodes) == 0: return False
        node_to_change = random.choice(op_nodes)

    # swap subtrees
    l = left_child(node_to_change)
    r = right_child(node_to_change)
    size = 1
    while l < len(tree):
        tree[l:l+size], tree[r:r+size] = tree[r:r+size], tree[l:l+size]
        l = left_child(l)
        r = left_child(r)
        size *= 2
    
    return True
    
def _change_op(tree, constants):
    """Randomly change the operation of a node."""
    op_nodes = [i for i, node in enumerate(tree) if isinstance(node, str)]
    if len(op_nodes) == 0: return False

    node_to_change = random.choice(op_nodes)
    tree[node_to_change] = random.choice([op for op in constants.OPERATIONS.keys() if op != tree[node_to_change]])
    make_not_trivial(tree, node_to_change, constants.FORBIDEN_IDENTICAL, constants.VALUES)
    
    return True
    

def _change_value(tree, constants):
    """Replace a leaf node with random value"""

    leaf_nodes = [i for i, node in enumerate(tree) if not isinstance(node, str) and not node is None]
    node_to_change = random.choice(leaf_nodes)
    allowed_values = [v for v in constants.VALUES if v != tree[node_to_change]]

    tree[node_to_change] = random.choice(allowed_values)
    make_not_trivial(tree, parent(node_to_change), constants.FORBIDEN_IDENTICAL, allowed_values)
       
    return True


def permute(tree, constants):
    """Randomly permutes the tree."""
    permutations = [_prune, _change_op, _swap, _add, _change_value]

    permutation = random.choice(permutations)

    print(permutation.__name__[1:])
    permuted = permutation(tree, constants)

    while not permuted:
        if permutation == _swap: print("Cannot swap if all the operations are alternate or the tree has no operation")
        elif permutation == _prune: print("Cannot prune tree with one node only")
        elif permutation == _change_op: print("Cannot change operator if there's no operator in the tree")

        permutations.remove(permutation)
        if len(permutations) == 0:
            print("Cannot permute")
            return False
        permutation = random.choice(permutations)

        print(permutation.__name__[1:])
        permuted = permutation(tree, constants)
    
    return True


# GETING THE TREE AND RESULT
def is_leaf(tree: list, node: int):
    left_child_idx = 2 * node + 1
    right_child_idx = 2 * node + 2
    return left_child_idx >= len(tree) or (tree[left_child_idx] is None and tree[right_child_idx] is None)

def left_child(node: int) -> int:
    return 2 * node + 1

def right_child(node: int) -> int:
    return 2 * node + 2

def parent(node: int) -> int:
    return math.floor((node - 1) / 2)

def formula_str(tree: list, index: int = 0) -> str:
    """Returns a string representation of the formula node with operation symbols."""    
    if is_leaf(tree, index):
        return str(tree[index])
    
    # if is a node, recursively get children strings
    left_expr = formula_str(tree, left_child(index))
    right_expr = formula_str(tree, right_child(index))
    return f"({left_expr} {tree[index]} {right_expr})"


def evaluate(operations: dict, tree: list, node: int = 0) -> Union[np.array, float]:
    """Recursively evaluates the tree."""
    if node == 0 and is_leaf(tree, node):
        return tree[node]

    left_result = tree[left_child(node)] if is_leaf(tree, left_child(node)) else evaluate(operations, tree, left_child(node))
    right_result = tree[right_child(node)] if is_leaf(tree, right_child(node)) else evaluate(operations, tree, right_child(node))
    return operations[tree[node]](left_result, right_result)


def depth(tree: list) -> int:
    """Calculates the depth of a binary tree represented as an array."""
    last_index = max((i for i, value in enumerate(tree) if value is not None), default=-1)
    if last_index == -1:
        return 0  # Tree is empty
    
    return math.floor(math.log2(last_index + 1))


def make_not_trivial(tree: list, index:int, forbidden_identical: List[str], allowed_values: List[Union[float, np.array]]) -> None:
    """Ensure that operations on integers are not trivial (e.g. x-x or x/x)"""
    left_ch = left_child(index)
    right_ch = right_child(index)

    # none of them is operand and they are the same for the forbidden_identical operation
    if not (isinstance(tree[left_ch], str) or isinstance(tree[right_ch], str)) and\
       tree[index] in forbidden_identical and tree[left_ch] == tree[right_ch]:
            tree[left_ch] = random.choice([v for v in allowed_values if v !=  tree[left_ch]])
            print(f"changed {tree[right_ch]} to {tree[left_ch]} because they were identical for operator {tree[index]}")


# GENERATING THE FORMULA
def random_exp(max_val: float):
    """Choose a random number from 0 to max_val with exponentially higher probability closer to max_val"""
    if max_val == 0: return 0

    values = np.arange(max_val+1)
    probabilities = np.exp(values / max_val)
    probabilities /= probabilities.sum()  # Normalizacja
    return np.random.choice(values, p=probabilities)

def generate_random_formula(constants:list, depth: int = 10, formula: list=None, index: int=0) -> list:
    """Recursively generates a random formula tree with required depth"""
    if formula is None:
        formula = [None] if depth == 0 else [None] * (2 ** (depth+1) - 1)

    if depth == 0:
        # Return a leaf node
        formula[index] = random.choice(constants.VALUES)
        return formula

    # Operation node
    formula[index] = random.choice(list(constants.OPERATIONS.keys()))

    #  ensure one branch has required depth
    depths = [random_exp(depth-1), depth-1]
    random.shuffle(depths)

    generate_random_formula(constants, depths[0], formula, index=left_child(index))
    generate_random_formula(constants, depths[1], formula, index=right_child(index))

    make_not_trivial(formula, index, constants.FORBIDEN_IDENTICAL , constants.VALUES)
    return formula



@dataclass(frozen=True)
class Constants:
    VALUES: List[float]
    OPERATIONS: Dict[str, callable]
    FORBIDEN_IDENTICAL: List[str]
    ALTERNATE_OPERATORS: List[str]

if __name__ == "__main__":

    # Define the constants
    constants = Constants(
        VALUES=[2.0, 3.0, 4.0, 5.0, 6.0],
        OPERATIONS={
            "+": np.add,
            "-": np.subtract,
            "*": np.multiply,
            "/": np.divide,
            "#": lambda A, B: np.divide((A - B), (A + B)),
        },
        FORBIDEN_IDENTICAL=['-', '/', '#'],
        ALTERNATE_OPERATORS=['+', '*']
    )

    formula = generate_random_formula(constants, depth=3)

    # formula = ['+', 3, "*", None, None, 4, 2] # 3-(4*2)
    # formula = ['#', "+", "*", "/", "-", "+", "-", 3, 2, 4, 6, 6, 3, 5, 4]
    
    # evaluate formula
    print(f"{formula_str(formula)} = {evaluate(constants.OPERATIONS, formula):.2f}, depth {depth(formula)}")

    for _ in range(30):
        perm = permute(formula, constants)
        if not perm: break
        print(f"{formula_str(formula)} = {evaluate(constants.OPERATIONS, formula):.2f}")
