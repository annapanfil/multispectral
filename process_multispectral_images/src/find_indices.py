from typing import Union, Callable, List
import numpy as np
import random

class FormulaNode:
    def __init__(self, operation: Callable[[np.array, np.array], np.array] = None, 
                 symbol: str = None, 
                 left: Union['FormulaNode', np.array] = None, 
                 right: Union['FormulaNode', np.array] = None):
        """
        Represents a node in the operation tree.
        :param operation: A function that takes two arguments (left and right) and returns the result.
        :param symbol: The symbol representing the operation (e.g., '+', '-', '*').
        :param left: Left operand (FormulaNode or array).
        :param right: Right operand (FormulaNode or array).
        """
        self.operation = operation
        self.symbol = symbol 
        self.left = left
        self.right = right

    def evaluate(self) -> np.array:
        """Recursively evaluates the tree."""
        if self.operation is None:
            return self.left  # Leaf node with a matrix or number
        left_result = self.left.evaluate() if isinstance(self.left, FormulaNode) else self.left
        right_result = self.right.evaluate() if isinstance(self.right, FormulaNode) else self.right
        return self.operation(left_result, right_result)
    
    def __str__(self):
        """Returns a string representation of the formula node with operation symbols."""
        if self.operation is None:
            return f"{self.left}"  # Leaf node with a matrix or scalar
        left_str = str(self.left) if isinstance(self.left, FormulaNode) else f"{self.left}"
        right_str = str(self.right) if isinstance(self.right, FormulaNode) else f"{self.right}"
        return f"({left_str} {self.symbol} {right_str})"


def generate_random_formula(values: list, operations: list, depth: int = 3) -> FormulaNode:
    """Recursively generates a random formula tree."""
    if depth == 0:
        # Return a leaf node
        return random.choice(values)

    # Randomly choose an operation and generate subtrees
    operation, symbol = random.choice(operations)
    
    # Recursively generate the left and right subtrees
    left_node = generate_random_formula(values, operations, random.randint(0, depth - 1))
    right_node = generate_random_formula(values, operations, random.randint(0, depth - 1))
    
    return FormulaNode(operation=operation, symbol=symbol, left=left_node, right=right_node)




# Create tree nodes
# A = np.array([[1, 2], [3, 4]], dtype=float)
# B = np.array([[5, 6], [7, 8]], dtype=float)
# C = np.array([[9, 10], [11, 12]], dtype=float)
# D = np.array([[13, 14], [15, 16]], dtype=float)
a,b,c,d,e = 2.0, 3.0, 4.0, 5.0, 6.0
channels = [a,b,c,d,e]

# Available operations and their symbols
operations = [
    (np.add, '+'),
    (np.subtract, '-'),
    (np.multiply, '*'),
    (np.divide, '/')
    #   (lambda A, B: (A - B) / (A + B), "normdiff")] # TODO: make safe
]

random_tree = generate_random_formula(channels, operations, depth=3)  # Depth 3 for complexity

# Evaluate the tree
print(random_tree)
print(f"Result: {random_tree.evaluate():.2f}")

