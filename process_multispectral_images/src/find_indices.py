from typing import Tuple, Union, Callable, List
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
        self.depth = self._calc_depth()

    def _calc_depth(self) -> int:
        """
        Calculates the depth of the tree recursively.
        The depth of the tree is the number of edges from the root to the deepest leaf.
        """
        # If the node is a leaf (no left or right child), depth is 0
        if not isinstance(self.left, FormulaNode) and not isinstance(self.right, FormulaNode):
            return 1
        
        # Otherwise, recursively calculate the depth of the left and right subtrees
        left_depth = 0
        right_depth = 0
        
        if isinstance(self.left, FormulaNode):
            left_depth = self.left._calc_depth()
        if isinstance(self.right, FormulaNode):
            right_depth = self.right._calc_depth()

        # The depth of the current node is the maximum of the left and right subtree depths + 1
        return max(left_depth, right_depth) + 1


    def evaluate(self) -> np.array:
        """Recursively evaluates the tree."""
        if self.operation is None:
            return self.left  # Leaf node with a matrix or number
        left_result = self.left.evaluate() if isinstance(self.left, FormulaNode) else self.left
        right_result = self.right.evaluate() if isinstance(self.right, FormulaNode) else self.right
        return self.operation(left_result, right_result)
    
    def __repr__(self):
        """Returns a string representation of the formula node with operation symbols."""
        if self.operation is None:
            return f"{self.left}"  # Leaf node with a matrix or scalar
        left_str = str(self.left) if isinstance(self.left, FormulaNode) else f"{self.left}"
        right_str = str(self.right) if isinstance(self.right, FormulaNode) else f"{self.right}"
        return f"({left_str} {self.symbol} {right_str})"
    

    def _prune_here(self, children_are_nodes: Tuple[bool, bool], values_for_removed_nodes: List):
        if sum(children_are_nodes) == 0:
            print("ERROR: No children")
            return False

        if sum(children_are_nodes) == 2:
            if random.choice([True, False]): # cut off left or right
                self.left = random.choice(values_for_removed_nodes)
            else:
                self.right = random.choice(values_for_removed_nodes)
        else: # only one child is a node, so cut this one off
            if children_are_nodes[0]:
                self.left = random.choice(values_for_removed_nodes)
            else:
                self.right = random.choice(values_for_removed_nodes)
        return True

    def _prune(self, values_for_removed_nodes=[None]):
        """Randomly remove one node (not leaf) from the tree and replace it with a value. 
        Not possible if the tree has only one operation (depth == 1)."""
        children_are_nodes = (isinstance(self.left, FormulaNode), isinstance(self.right, FormulaNode))

        # do not cut the leaves
        if sum(children_are_nodes) == 0:
           return False
        
        if random.choice([True, False]):
            return self._prune_here(children_are_nodes, values_for_removed_nodes)

        # If not pruned, recursively permute left and right subtrees if they exist
        if isinstance(self.left, FormulaNode):
            prune =  self.left._prune(values_for_removed_nodes)
        if isinstance(self.right, FormulaNode):
            prune = self.right._prune(values_for_removed_nodes)

        # do prunning on final node if not pruned earlier
        if not prune:
            return self._prune_here(children_are_nodes, values_for_removed_nodes)
        else: return True


    def _add(self, operations: List, values: List):
        """Replace a leaf node with random operation"""
        # Randomly decide whether to go to the left or right node

        target = 'left' if random.choice([True, False]) else 'right'
        next_node = getattr(self, target)

        if not isinstance(next_node, FormulaNode):
            # We've reached a leaf node, so we replace it
            operation, symbol = random.choice(operations)
            new_node = FormulaNode(operation=operation,
                                symbol=symbol,
                                left=random.choice(values),
                                right=random.choice(values))
            setattr(self, target, new_node)  # Update the actual left or right node
        else:
            # Continue the recursion
            next_node._add(operations, values)
        
        return True

    

    def permute(self, operations, channels):
        """Randomly permutes the tree."""

        # random.choice([self._prune, self._add, self._swap, self._change_op, self._change_channel])
        
        permuted = self._add(operations, channels)

        # permuted = self._prune(channels)
        # if not permuted:
        #     print("Cannot prune tree with one node only")


def generate_random_formula(values: list, operations: list, depth: int = 10) -> FormulaNode:
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

# root = generate_random_formula(channels, operations, depth=3)  # Depth 3 for complexity


# (5 + 3) * (5 - (4 * 2))
node1 = FormulaNode(operation=np.add, symbol='+', left=5, right=3)  # (5 + 3)
node2 = FormulaNode(operation=np.multiply, symbol='*', left=4, right=2)  # (4 * 2)
node3 = FormulaNode(operation=np.subtract, symbol='-', left=5, right=node2)
root = FormulaNode(operation=np.multiply, symbol='*', left=node1, right=node3)


# Evaluate the tree
print(root, root.depth)
print(f"Result: {root.evaluate():.2f}")


for _ in range(5):
    root.permute(operations, channels)
    print(root, root._calc_depth())


