from typing import Tuple, Union, Callable, List
import numpy as np
import random



class FormulaNode:
    # Available operations and their symbols
    
    OPERATIONS = [       
            (np.add, '+'),
            (np.subtract, '-'),
            (np.multiply, '*'),
            (np.divide, '/')
            #   (lambda A, B: (A - B) / (A + B), "normdiff")] # TODO: make safe
        ] 
    
    ALTERNATE_OPERATORS = ['+', '*']
    FORBIDEN_IDENTICAL = ['-', '/']
    
    VALUES = [2.0, 3.0, 4.0, 5.0, 6.0]


    def __init__(self, operation: Tuple[Callable[[np.array, np.array], np.array], str] = None, 
                 left: Union['FormulaNode', np.array] = None, 
                 right: Union['FormulaNode', np.array] = None
                ):
        """
        Represents a node in the operation tree.
        :param operation: A function that takes two arguments (left and right) and returns the result. And a symbol representing the operation (e.g., '+', '-', '*').
        :param left: Left operand (FormulaNode or array).
        :param right: Right operand (FormulaNode or array).
        """
        changing = left is None or right is None

        self.operation, self.symbol = operation if operation is not None else random.choice(self.OPERATIONS)
        self.left = left if left is not None else random.choice(self.VALUES)
        self.right = right if right is not None else random.choice(self.VALUES)

        if changing: self.make_not_trivial(self.VALUES) # if the user makes trivial operation then we don't change anything
        
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
        left_result = self.left.evaluate() if isinstance(self.left, FormulaNode) else self.left
        right_result = self.right.evaluate() if isinstance(self.right, FormulaNode) else self.right
        return self.operation(left_result, right_result)
    
    def __call__(self):
        return self.evaluate()

    def make_not_trivial(self, allowed_values: list):
        """Ensure that operations on integers are not trivial (e.g. x-x or x/x)"""
        if isinstance(self.left, FormulaNode) and isinstance(self.right, FormulaNode):
            return

        if self.symbol in self.FORBIDEN_IDENTICAL and self.left == self.right:
            self.left = random.choice([v for v in allowed_values if v != self.left])
            print(f"changed {self.right} to {self.left} because the operands were identical for {self.symbol}")
        

    def __repr__(self):
        """Returns a string representation of the formula node with operation symbols."""
        if self.operation is None:
            return f"{self.left}"  # Leaf node with a matrix or scalar
        left_str = str(self.left) if isinstance(self.left, FormulaNode) else f"{self.left}"
        right_str = str(self.right) if isinstance(self.right, FormulaNode) else f"{self.right}"
        return f"({left_str} {self.symbol} {right_str})"
    
    ## PERMUTATIONS
    def _prune_here(self, children_are_nodes: Tuple[bool, bool]):
        if sum(children_are_nodes) == 0:
            print("ERROR: No children")
            return False

        if sum(children_are_nodes) == 2:
            if random.choice([True, False]): # cut off left or right
                self.left = random.choice(self.VALUES)
            else:
                self.right = random.choice(self.VALUES)
        elif children_are_nodes[0]: # only left child is a node, so cut this one off
            self.left = random.choice(self.VALUES)
        else:  # only right child is a node, so cut this one off
            self.right = random.choice(self.VALUES)
        return True

    def _prune(self):
        """Randomly remove one node (not leaf) from the tree and replace it with a value. 
        Not possible if the tree has only one operation (depth == 1)."""
        children_are_nodes = (isinstance(self.left, FormulaNode), isinstance(self.right, FormulaNode))

        # do not cut the leaves
        if sum(children_are_nodes) == 0:
           return False
        
        if random.choice([True, False]):
            return self._prune_here(children_are_nodes)

        # If not pruned, recursively permute left and right subtrees if they exist
        if isinstance(self.left, FormulaNode):
            prune =  self.left._prune()
        if isinstance(self.right, FormulaNode):
            prune = self.right._prune()

        # do prunning on final node if not pruned earlier
        if not prune:
            return self._prune_here(children_are_nodes)
        else: return True


    def _add(self):
        """Replace a leaf node with random operation"""
        # Randomly decide whether to go to the left or right node

        target = 'left' if random.choice([True, False]) else 'right'
        next_node = getattr(self, target)

        if not isinstance(next_node, FormulaNode):
            # We've reached a leaf node, so we replace it with a random operation
            new_node = FormulaNode()
            setattr(self, target, new_node)
        else:
            # Continue the recursion
            next_node._add()
        
        return True

    def _swap(self):
        """Randomly remove one node (not leaf) from the tree and replace it with a value. 
        Not possible if the tree has only one operation (depth == 1)."""
        children_are_nodes = (isinstance(self.left, FormulaNode), isinstance(self.right, FormulaNode))

        # if we have no other node to go to, swap leaves
        if sum(children_are_nodes) == 0:
            if self.symbol in self.ALTERNATE_OPERATORS: # pointless to swap
                return False
            else:
                self.left, self.right = self.right, self.left
                return True 
        
        # decide if to swap here
        if self.symbol not in self.ALTERNATE_OPERATORS:
            if random.choice([True, False]): 
                self.left, self.right = self.right, self.left
                return True
        
        # Go to child node
        if sum(children_are_nodes) == 2:  # Both children are nodes
            next_node = random.choice([self.left, self.right])
            permuted = next_node._swap()
            if not permuted: # if didn't permute there, check another path
                next_node = self.left if next_node == self.right else self.right
                permuted = next_node._swap()
        elif children_are_nodes[0]:  # Only the left child is a node
            permuted = self.left._swap()
        elif children_are_nodes[1]:  # Only the right child is a node
            permuted = self.right._swap()

        if not permuted:
            # swap if possible
            if self.symbol in self.ALTERNATE_OPERATORS: # pointless to swap
                return False
            else:
                self.left, self.right = self.right, self.left
                return True 

        return True

    def _change_op(self):
        """Randomly change the operation of a node."""
        children_are_nodes = (isinstance(self.left, FormulaNode), isinstance(self.right, FormulaNode))
        
        # decide if to change here
        if sum(children_are_nodes) == 0 or random.choice([True, False]):
            self.operation, self.symbol = random.choice([op for op in self.OPERATIONS if op != (self.operation, self.symbol)])
            self.make_not_trivial(self.VALUES)
            return True
            
        # Go to child node
        if sum(children_are_nodes) == 2:  # Both children are nodes
            permuted = random.choice([self.left, self.right])._change_op()
        elif children_are_nodes[0]:  # Only the left child is a node
            permuted = self.left._change_op()
        elif children_are_nodes[1]:  # Only the right child is a node
            permuted = self.right._change_op()

        if not permuted:
            # change here
            self.operation, self.symbol = random.choice([op for op in self.OPERATIONS if op != (self.operation, self.symbol)])
            self.make_not_trivial(self.VALUES)

        return True
        
    def _change_value(self):
        """Replace a leaf node with random value"""

        # Randomly decide whether to go to the left or right node
        target = 'left' if random.choice([True, False]) else 'right'
        next_node = getattr(self, target)

        if not isinstance(next_node, FormulaNode):
            # We've reached a leaf node, so we replace it
            
            allowed_values = [v for v in self.VALUES if v != next_node]
            new_value = random.choice(allowed_values)
            self.make_not_trivial(allowed_values)
            setattr(self, target, new_value)
        else:
            # Continue the recursion
            next_node._change_value()
        
        return True



    def permute(self):
        """Randomly permutes the tree."""
        permutations = [self._prune, self._add, self._swap, self._change_op, self._change_value]
    
        permutation = random.choice(permutations)

        permuted = False

        while not permuted:
            permuted = permutation()
            print(permutation.__name__[1:])
            if not permuted:
                if permutation == self._swap: print("Cannot swap if all the operations are alternate")
                elif permutation == self._prune: print("Cannot prune tree with one node only")
            permutations.remove(permutation)
            permutation = random.choice(permutations)
            

def random_exp(max_val: float):
    """Choose a random number from 0 to max_val with exponentially higher probability closer to max_val"""
    if max_val == 0: return 0

    values = np.arange(max_val+1)
    probabilities = np.exp(values / max_val)
    probabilities /= probabilities.sum()  # Normalizacja
    return np.random.choice(values, p=probabilities)

def generate_random_formula(values: list, operations: list, depth: int = 10) -> FormulaNode:
    """Recursively generates a random formula tree with required depth"""
    if depth == 0:
        # Return a leaf node
        return random.choice(values)

    #  ensure one branch has required depth
    depths = [random_exp(depth-1), depth-1]
    random.shuffle(depths)

    left_node = generate_random_formula(values, operations, depths[0])
    right_node = generate_random_formula(values, operations, depths[1])

    node = FormulaNode(random.choice(operations), left=left_node, right=right_node)
    node.make_not_trivial(FormulaNode.VALUES)
    return node



root = generate_random_formula(FormulaNode.VALUES, FormulaNode.OPERATIONS, depth=3)

# (5 + 3) / (5 - (4 * 2))
# node1 = FormulaNode((np.add, '+'), 5, 3)  # (5 + 3)
# node2 = FormulaNode((np.multiply, '-'), 4, 2) # (4 * 2)
# node3 = FormulaNode((np.subtract, '+'), 5, node2)  # (5 - (4 * 2))
# root = FormulaNode((np.divide, '+'), node1, node3)

# Evaluate the tree
print(f"{root} = {root():.2f} d={root.depth}")

for _ in range(5):
    root.permute()
    print(f"{root} = {root():.2f}")


