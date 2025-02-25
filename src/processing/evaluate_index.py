import numpy as np
import micasense.imageutils as imageutils
from processing.consts import OPERATIONS, CHANNELS

def evaluate_postfix(expression: str, variables: dict) -> np.array:
    """
    Evaluate a prefix expression.
    Args:
        expression (str): The prefix expression (e.g., "#RG").
        variables (dict): A dictionary of variable values.
    Returns:
        The result of the expression.
    """
    stack = []

    for token in expression:
        if token in "+-*/#":
            a = stack.pop()
            b = stack.pop()
            if type(a) == str and a in variables: a = variables[a]
            if type(b) == str and b in variables: b = variables[b]
            stack.append(OPERATIONS[token](a, b))

        elif token in variables:
            stack.append(token)
        else:
            stack.append(int(token))  # Handle constants

    return stack[0]

def is_operator(c):
    """
    Check if the character is an operator – any character that is neither an alphabet letter nor a digit.
    Parameters:
    c (str): The character to check.
    Returns:
    bool: True if the character is an operator, False otherwise.
    """
    
    return (not c.isalpha()) and (not c.isdigit())

def get_priority(c):
    # Get the priority of operators
    if c == '-' or c == '+':
        return 1
    elif c == '*' or c == '/' or c == "#":
        return 2
    return 0

def infix_to_postfix(infix):
    """
    Convert an infix expression to a postfix expression.
    This function takes an infix expression (a mathematical notation where operators are placed between operands)
    and converts it to a postfix expression (also known as Reverse Polish Notation, where operators follow their operands).
    Args:
        infix (str): The infix expression to be converted.
    Returns:
        str: The resulting postfix expression.
    Example:
        >>> infix_to_postfix("A*(B+C)/D")
        'ABC+*D/'
    Note:
        This function assumes that the input infix expression is valid and contains only single-letter variables,
        digits, and the operators +, -, *, /, and ^. Parentheses are also supported for grouping.
    """
    infix = infix[::-1]
    reversed_infix = []

    for i in infix:
        if i == '(':
            reversed_infix.append(')')
        elif i == ')':
            reversed_infix.append('(')
        else:
            reversed_infix.append(i)
            
    reversed_infix = ''.join(reversed_infix)

    reversed_infix = '(' + reversed_infix + ')'
    l = len(reversed_infix)
    char_stack = []
    output = ""

    for i in range(l):
        
        # Check if the character is alphabet or digit
        if reversed_infix[i].isalpha() or reversed_infix[i].isdigit():
            output += reversed_infix[i]
            
        # If the character is '(' push it in the stack
        elif reversed_infix[i] == '(':
            char_stack.append(reversed_infix[i])
        
        # If the character is ')' pop from the stack
        elif reversed_infix[i] == ')':
            while char_stack[-1] != '(':
                output += char_stack.pop()
            char_stack.pop()
        
        # Found an operator
        else:
            if is_operator(char_stack[-1]):
                if reversed_infix[i] == '^':
                    while get_priority(reversed_infix[i]) <= get_priority(char_stack[-1]):
                        output += char_stack.pop()
                else:
                    while get_priority(reversed_infix[i]) < get_priority(char_stack[-1]):
                        output += char_stack.pop()
                char_stack.append(reversed_infix[i])

    while len(char_stack) != 0:
        output += char_stack.pop()
    return output


def get_custom_index(formula: str, img_aligned: np.array, norm_to_255=False) -> np.array:
    """Get a custom index from an image using your formula.
    Args:
        formula (str): The formula to calculate the index.
        img_aligned (np.array): The image.
        norm_to_255 (bool): If True, before calculating the index normalize the channels to [0, 255], which sometimes works better for more complex formulas because of the numerical issues. (default: False)

    Returns:
        np.array: The calculated index.
    """

    img_aligned = img_aligned.copy()

    for i in range(img_aligned.shape[2]):
        img_aligned[:,:,i] =  imageutils.normalize(img_aligned[:,:,i])
        if norm_to_255:  img_aligned[:,:,i] *= 255

    allowed_vars = {"R": img_aligned[:, :, CHANNELS["R"]],
                    "G": img_aligned[:, :, CHANNELS["G"]],
                    "B": img_aligned[:, :, CHANNELS["B"]],
                    "E": img_aligned[:, :, CHANNELS["E"]],
                    "N": img_aligned[:, :, CHANNELS["N"]]}

    try:
        if "#" in formula:
            formula = infix_to_postfix(formula.replace(" ", ""))
            index = evaluate_postfix(formula, allowed_vars)
        else:
            index = eval(formula, {"__builtins__": None}, allowed_vars)  

        index = np.where(np.isnan(index), 1, index) # replace division by zero by the highest value
        index = (index - np.min(index)) / (np.max(index) - np.min(index))
        return index
    except Exception as e:
        raise("Error in formula:", e)