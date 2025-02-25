import random
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../libraries/imageprocessing')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import micasense.imageutils as imageutils
from src.processing.evaluate_index import evaluate_postfix, infix_to_postfix, get_custom_index
from src.processing.consts import EPSILON

def test_infix_to_postfix():
    assert infix_to_postfix("A+B") == "BA+"
    assert infix_to_postfix("(A+B)*C") == "CBA+*"
    assert infix_to_postfix("A+(B*C)") == "CB*A+"
    assert infix_to_postfix("((A+B)*C)") == "CBA+*"
    assert infix_to_postfix("(A*(B+C))") == "CB+A*"
    assert infix_to_postfix("((A+B)*(C+D))") == "DC+BA+*"
    assert infix_to_postfix("A#B") == "BA#"
    assert infix_to_postfix("(((A+B)#(C+D))+B)") == "BDC+BA+#+"
    assert infix_to_postfix("A-B") == "BA-"

def test_evaluate_postfix_integers():
    variables = {"A": 3, "B": 4, "C": 2, "D": 5}
    assert evaluate_postfix("AB+", variables) == 7
    assert evaluate_postfix("ABC*+", variables) == 11
    assert evaluate_postfix("AB*C+", variables) == 14
    assert evaluate_postfix("AB*C+D*", variables) == 70
    assert evaluate_postfix("AB-", variables) == 4-3
    assert np.isclose(evaluate_postfix("AB#", variables), (4 - 3) / (4 + 3))


def test_evaluate_postfix_np_arrays():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.array([[2, 2], [2, 2]])
    variables = {"A": A, "B": B, "C": C}

    np.testing.assert_array_equal(evaluate_postfix("AB+", variables), A + B)
    np.testing.assert_array_equal(evaluate_postfix("ABC*+", variables), A + B * C)
    np.testing.assert_array_equal(evaluate_postfix("AB*C+", variables), A * B + C)
    np.testing.assert_array_equal(evaluate_postfix("AB-", variables), B - A)
    np.testing.assert_array_almost_equal(evaluate_postfix("AB#", variables), (B - A) / (B + A))
    np.testing.assert_array_almost_equal(evaluate_postfix("AB/", variables), B / A)


def test_get_custom_index():
    np.random.seed(42)
    img_aligned = np.random.rand(10, 10, 5)  # Random values for testing
    for i in range(img_aligned.shape[2]):
        img_aligned[:,:,i] =  imageutils.normalize(img_aligned[:,:,i])

    b = img_aligned[:, :, 0]
    g = img_aligned[:, :, 1]
    r = img_aligned[:, :, 2]
    n = img_aligned[:, :, 3]
    e = img_aligned[:, :, 4]

    np.testing.assert_array_almost_equal(get_custom_index("R", img_aligned), (r - np.min(r)) / (np.max(r) - np.min(r)))

    expected = r * g + b
    expected = np.where(np.isnan(expected), 1, expected) # replace division by zero by the highest value
    expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    result = get_custom_index("R*G+B", img_aligned)
    np.testing.assert_array_almost_equal(result, expected,2)

    expected = r / (g + b)
    expected = np.where(np.isnan(expected), 1, expected) # replace division by zero by the highest value
    expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    result = get_custom_index("R/(G+B)", img_aligned)
    np.testing.assert_array_almost_equal(result, expected, 2)

    expected = g - e
    expected = np.where(np.isnan(expected), 1, expected) # replace division by zero by the highest value
    expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    result = get_custom_index("G-E", img_aligned)
    np.testing.assert_array_almost_equal(result, expected)

    expected = (g - n) / (g + n + EPSILON)
    expected = np.where(np.isnan(expected), 1, expected) # replace division by zero by the highest value
    expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    result = get_custom_index("G#N", img_aligned)
    np.testing.assert_array_almost_equal(result, expected)

    sm_expr = g / (r * b + EPSILON)
    expr = (g / (e + EPSILON)) + g + ((r - sm_expr) / (r + sm_expr + EPSILON))
    expected = ((g-e)/(g + e + EPSILON)) * ((n - expr)/(n + expr + EPSILON))
    expected = np.where(np.isnan(expected), 1, expected) # replace division by zero by the highest value
    expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    result = get_custom_index("((G # E) * (N # (((G / E) + (R # (G / (R * B)))) + G)))", img_aligned)
    np.testing.assert_array_almost_equal(result, expected, 4)