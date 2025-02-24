import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../libraries/imageprocessing')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from src.processing.evaluate_index import evaluate_postfix, infix_to_postfix, get_custom_index
from src.processing.consts import EPSILON, CHANNELS

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
    img_aligned = np.random.rand(5, 5, 5)  # Random values for testing

    result = get_custom_index("R*G+B", img_aligned)
    expected = img_aligned[:, :, 2] * img_aligned[:, :, 1] + img_aligned[:, :, 0]
    expected = np.where(np.isnan(expected), 1, expected) # replace division by zero by the highest value
    expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    np.testing.assert_array_almost_equal(result, expected)

    result = get_custom_index("R/(G+B)", img_aligned)
    expected = img_aligned[:, :, 2] / (img_aligned[:, :, 1] + img_aligned[:, :, 0])
    expected = np.where(np.isnan(expected), 1, expected) # replace division by zero by the highest value
    expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    np.testing.assert_array_almost_equal(result, expected)

    result = get_custom_index("G-E", img_aligned)
    expected = img_aligned[:, :, 1] - img_aligned[:, :, 4]
    expected = np.where(np.isnan(expected), 1, expected) # replace division by zero by the highest value
    expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    np.testing.assert_array_almost_equal(result, expected)

    result = get_custom_index("G#N", img_aligned)
    expected = (img_aligned[:, :, 1] - img_aligned[:, :, 3]) / (img_aligned[:, :, 1] + img_aligned[:, :, 3] + EPSILON)
    expected = np.where(np.isnan(expected), 1, expected) # replace division by zero by the highest value
    expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    np.testing.assert_array_almost_equal(result, expected)
