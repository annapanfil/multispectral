import operator
import numpy as np

def normalised_difference(a, b):
    res = (a-b) / (a+b + EPSILON)
    return res

CHANNELS = {'B': 0, 'G': 1, 'R': 2, 'N': 3, 'E': 4} # N - near infrared, E - red edge
CHANNEL_NAMES = ["B", "G", "R", "N", "E"]

NAMES_CONVERSION = {"RE": "E", "NIR": "N"}
EPSILON = 1e-10 # not to divide by zero


OPERATIONS = {"+": operator.add, 
            "-": operator.sub, 
            "*": operator.mul, 
            "/": lambda a, b: a / b if b.all() != 0 else a / (b + EPSILON),
            "#": normalised_difference}

CAM_HFOV = np.deg2rad(49.6)  # rad
CAM_VFOV = np.deg2rad(38.3)  # rad

CAMERA_MATRIX = [[     1575.3,           0,      714.41],
                 [          0,      1575.3,      471.71],
                 [          0,           0,           1]] # for ch 0

DISTORTION_COEFFS = [   -0.08933,     0.11349, -0.00045258,  0.00056019,   0.0085821] # k1, k2, p1, p2, k3