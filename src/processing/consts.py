import operator
import numpy as np

def normalised_difference(a, b):
    res = (a-b) / (a+b + EPSILON)
    return res

CHANNELS = {'B': 0, 'G': 1, 'R': 2, 'N': 3, 'E': 4} # N - near infrared, E - red edge
"""Mapping of channel names to their indices."""

CHANNEL_NAMES = ["B", "G", "R", "N", "E"]
"""List of channel names in the order they are used in the MicaSense RedEdge camera."""

NAMES_CONVERSION = {"RE": "E", "NIR": "N"}
"""Mapping channel names with different abbreviations to the one letter ones."""

EPSILON = 1e-10 # not to divide by zero

OPERATIONS = {"+": operator.add, 
            "-": operator.sub, 
            "*": operator.mul, 
            "/": lambda a, b: a / b if b.all() != 0 else a / (b + EPSILON),
            "#": normalised_difference}
"""Operations that can be used in the formula (dict)."""


CAM_HFOV = np.deg2rad(49.6)  # rad
"""Camera horizontal field of view in radians."""

CAM_VFOV = np.deg2rad(38.3)  
"""Camera vertical field of view in radians."""

CAMERA_MATRIX = [[     1575.3,           0,      714.41],
                 [          0,      1575.3,      471.71],
                 [          0,           0,           1]]
"""Camera matrix for the MicaSense RedEdge camera (for channel 0)."""

DISTORTION_COEFFS = [   -0.08933,     0.11349, -0.00045258,  0.00056019,   0.0085821] 
"""Distortion coefficients for the MicaSense RedEdge camera (for channel 0). In order: k1, k2, p1, p2, k3"""
