import operator

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
            "/": operator.truediv,
            "#": normalised_difference}
        