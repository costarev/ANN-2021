import numpy as np


def make_binary_matrix(array):
    return np.unpackbits(array[:, np.newaxis], axis=1)


input_array = np.fromfile("input.csv", dtype=np.uint8, sep=",")
np.savetxt("output.csv", make_binary_matrix(input_array), delimiter=",", fmt="%d")