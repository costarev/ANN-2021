import numpy as np


# m - number of rows
# n - number of columns
# a - first chess number
# b - second chess number
def chess_board(m, n, a, b):
    matrix = np.ones((m, n), dtype=int) * a
    matrix[1::2, ::2] = b
    matrix[::2, 1::2] = b
    return matrix


try:
    data = np.fromfile("data.csv", dtype="int", sep=" ")
    np.savetxt("result.csv", chess_board(data[0], data[1], data[2], data[3]), delimiter=" ", fmt="%d")
except IndexError:
    print("error in input file: wrong data, must be 4 integers separated by space: m n a b")
