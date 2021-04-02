import numpy as np


def pr3(matrix: np.ndarray, y, x, width, height):
    assert width > 0
    assert height > 0
    shape = matrix.shape
    col_start = y - width // 2
    col_end = y + width // 2 + width % 2
    row_start = x - height // 2
    row_end = x + height // 2 + height % 2
    assert (col_start >= 0)
    assert (row_start >= 0)
    assert (col_end <= shape[1])
    assert (row_end <= shape[0])
    return matrix[row_start:row_end, col_start:col_end]


if __name__ == '__main__':
    np.savetxt('out.csv',
               pr3(np.genfromtxt('data.csv', dtype='int', delimiter=';'), int(input('x = ')), int(input('y = ')),
                   int(input('width = ')), int(input('height = '))),
               fmt='%s;')
