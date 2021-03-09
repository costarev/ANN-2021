import numpy as np


def trim_matrix(matr, max_el):
    cols = int(np.ceil((len(np.base_repr(max_el)) + 1) / 8))
    trimmed_matr = np.delete(matr, np.arange(matr.shape[1] - cols), axis=1)
    return trimmed_matr


def vec_to_bin():
    x = np.fromfile('input.csv', dtype='int', sep=';')
    x_uint8 = np.flip(x.reshape([x.shape[0], 1]).view("uint8"), axis=1)
    # следующая строчка нужна исключительно для того, чтобы было удобнее читать результат
    x_uint8 = trim_matrix(x_uint8, max(np.abs(x)))
    y = np.unpackbits(x_uint8, axis=1)
    np.savetxt('output.csv', y, fmt='%d', delimiter=';', newline='\n')
    return y


res = vec_to_bin()
print(res)
