import numpy as np

input_arr = np.fromfile('input_pr3.txt', dtype=int, sep=' ')

M = input_arr[0]
N = input_arr[1]
a = input_arr[2]
b = input_arr[3]

z = np.ones((M, N)) * a
z[1::2, ::2] = b
z[::2, 1::2] = b

np.savetxt('output_pr3.txt', z, fmt="%d", delimiter=' ')
