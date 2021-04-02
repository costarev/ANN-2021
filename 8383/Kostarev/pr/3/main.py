import numpy as np

def get_dot_sum(m, v):
    if len(m) != len(v): raise IndexError()
    v_shape = v.shape[1]
    dot_sum = np.zeros([v_shape, 1], dtype=int)
    for i in range(len(m)):
        dot_sum = dot_sum + m[i].dot(v[i])
    if dot_sum.shape[1] != 1: raise ValueError()
    return dot_sum

try:
    data = np.genfromtxt('input.csv', dtype='int', delimiter=',')
    m = np.reshape(data[:len(data) - 3], [3, 4, 4])
    v = np.reshape(data[len(data) - 3:], [3, 4, 1])
    dot_sum = get_dot_sum(m, v)
    np.savetxt('output.csv', dot_sum, fmt='%d', delimiter=',')
except IndexError:
    print('Количество векторов и матриц неодинаковая или их размерность неверна!')
except ValueError:
    print('Размерность векторов или матриц неверна!')