import numpy as np


def cheeseOrder(x):
    result = np.ones([x[0], x[1]])
    result[0:x[0]:2, 0:x[1]:2] = x[2]
    result[0:x[0]:2, 1:x[1]:2] = x[3]
    result[1:x[0]:2, 0:x[1]:2] = x[3]
    result[1:x[0]:2, 1:x[1]:2] = x[2]
    return result


x = np.fromfile('data.csv', dtype='int', sep=';')
result1 = cheeseOrder(x)
np.savetxt('result.csv', result1, fmt="%d", delimiter=',')
