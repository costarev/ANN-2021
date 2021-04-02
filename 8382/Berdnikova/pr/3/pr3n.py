import numpy as np


def make_dot(matrix, vectors):
    if len(matrix) != len(vectors):
        print("Количество матриц и векторов не сопадает")
        return 0
    v_shape = vectors.shape[1]
    total = np.zeros([v_shape, 1], dtype=int)
    for i in range(len(matrix)):
        total = total + matrix[i].dot(vectors[i])
    return total


try:
    data = np.genfromtxt('input.csv', dtype='int', delimiter=',')
except IndexError:
    print("Ошибка во входных данных")

matrix = np.reshape(data[:len(data) - 3], [3, 4, 4])
vectors = np.reshape(data[len(data) - 3:], [3, 4, 1])
answer = make_dot(matrix, vectors)
print(answer)
np.savetxt('output.csv', answer, fmt='%d', delimiter=',')
