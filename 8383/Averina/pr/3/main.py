import numpy as np


def get_multiplication(matrices, vectors):
    shape = vectors.shape[1]
    result = np.zeros([shape, 1], dtype=int)

    for i in range(len(matrices)):
        result += matrices[i].dot(vectors[i])
    return result


if __name__ == '__main__':
    with open('config.txt') as f:
        p = int(f.readline())
        n = int(f.readline())

    try:
        data = np.genfromtxt('data.csv', dtype='int', delimiter=' ')
        border = len(data) - p
        matrices = np.reshape(data[:border], [p, n, n])
        vectors = np.reshape(data[border:], [p, n, 1])

        result = get_multiplication(matrices, vectors)

        np.savetxt('result.csv', result, fmt='%d', delimiter=' ')

    except ValueError:
        print("Ошибка во входных данных! Проверьте данные и попробуйте снова.")
