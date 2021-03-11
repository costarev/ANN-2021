import numpy


def mult(mat, vec, n ,p):
    ans = numpy.zeros([n, 1], dtype=int)
    for i in range(p):
        ans = ans + mat[i].dot(vec[i])
    return ans


try:
    data = numpy.genfromtxt('input.csv', dtype='int', delimiter=',')
except IndexError:
    print("Ошибка входных данных")

m = len(data)
n = int(len(data[0]))

if m % (n + 1) != 0:
    print("Ошибка входных данных")
else:
    p = int(m / (n + 1))
    matrix = numpy.reshape(data[:m - p], [p, n, n])
    vectors = numpy.reshape(data[m - p:], [p, n, 1])
    numpy.savetxt('output.csv', mult(matrix, vectors, n, p), fmt='%d', delimiter=',')