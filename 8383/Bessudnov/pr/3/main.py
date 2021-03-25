import numpy

def uniqueRows(mtrx):
    # Выходная матрица
    res = numpy.array(mtrx[0], ndmin=2)
    # Проходим по всем строкам в исходной матрице
    for i in range(1, mtrx.shape[0]):
        # Проходим по всем строкам выходной матрицы
        for j in range(res.shape[0]):
            # Если встречаются одинаковые точки, то осуществляем выход
            if numpy.array_equal(mtrx[i], res[j]):
                break
            # Если дошли до конца, то добавляем строку в выходную матрицу
            if res.shape[0] - 1 == j: 
                res = numpy.append(res, numpy.array(mtrx[i], ndmin=2), axis=0)

    return res

matrix = numpy.loadtxt("test.txt")
rows = uniqueRows(matrix)
# uniqueRows = numpy.unique(matrix, axis=0)
numpy.savetxt("result.txt", rows, fmt = '%2.0d')