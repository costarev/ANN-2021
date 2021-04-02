# Необходимо использовать модуль numpy
# Все данные должны считываться из файла в виде массива numpy
# Результаты необходимо сохранять в файл

# Задача 7
#
# Написать функцию, которая стандартизирует все значения тензор (отнять мат. ожидание и поделить на СКО)

import numpy

def func(arr):
    mean = arr.mean(axis=0)
    arr -= mean
    std = arr.std(axis=0)
    arr /= std
    return arr

arr = numpy.fromfile("numbers.csv", dtype='float', count=-1, sep=',')
# print(arr)
arr = func(arr)
# print(arr)
numpy.savetxt('res.csv', arr, fmt="%f")