import numpy as np


def read_data(type_data):
    file_read = ""

    while not file_read:
        file_read = input('Введите название файла для чтения:')

    return np.genfromtxt(fname=file_read, dtype=type_data, delimiter=';')


def save_data(matr, type_data):
    file_save = ""
    format_ = '%i'

    while not file_save:
        file_save = input('Введите название файла для сохранения результата работы программы:')

    if (type_data == 'int'):
        format_ = '%i'

    elif (type_data == 'float'):
        format_ = '%f'

    np.savetxt(fname=file_save, X=matr, delimiter=';', fmt=format_, header='Результат работы программы:')


def unique_strings(matr):
    cur_row = 0

    print("Исходная матрица:")
    print(matr)

    while cur_row < matr.shape[0]:

        print("\nПроверка уникальности для строки: ")
        print(matr[cur_row])

        temp_matr = matr[~((matr - matr[cur_row]) == 0).all(axis=1)]

        if temp_matr.shape[0] + 1 == matr.shape[0]:
            print("\nСтрока уникальна!")
            cur_row += 1

        else:
            print("\nУдаление повторящийся строки!")
            print("Количество повторов строки равно: ")
            print(matr.shape[0] - temp_matr.shape[0])
            matr = temp_matr

    return matr


type_data = ""
while not type_data:
    type_data = input('Укажите тип данных элементов матрицы:')
matr = read_data(type_data)

if matr.ndim != 2:
    print("Не матрица!")
else:
    matr = unique_strings(matr)
    if (matr.shape[0] > 0):
        print("\nМатрица, состоящая только из уникальных строк:")
        print(matr)
    else:
        print("\nВ исходной матрице не обнаружены уникальные строки.")
    save_data(matr, type_data)
