# Практическая 3
## Вариант 3

### Задача
Написать функцию, которая возвращает все уникальные строки матрицы.
____
Были написаны 2 функции:
 - ```python find_unique_str_np(file_read, file_save, dtype_='int', delim=';', console_log=False)``` - реализована с помощью стандартных возможностей *numpy*.
 - ```python naive_find_unique_str(file_read, file_save, dtype_='int', delim=';', console_log=False)``` - реализована с помощью обычного цикла по строкам матрицы.

В обеих функциях для считывания используется функция ```np.genfromtxt()```, а для вывода результата в файл ```np.savetxt()```.
