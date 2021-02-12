from math import sqrt


def getStandardDeviation(v1, v2):
    d = 0
    for num1, num2 in zip(v1, v2):
        d += (num1 - num2) ** 2
    return sqrt(d / len(v1))


s1 = input("Введите первый вектор через пробел:\n").split(' ')
s2 = input("Введите второй вектор через пробел:\n").split(' ')

v1 = [int(num) for num in s1]
v2 = [int(num) for num in s2]
if len(v1) > len(v2):
    del v1[len(v2):]
elif len(v2) > len(v1):
    del v2[len(v1):]

print("Среднеквадратическое отклонение векторов: ", getStandardDeviation(v1, v2))