import math


def pyramid(n):
    k = 1
    sumN = 1
    while (sumN < n):
        k += 1
        sumN += math.pow(k, 2)
    return k if sumN == n else "It is impossible"


n = input("Введите n: ")
try:
    n = int(n)
    print(pyramid(n)) if n > 0 else print("n должно быть больше 0!")

except ValueError:
    print("Введите натуральное число!")