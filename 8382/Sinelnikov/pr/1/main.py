import numpy as np

def numberOfMultiplications(n):
    if n < 0 or not isinstance(n,int):
        print("Incorrect input")
        return
    counter = 0
    while n // 10 != 0:
        n = list(map(int,str(n)))
        n = np.asarray(n, dtype=np.int)
        n = np.prod(n)
        counter+=1

    return counter

print("Введите число: ")
n = int(input())
print("Число итераций: ", numberOfMultiplications(n))