import numpy as np
arr = np.genfromtxt('inp.csv', delimiter=',')


def func(arr, x, y, n, fill):
    startX = x - int((n)/2)
    startY = y - int((n)/2)
    res = [[str(fill) for idx in range(n)] for jdx in range(n)]
    for i in range(startY, startY + n):
        for j in range(startX, startX + n):
            if(i >= 0 and j >= 0 and i < len(arr) and j < len(arr[0])):
                res[i - startY][j - startX] = arr[i][j]
    return (res)


def save(val):
    np.savetxt('outp.csv', np.asarray(val), fmt="%s,")


# res2 = func(arr, 1, 1, 5, -599)
# res3 = func(arr, 3, 4, 1, 'a')
# res4 = func(arr, 8, 8, 10, 3)
save(func(arr, 8, 8, 10, 3))
