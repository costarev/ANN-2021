import numpy as np
import pandas as pd

def solve(x):
    # x = np.unique(x, axis=0) # использование библиотечной функции
    for i in reversed(range(0, x.shape[0])):
        for j in reversed(range(i+1, x.shape[0])):
            if(np.array_equal(x[i], x[j])):
                x = np.delete(x, j, axis=0)
                x = np.array(x)
    output = pd.DataFrame(data=x.astype(int))
    output.to_csv('output.csv', sep = ';', header = False, index=False)

try:
    x = np.genfromtxt('input.csv',dtype = 'int',delimiter=';')
    solve(x)
except:
    print('Входные данные неверны')
