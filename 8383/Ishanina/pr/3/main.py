import numpy as np

def standardization(x):
    # мат ожидание
    mat_exp = x.mean(axis=0)
    # СКО
    sko = x.std(axis=0)
    answer = (x - mat_exp) / sko
    return answer

x = np.fromfile('data1.csv', dtype = 'float', sep = ';')
np.savetxt("output.csv", standardization(x), fmt="%f")